import Foundation
import ImageIO
import LocalLabCore
import Observation

/// The Chat screen (plan §5.12): opens ready, on the best installed model this Mac runs well
/// — or the one the user picked — and gives the memory back when asked, when idle, or when a
/// job needs the machine.
@MainActor
@Observable
public final class ChatModel {
    public enum Selection: Equatable, Sendable {
        /// Smart Fit: the best installed model this Mac runs well, re-chosen as things change.
        case smartFit
        case manual(repo: String)
    }

    public enum EngineState: Equatable {
        case unloaded
        case loading(repo: String)
        case loaded(repo: String)
        case failed(String)
    }

    public private(set) var conversations: [Conversation] = []
    public private(set) var activeID: UUID?
    public var draft = ""
    public private(set) var selection: Selection
    /// The model a send would use, graded for this Mac right now.
    public private(set) var current: ChatFit?
    /// Why nothing can be used, when nothing can.
    public private(set) var unavailableReason: String?
    /// A better model this Mac could run — named, never downloaded.
    public private(set) var suggestion: ChatFit?
    /// What to install when no chat model is installed yet.
    public private(set) var recommendations: [ChatFit] = []
    public private(set) var installed: Set<String> = []
    /// Installed chat models: the curated ones, and any other the Browse screen installed.
    public private(set) var installedSpecs: [ChatModelSpec] = []
    public private(set) var engineState: EngineState = .unloaded
    public private(set) var isGenerating = false
    public private(set) var errorMessage: String?
    public private(set) var installs: [String: Double] = [:]
    public private(set) var installErrors: [String: String] = [:]
    /// Context length chosen by the user; nil follows the model's default for this Mac.
    public private(set) var contextOverride: Int?
    public var thinking: Bool {
        didSet { defaults.set(thinking, forKey: Self.thinkingKey) }
    }

    private let store: ModelStore
    private let installer: ModelInstaller
    private let queue: JobQueue
    private let backend: ChatBackend
    private let conversationStore: ConversationStore
    private let defaults: UserDefaults
    private let detectHardware: @Sendable () -> HardwareProfile
    /// Set only by tests; the app follows `idleUnloadMinutes`.
    private let idleEvictionOverride: Duration?
    @ObservationIgnored private var generation: Task<Void, Never>?
    @ObservationIgnored private var idleTimer: Task<Void, Never>?

    static let selectionKey = "LocalLab.chatModel"
    static let thinkingKey = "LocalLab.chatThinking"
    static let renderModesKey = "LocalLab.chatRenderModes"
    static let idleUnloadKey = "LocalLab.chatIdleUnloadMinutes"
    public static let defaultIdleUnloadMinutes = 5
    /// The choices Settings offers; 0 is "never".
    public static let idleUnloadChoices = [1, 5, 15, 30, 60, 0]

    /// Minutes an unused model stays loaded before its memory is given back; 0 keeps it
    /// loaded until ejected or a job needs the machine. Unloading keeps the conversation —
    /// only the model's cached reading of it goes, so the next reply re-reads it once.
    public var idleUnloadMinutes: Int {
        didSet {
            defaults.set(idleUnloadMinutes, forKey: Self.idleUnloadKey)
            if loadedRepo != nil, !isGenerating { scheduleIdleEviction() }
        }
    }

    public init(
        store: ModelStore,
        installer: ModelInstaller,
        queue: JobQueue,
        backend: ChatBackend,
        conversationStore: ConversationStore = ConversationStore(),
        defaults: UserDefaults = .standard,
        idleEviction: Duration? = nil,
        detectHardware: @Sendable @escaping () -> HardwareProfile = { HardwareProfile.detect() }
    ) {
        self.store = store
        self.installer = installer
        self.queue = queue
        self.backend = backend
        self.conversationStore = conversationStore
        self.defaults = defaults
        self.idleEvictionOverride = idleEviction
        let savedMinutes = defaults.object(forKey: Self.idleUnloadKey) as? Int
        idleUnloadMinutes = savedMinutes ?? Self.defaultIdleUnloadMinutes
        self.detectHardware = detectHardware
        let saved = defaults.string(forKey: Self.selectionKey)
        selection = saved.map { .manual(repo: $0) } ?? .smartFit
        thinking = defaults.bool(forKey: Self.thinkingKey)
        conversations = conversationStore.loadAll()
        activeID = conversations.first?.id
        refresh()
        // A job gets the whole machine: an idle model goes, a streaming reply finishes first.
        queue.beforeEachJob { [weak self] in await self?.yieldToJob() }
    }

    // MARK: - Derived

    public var active: Conversation? {
        conversations.first { $0.id == activeID }
    }

    public var messages: [ChatMessage] { active?.messages ?? [] }

    /// Memory the loaded chat model holds. It counts as *available to chat*: otherwise
    /// re-grading after a reply measures free memory with the model's own 5 GB missing, calls
    /// it "close some apps", and cuts the context from 32k to 4k.
    public private(set) var residentBytes: Int64 = 0

    public var picker: ChatModelPicker {
        let hardware = detectHardware()
        return ChatModelPicker(
            hardware: hardware.withAvailableMemory(hardware.availableMemoryBytes + residentBytes),
            calibration: queue.calibration
        )
    }

    /// Every curated model, and every other installed one, graded for this Mac — installed
    /// first.
    public var choices: [ChatFit] {
        let picker = self.picker
        let extra = installedSpecs.filter { !$0.isCurated }
        return (ChatCatalog.models + extra).map { picker.fit($0) }.sorted { a, b in
            let ai = installed.contains(a.spec.repo), bi = installed.contains(b.spec.repo)
            if ai != bi { return ai }
            return a.spec.qualityScore > b.spec.qualityScore
        }
    }

    public var canSend: Bool {
        guard !isGenerating, current != nil, unavailableReason == nil else { return false }
        return !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !pendingAttachments.isEmpty
    }

    // MARK: - Images

    /// Images attached to the message being written: names in the conversation store.
    public private(set) var pendingAttachments: [String] = []
    public private(set) var attachError: String?

    /// This conversation needs a model that can see: it has images, or one is about to be sent.
    public var needsVision: Bool {
        !pendingAttachments.isEmpty || messages.contains { !($0.attachments ?? []).isEmpty }
    }

    /// Some installed model can see images — whether attaching is possible at all.
    public var canAttachImages: Bool { installedSpecs.contains { $0.seesImages } }

    public func attachmentURL(_ name: String) -> URL { conversationStore.attachmentURL(name) }

    /// Add images to the message being written. Copied into the conversation store, so the
    /// conversation keeps them. Anything that isn't a readable image is refused.
    public func attach(_ urls: [URL]) {
        attachError = nil
        for url in urls {
            let access = url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() } }
            guard CGImageSourceCreateWithURL(url as CFURL, nil).flatMap({ CGImageSourceGetCount($0) > 0 ? $0 : nil }) != nil else {
                attachError = "\(url.lastPathComponent) isn't an image LocalLab can read."
                continue
            }
            do {
                pendingAttachments.append(try conversationStore.importAttachment(url))
            } catch {
                attachError = "Couldn't attach \(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
        refresh()
    }

    public func removeAttachment(_ name: String) {
        pendingAttachments.removeAll { $0 == name }
        try? FileManager.default.removeItem(at: conversationStore.attachmentURL(name))
        refresh()
    }

    public var loadedRepo: String? {
        if case .loaded(let repo) = engineState { return repo }
        return nil
    }

    // MARK: - Choosing a model

    /// Re-resolve the model for this Mac as it is now: after launch, an install, a job, or a
    /// change of selection. Smart Fit picks up a better installed model by itself.
    public func refresh() {
        installedSpecs = InstalledChatModels.specs(in: store)
        installed = Set(installedSpecs.map(\.repo))
        let picker = self.picker
        unavailableReason = nil
        switch selection {
        case .smartFit:
            let vision = needsVision
            current = picker.smartFit(candidates: installedSpecs, vision: vision).map { fit in
                contextOverride.map { picker.fit(fit.spec, context: $0, vision: vision) } ?? fit
            }
            if current == nil {
                unavailableReason = installed.isEmpty
                    ? "No chat model is installed yet."
                    : vision && !canAttachImages
                        ? "None of the installed chat models can see images. Browse has ones that can."
                        : "None of the installed chat models runs well on this Mac right now."
            }
        case .manual(let repo):
            guard let spec = installedSpecs.first(where: { $0.repo == repo }) else {
                current = nil
                unavailableReason = "The chosen model isn't installed."
                break
            }
            let fit = picker.fit(spec, context: contextOverride, vision: needsVision)
            current = fit
            if case .red(let because) = fit.grade {
                unavailableReason = "\(spec.displayName) won't run here: \(because)."
            } else if needsVision && !spec.seesImages {
                unavailableReason = "\(spec.displayName) can't see images. Choose Smart Fit, or a model that can."
            }
        }
        recommendations = installed.isEmpty ? picker.recommendations(limit: 2) : []
        suggestion = installed.isEmpty ? nil : picker.betterThan(current, installed: installed)
    }

    public func select(_ selection: Selection) {
        self.selection = selection
        contextOverride = nil
        switch selection {
        case .smartFit: defaults.removeObject(forKey: Self.selectionKey)
        case .manual(let repo): defaults.set(repo, forKey: Self.selectionKey)
        }
        refresh()
    }

    public func setContext(_ tokens: Int?) {
        contextOverride = tokens
        refresh()
    }

    // MARK: - Conversations

    public func newConversation() {
        guard !isGenerating else { return }
        if let active, active.messages.isEmpty { return }
        let conversation = Conversation()
        conversations.insert(conversation, at: 0)
        activeID = conversation.id
    }

    public func open(_ id: UUID) {
        guard !isGenerating else { return }
        activeID = id
        refresh()
    }

    public func delete(_ id: UUID) {
        guard !(isGenerating && id == activeID) else { return }
        conversations.removeAll { $0.id == id }
        conversationStore.delete(id)
        if activeID == id { activeID = conversations.first?.id }
    }

    // MARK: - Sending

    public func send() {
        let typed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSend, let fit = current else { return }
        let images = pendingAttachments
        // An image on its own is a question too.
        let prompt = typed.isEmpty ? "Describe this image." : typed
        let vision = needsVision
        draft = ""
        pendingAttachments = []
        errorMessage = nil
        if active == nil { newConversation() }
        guard let conversationID = activeID else { return }

        update(conversationID) { conversation in
            if conversation.messages.isEmpty { conversation.title = Conversation.title(forFirstPrompt: prompt) }
            conversation.messages.append(ChatMessage(role: .user, text: prompt,
                                                     attachments: images.isEmpty ? nil : images))
            conversation.messages.append(ChatMessage(role: .assistant, text: ""))
            conversation.modelRepo = fit.spec.repo
        }
        let store = conversationStore
        let turns = (conversation(conversationID)?.messages.dropLast() ?? []).map {
            ChatTurn($0.role == .user ? .user : .assistant, $0.text,
                     images: ($0.attachments ?? []).map(store.attachmentURL))
        }
        let thinks = thinking && fit.spec.thinks
        // Reasoning runs long — thousands of tokens before the answer — so a thinking reply
        // gets more room, within half the context.
        let options = ChatOptions(maxTokens: min(thinks ? 8_192 : 2_048, fit.context / 2),
                                  contextTokens: fit.context, thinking: thinks)
        isGenerating = true
        idleTimer?.cancel()

        generation = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.ensureLoaded(fit.spec, vision: vision || fit.spec.needsVisionLoad)
                for try await event in self.backend.stream(Array(turns), options: options) {
                    switch event {
                    case .text(let text):
                        self.updateLastMessage(conversationID) { $0.text += text }
                    case .trimmed(let dropped):
                        self.errorMessage = "The oldest \(dropped) message\(dropped == 1 ? " was" : "s were") left out to fit the \(ChatModelPicker.contextLabel(fit.context)) context."
                    case .finished(let stats):
                        self.updateLastMessage(conversationID) { $0.stats = stats }
                        self.residentBytes = await self.backend.residentBytes()
                        self.record(stats, for: fit.spec, vision: vision || fit.spec.needsVisionLoad)
                    }
                }
                // A stopped stream just ends — it doesn't throw — so check here.
                if Task.isCancelled { self.updateLastMessage(conversationID) { $0.stopped = true } }
            } catch is CancellationError {
                self.updateLastMessage(conversationID) { $0.stopped = true }
            } catch {
                self.errorMessage = error.localizedDescription
                self.updateLastMessage(conversationID) { $0.stopped = true }
            }
            self.isGenerating = false
            self.generation = nil
            self.save(conversationID)
            self.scheduleIdleEviction()
        }
    }

    public func stop() {
        generation?.cancel()
    }

    public func setRenderMode(_ mode: RenderMode?, for messageID: UUID) {
        guard let conversationID = activeID else { return }
        update(conversationID) { conversation in
            guard let index = conversation.messages.firstIndex(where: { $0.id == messageID }) else { return }
            conversation.messages[index].renderMode = mode
        }
        save(conversationID)
        // Remembered per model: a model's output shape is stable (plan §5.5).
        if let mode, let repo = active?.modelRepo {
            var modes = defaults.dictionary(forKey: Self.renderModesKey) as? [String: String] ?? [:]
            modes[repo] = mode.rawValue
            defaults.set(modes, forKey: Self.renderModesKey)
        }
    }

    /// The renderer a message uses: its own override, else the last one chosen for its model.
    public func renderMode(for message: ChatMessage) -> RenderMode {
        if let mode = message.renderMode { return mode }
        guard let repo = active?.modelRepo,
              let raw = (defaults.dictionary(forKey: Self.renderModesKey) as? [String: String])?[repo] else {
            return .auto
        }
        return RenderMode(rawValue: raw) ?? .auto
    }

    // MARK: - Memory

    public func eject() async {
        guard !isGenerating else { return }
        idleTimer?.cancel()
        await backend.eject()
        engineState = .unloaded
        loadedWithVision = false
        residentBytes = 0
        refresh()
    }

    /// Before a job: let a streaming reply finish, then give the memory back.
    func yieldToJob() async {
        while isGenerating { try? await Task.sleep(for: .milliseconds(100)) }
        if loadedRepo != nil { await eject() }
    }

    /// Whether the loaded model carries its vision half.
    public private(set) var loadedWithVision = false

    private func ensureLoaded(_ spec: ChatModelSpec, vision: Bool) async throws {
        if loadedRepo == spec.repo, loadedWithVision == vision { return }
        engineState = .loading(repo: spec.repo)
        do {
            try await backend.load(directory: store.directory(forRepo: spec.repo), vision: vision)
            loadedWithVision = vision
            engineState = .loaded(repo: spec.repo)
            residentBytes = await backend.residentBytes()
        } catch {
            engineState = .failed(error.localizedDescription)
            throw error
        }
    }

    private func scheduleIdleEviction() {
        idleTimer?.cancel()
        let delay: Duration
        if let override = idleEvictionOverride {
            delay = override
        } else if idleUnloadMinutes > 0 {
            delay = .seconds(idleUnloadMinutes * 60)
        } else {
            return
        }
        idleTimer = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled, let self, !self.isGenerating else { return }
            await self.eject()
        }
    }


    /// A reply's generation speed, per byte of weights read, so it prices every model here.
    /// Short replies are skipped: their fixed costs would dominate.
    private func record(_ stats: ChatStats, for spec: ChatModelSpec, vision: Bool) {
        guard stats.generatedTokens >= 32, stats.generateSeconds > 0 else { return }
        let machine = detectHardware().machineKey
        queue.recordMeasurements([CalibrationSample(
            engineID: ChatModelPicker.engineID, phase: vision ? ChatModelPicker.visionPhase : "decode",
            workUnits: Double(stats.generatedTokens) * ChatModelPicker.gbReadPerToken(spec),
            seconds: stats.generateSeconds, peakBytes: 0, machine: machine,
            note: "\(spec.displayName), \(stats.generatedTokens) tokens"
        )])
        refresh()
    }

    // MARK: - Installing

    public func install(_ spec: ChatModelSpec) async {
        guard installs[spec.repo] == nil else { return }
        installs[spec.repo] = 0
        installErrors[spec.repo] = nil
        do {
            try await installer.install(spec.repo) { progress in
                Task { @MainActor [weak self] in
                    guard let self, self.installs[spec.repo] != nil else { return }
                    self.installs[spec.repo] = progress.fraction
                }
            }
        } catch is CancellationError {
        } catch {
            installErrors[spec.repo] = String(describing: error)
        }
        installs[spec.repo] = nil
        refresh()
    }

    public func cancelInstall(_ spec: ChatModelSpec) async {
        await installer.cancel(spec.repo)
        installs[spec.repo] = nil
    }

    // MARK: - Storage

    private func conversation(_ id: UUID) -> Conversation? {
        conversations.first { $0.id == id }
    }

    private func update(_ id: UUID, _ change: (inout Conversation) -> Void) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        change(&conversations[index])
        conversations[index].updatedAt = Date()
    }

    private func updateLastMessage(_ id: UUID, _ change: (inout ChatMessage) -> Void) {
        guard let index = conversations.firstIndex(where: { $0.id == id }),
              !conversations[index].messages.isEmpty else { return }
        change(&conversations[index].messages[conversations[index].messages.count - 1])
    }

    private func save(_ id: UUID) {
        guard let conversation = conversation(id) else { return }
        try? conversationStore.save(conversation)
    }
}
