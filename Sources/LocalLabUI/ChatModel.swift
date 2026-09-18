import Foundation
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
    private let idleEviction: Duration
    @ObservationIgnored private var generation: Task<Void, Never>?
    @ObservationIgnored private var idleTimer: Task<Void, Never>?

    static let selectionKey = "LocalLab.chatModel"
    static let thinkingKey = "LocalLab.chatThinking"
    static let renderModesKey = "LocalLab.chatRenderModes"

    public init(
        store: ModelStore,
        installer: ModelInstaller,
        queue: JobQueue,
        backend: ChatBackend,
        conversationStore: ConversationStore = ConversationStore(),
        defaults: UserDefaults = .standard,
        idleEviction: Duration = .seconds(300),
        detectHardware: @Sendable @escaping () -> HardwareProfile = { HardwareProfile.detect() }
    ) {
        self.store = store
        self.installer = installer
        self.queue = queue
        self.backend = backend
        self.conversationStore = conversationStore
        self.defaults = defaults
        self.idleEviction = idleEviction
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

    public var picker: ChatModelPicker {
        ChatModelPicker(hardware: detectHardware(), calibration: queue.calibration)
    }

    /// Every catalog model graded for this Mac, installed first.
    public var choices: [ChatFit] {
        let picker = self.picker
        return ChatCatalog.models.map { picker.fit($0) }.sorted { a, b in
            let ai = installed.contains(a.spec.repo), bi = installed.contains(b.spec.repo)
            if ai != bi { return ai }
            return a.spec.qualityScore > b.spec.qualityScore
        }
    }

    public var canSend: Bool {
        guard !isGenerating, current != nil, unavailableReason == nil else { return false }
        return !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var loadedRepo: String? {
        if case .loaded(let repo) = engineState { return repo }
        return nil
    }

    // MARK: - Choosing a model

    /// Re-resolve the model for this Mac as it is now: after launch, an install, a job, or a
    /// change of selection. Smart Fit picks up a better installed model by itself.
    public func refresh() {
        installed = Set(ChatCatalog.models.map(\.repo).filter { store.isInstalled(repo: $0) })
        let picker = self.picker
        unavailableReason = nil
        switch selection {
        case .smartFit:
            current = picker.smartFit(installed: installed).map { fit in
                contextOverride.map { picker.fit(fit.spec, context: $0) } ?? fit
            }
            if current == nil {
                unavailableReason = installed.isEmpty
                    ? "No chat model is installed yet."
                    : "None of the installed chat models runs well on this Mac right now."
            }
        case .manual(let repo):
            guard let spec = ChatCatalog.spec(forRepo: repo), installed.contains(repo) else {
                current = nil
                unavailableReason = "The chosen model isn't installed."
                break
            }
            let fit = picker.fit(spec, context: contextOverride)
            current = fit
            if case .red(let because) = fit.grade {
                unavailableReason = "\(spec.displayName) won't run here: \(because)."
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
    }

    public func delete(_ id: UUID) {
        guard !(isGenerating && id == activeID) else { return }
        conversations.removeAll { $0.id == id }
        conversationStore.delete(id)
        if activeID == id { activeID = conversations.first?.id }
    }

    // MARK: - Sending

    public func send() {
        let prompt = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSend, let fit = current else { return }
        draft = ""
        errorMessage = nil
        if active == nil { newConversation() }
        guard let conversationID = activeID else { return }

        update(conversationID) { conversation in
            if conversation.messages.isEmpty { conversation.title = Conversation.title(forFirstPrompt: prompt) }
            conversation.messages.append(ChatMessage(role: .user, text: prompt))
            conversation.messages.append(ChatMessage(role: .assistant, text: ""))
            conversation.modelRepo = fit.spec.repo
        }
        let turns = (conversation(conversationID)?.messages.dropLast() ?? []).map {
            ChatTurn($0.role == .user ? .user : .assistant, $0.text)
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
                try await self.ensureLoaded(fit.spec)
                for try await event in self.backend.stream(Array(turns), options: options) {
                    switch event {
                    case .text(let text):
                        self.updateLastMessage(conversationID) { $0.text += text }
                    case .trimmed(let dropped):
                        self.errorMessage = "The oldest \(dropped) message\(dropped == 1 ? " was" : "s were") left out to fit the \(ChatModelPicker.contextLabel(fit.context)) context."
                    case .finished(let stats):
                        self.updateLastMessage(conversationID) { $0.stats = stats }
                        self.record(stats, for: fit.spec)
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
    }

    /// Before a job: let a streaming reply finish, then give the memory back.
    func yieldToJob() async {
        while isGenerating { try? await Task.sleep(for: .milliseconds(100)) }
        if loadedRepo != nil { await eject() }
    }

    private func ensureLoaded(_ spec: ChatModelSpec) async throws {
        if loadedRepo == spec.repo { return }
        engineState = .loading(repo: spec.repo)
        do {
            try await backend.load(directory: store.directory(forRepo: spec.repo))
            engineState = .loaded(repo: spec.repo)
        } catch {
            engineState = .failed(error.localizedDescription)
            throw error
        }
    }

    private func scheduleIdleEviction() {
        idleTimer?.cancel()
        let delay = idleEviction
        idleTimer = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled, let self, !self.isGenerating else { return }
            await self.eject()
        }
    }

    /// A reply's generation speed, per byte of weights read, so it prices every model here.
    /// Short replies are skipped: their fixed costs would dominate.
    private func record(_ stats: ChatStats, for spec: ChatModelSpec) {
        guard stats.generatedTokens >= 32, stats.generateSeconds > 0 else { return }
        let machine = detectHardware().machineKey
        queue.recordMeasurements([CalibrationSample(
            engineID: ChatModelPicker.engineID, phase: "decode",
            workUnits: Double(stats.generatedTokens) * spec.activeWeightBytes / 1e9,
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
