import Foundation
import LocalLabCore
import Testing
@testable import LocalLabUI


/// Streams a scripted reply; records loads and ejects.
final class FakeChatBackend: ChatBackend, @unchecked Sendable {
    let reply: [String]
    let delay: Duration
    let loads = Locked<[URL]>([])
    let ejects = Locked(0)
    let lastTurns = Locked<[ChatTurn]>([])

    let resident: Int64
    let loaded = Locked(false)

    init(reply: [String] = ["Hello ", "**world**"], delay: Duration = .zero, resident: Int64 = 0) {
        self.reply = reply
        self.delay = delay
        self.resident = resident
    }

    let visionLoads = Locked<[Bool]>([])

    func load(directory: URL, vision: Bool) async throws {
        loads.withLock { $0.append(directory) }
        visionLoads.withLock { $0.append(vision) }
        loaded.withLock { $0 = true }
    }
    func eject() async {
        ejects.withLock { $0 += 1 }
        loaded.withLock { $0 = false }
    }
    func residentBytes() async -> Int64 { loaded.withLock { $0 } ? resident : 0 }

    func stream(_ turns: [ChatTurn], options: ChatOptions) -> AsyncThrowingStream<ChatEvent, Error> {
        lastTurns.withLock { $0 = turns }
        let reply = reply, delay = delay
        return AsyncThrowingStream { continuation in
            let task = Task {
                for piece in reply {
                    if delay > .zero { try? await Task.sleep(for: delay) }
                    if Task.isCancelled { continuation.finish(throwing: CancellationError()); return }
                    continuation.yield(.text(piece))
                }
                continuation.yield(.finished(ChatStats(promptTokens: 10, generatedTokens: 64,
                                                       promptSeconds: 0.1, generateSeconds: 2)))
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

@Suite("Chat screen model")
@MainActor
struct ChatModelTests {
    let small = "mlx-community/Qwen3.5-0.8B-4bit"
    let medium = "mlx-community/Qwen3.5-9B-4bit"

    func make(_ workspace: Workspace, backend: FakeChatBackend = FakeChatBackend(),
              executor: JobExecutor = FakeExecutor()) -> (ChatModel, JobQueue) {
        let queue = JobQueue(store: workspace.store, executor: executor, activity: NoActivity(),
                             calibrationURL: workspace.calibrationURL)
        let name = "locallab-chat-tests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name) ?? .standard
        defaults.removePersistentDomain(forName: name)
        let chat = ChatModel(
            store: workspace.store,
            installer: ModelInstaller(store: workspace.store, hub: FakeHubStub(), availableBytes: { 1 << 40 }),
            queue: queue, backend: backend,
            conversationStore: ConversationStore(directory: workspace.root.appendingPathComponent("conversations")),
            defaults: defaults,
            detectHardware: { m2Max(availableGB: 19) }
        )
        return (chat, queue)
    }

    func install(_ repo: String, in workspace: Workspace) {
        let directory = workspace.store.directory(forRepo: repo)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: workspace.store.installedMarker(forRepo: repo).path, contents: nil)
    }

    func waitUntilIdle(_ chat: ChatModel) async {
        for _ in 0 ..< 200 where chat.isGenerating { try? await Task.sleep(for: .milliseconds(10)) }
    }

    @Test("With nothing installed, it recommends models for this Mac and downloads nothing")
    func recommendsWhenEmpty() {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        let (chat, _) = make(workspace)
        #expect(chat.current == nil)
        #expect(chat.unavailableReason == "No chat model is installed yet.")
        #expect(!chat.recommendations.isEmpty)
        #expect(chat.installs.isEmpty)
    }

    @Test("Smart Fit picks the best installed model that runs well, and names a better one")
    func smartFitPick() {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install(small, in: workspace)
        let (chat, _) = make(workspace)
        #expect(chat.current?.spec.repo == small)
        #expect(chat.suggestion != nil, "a 32 GB Mac can run much better than 0.8B")

        install(medium, in: workspace)
        chat.refresh()
        #expect(chat.current?.spec.repo == medium, "installing a better model upgrades Smart Fit by itself")
    }

    @Test("A reply streams in, is saved, and its speed is measured for next time")
    func sendAndRecord() async throws {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install(small, in: workspace)
        let backend = FakeChatBackend()
        let (chat, queue) = make(workspace, backend: backend)
        chat.draft = "Say hello\nplease"
        chat.send()
        await waitUntilIdle(chat)

        #expect(chat.messages.map(\.text) == ["Say hello\nplease", "Hello **world**"])
        #expect(chat.messages.last?.stats?.generatedTokens == 64)
        #expect(chat.active?.title == "Say hello")
        #expect(chat.loadedRepo == small)
        #expect(backend.loads.withLock { $0 } == [workspace.store.directory(forRepo: small)])
        #expect(backend.lastTurns.withLock { $0 } == [ChatTurn(.user, "Say hello\nplease")])

        let saved = ConversationStore(directory: workspace.root.appendingPathComponent("conversations")).loadAll()
        #expect(saved.first?.messages.count == 2)
        #expect(queue.calibration.samples.contains { $0.engineID == ChatModelPicker.engineID })
        guard case .measured = chat.current?.speed else {
            Issue.record("after a reply, speed should be measured here")
            return
        }
    }

    @Test("The loaded model's own memory doesn't count against it: no false warning after a reply")
    func ownMemoryIsNotPressure() async {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install(medium, in: workspace)
        let free = Locked(13.0)
        let backend = FakeChatBackend(resident: 5 * 1_073_741_824)
        let queue = JobQueue(store: workspace.store, executor: FakeExecutor(), activity: NoActivity(),
                             calibrationURL: workspace.calibrationURL)
        let chat = ChatModel(
            store: workspace.store,
            installer: ModelInstaller(store: workspace.store, hub: FakeHubStub(), availableBytes: { 1 << 40 }),
            queue: queue, backend: backend,
            conversationStore: ConversationStore(directory: workspace.root.appendingPathComponent("conversations")),
            defaults: UserDefaults(suiteName: "locallab-own-\(UUID().uuidString)") ?? .standard,
            detectHardware: { m2Max(availableGB: free.withLock { $0 }) }
        )
        #expect(chat.current?.grade == .green)
        #expect(chat.current?.context == 32_768)

        chat.draft = "hi"
        chat.send()
        // Loading the model takes 5 GB of what was free.
        free.withLock { $0 = 8 }
        await waitUntilIdle(chat)
        #expect(chat.current?.grade == .green, "its own 5 GB is not someone else's")
        #expect(chat.current?.context == 32_768, "the context isn't cut mid-conversation")

        await chat.eject()
        free.withLock { $0 = 13 }
        chat.refresh()
        #expect(chat.residentBytes == 0)
        #expect(chat.current?.context == 32_768)
    }

    @Test("The idle-unload time is a setting that sticks; five minutes unless changed")
    func idleSetting() {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        let name = "locallab-idle-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name) ?? .standard
        defaults.removePersistentDomain(forName: name)
        func make() -> ChatModel {
            ChatModel(
                store: workspace.store,
                installer: ModelInstaller(store: workspace.store, hub: FakeHubStub(), availableBytes: { 1 << 40 }),
                queue: JobQueue(store: workspace.store, executor: FakeExecutor(), activity: NoActivity(),
                                calibrationURL: workspace.calibrationURL),
                backend: FakeChatBackend(),
                conversationStore: ConversationStore(directory: workspace.root.appendingPathComponent("conversations")),
                defaults: defaults, detectHardware: { m2Max(availableGB: 19) }
            )
        }
        #expect(make().idleUnloadMinutes == 5)
        let chat = make()
        chat.idleUnloadMinutes = 0
        #expect(make().idleUnloadMinutes == 0, "Never, remembered")
        chat.idleUnloadMinutes = 30
        #expect(make().idleUnloadMinutes == 30)
    }

    func installUncurated(_ repo: String, config: String, in workspace: Workspace) throws {
        let directory = workspace.store.directory(forRepo: repo)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(config.utf8).write(to: directory.appendingPathComponent("config.json"))
        FileManager.default.createFile(atPath: workspace.store.installedMarker(forRepo: repo).path, contents: nil)
        var registry = InstallRegistry.load(workspace.store)
        registry.models[repo] = try JSONDecoder.iso.decode(InstalledModel.self, from: Data("""
        {"repo":"\(repo)","revision":"r","installedAt":"2026-09-18T00:00:00Z","sizeBytes":4000000000,
         "files":[{"path":"model.safetensors","size":4000000000,"checksum":null}]}
        """.utf8))
        try registry.save(workspace.store)
    }

    @Test("An attached image picks a model that can see, loads its vision half, and travels with the message")
    func imageQuestion() async throws {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install(small, in: workspace)   // Qwen3.5 0.8B: sees images
        try installUncurated("someone/Llama-7B-4bit", config: #"{"model_type":"llama","num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"hidden_size":4096,"max_position_embeddings":8192}"#, in: workspace)
        let backend = FakeChatBackend(reply: ["A ", "boy."])
        let (chat, _) = make(workspace, backend: backend)
        #expect(chat.current?.spec.repo == "someone/Llama-7B-4bit", "text-only: the larger model")
        #expect(chat.canAttachImages)

        chat.attach([try workspace.makeImage()])
        #expect(chat.pendingAttachments.count == 1)
        #expect(chat.current?.spec.repo == small, "with an image: the one that can see")

        chat.send()
        await waitUntilIdle(chat)
        let sent = backend.lastTurns.withLock { $0 }
        #expect(sent.last?.content == "Describe this image.", "an image on its own is a question too")
        #expect(sent.last?.images.count == 1)
        #expect(backend.visionLoads.withLock { $0 } == [true])
        #expect(chat.messages.first?.attachments?.count == 1)
        #expect(chat.pendingAttachments.isEmpty)
        if let name = chat.messages.first?.attachments?.first {
            #expect(FileManager.default.fileExists(atPath: chat.attachmentURL(name).path), "copied into the conversation")
        }
        #expect(chat.loadedWithVision)
    }

    @Test("A model that can't see images is refused for an image conversation, and says why")
    func blindModel() throws {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install(small, in: workspace)
        try installUncurated("someone/Llama-7B-4bit", config: #"{"model_type":"llama","num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"hidden_size":4096,"max_position_embeddings":8192}"#, in: workspace)
        let (chat, _) = make(workspace)
        chat.select(.manual(repo: "someone/Llama-7B-4bit"))
        chat.attach([try workspace.makeImage()])
        #expect(chat.unavailableReason?.contains("can't see images") == true)
        #expect(!chat.canSend)
    }

    @Test("Only images can be attached")
    func notAnImage() throws {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install(small, in: workspace)
        let (chat, _) = make(workspace)
        let text = workspace.root.appendingPathComponent("notes.txt")
        try Data("hello".utf8).write(to: text)
        chat.attach([text])
        #expect(chat.pendingAttachments.isEmpty)
        #expect(chat.attachError?.contains("isn't an image") == true)
    }

    @Test("Stop keeps what arrived and marks the reply stopped")
    func stop() async {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install(small, in: workspace)
        let (chat, _) = make(workspace, backend: FakeChatBackend(reply: ["a", "b", "c", "d"], delay: .milliseconds(80)))
        chat.draft = "count"
        chat.send()
        try? await Task.sleep(for: .milliseconds(120))
        chat.stop()
        await waitUntilIdle(chat)
        #expect(chat.messages.last?.stopped == true)
        #expect((chat.messages.last?.text.count ?? 0) < 4)
    }

    @Test("A job gets the machine: the idle chat model is ejected before it starts")
    func handsMemoryToJobs() async throws {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install(small, in: workspace)
        let backend = FakeChatBackend()
        let (chat, queue) = make(workspace, backend: backend)
        chat.draft = "hi"
        chat.send()
        await waitUntilIdle(chat)
        #expect(chat.loadedRepo == small)

        let video = try await workspace.makeVideo()
        let probed = try await VideoIO.probe(video)
        queue.enqueue(title: "clip", kind: .upscale(UpscaleJobSpec(
            source: .video(probed), target: .scale(2), quality: .balanced, variant: .threeBInt8,
            outputURL: workspace.root.appendingPathComponent("out.mp4")
        )))
        for _ in 0 ..< 200 where queue.activeCount > 0 { try? await Task.sleep(for: .milliseconds(10)) }
        #expect(backend.ejects.withLock { $0 } >= 1)
        #expect(chat.engineState == .unloaded)
    }

    @Test("Choosing a model that isn't installed says so instead of failing later")
    func manualNotInstalled() {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install(small, in: workspace)
        let (chat, _) = make(workspace)
        chat.select(.manual(repo: medium))
        #expect(chat.unavailableReason == "The chosen model isn't installed.")
        #expect(!chat.canSend)
        chat.select(.smartFit)
        #expect(chat.current?.spec.repo == small)
    }

    @Test("A render choice sticks to the message and is remembered for the model")
    func renderMode() async {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install(small, in: workspace)
        let (chat, _) = make(workspace)
        chat.draft = "hi"
        chat.send()
        await waitUntilIdle(chat)
        guard let reply = chat.messages.last else { return }
        #expect(chat.renderMode(for: reply) == .auto)
        chat.setRenderMode(.raw, for: reply.id)
        #expect(chat.messages.last?.renderMode == .raw)
        let fresh = ChatMessage(role: .assistant, text: "x")
        #expect(chat.renderMode(for: fresh) == .raw, "remembered per model")
    }
}
