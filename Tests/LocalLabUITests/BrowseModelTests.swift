import Foundation
import LocalLabCore
import Testing
@testable import LocalLabUI

/// Serves fixed pages; can be told to fail, like being offline.
final class FakeCatalog: CatalogClient, @unchecked Sendable {
    let pages: [[CatalogEntry]]
    let failing: Locked<Bool>
    let searches = Locked(0)

    init(pages: [[CatalogEntry]], failing: Bool = false) {
        self.pages = pages
        self.failing = Locked(failing)
    }

    func search(_ query: CatalogQuery, pageSize: Int) async throws -> CatalogPage {
        searches.withLock { $0 += 1 }
        if failing.withLock({ $0 }) { throw URLError(.notConnectedToInternet) }
        return CatalogPage(entries: pages.first ?? [], next: pages.count > 1 ? URL(string: "https://example.com/page/1") : nil)
    }

    func page(at url: URL) async throws -> CatalogPage {
        let index = Int(url.lastPathComponent) ?? 1
        return CatalogPage(entries: pages[index], next: index + 1 < pages.count ? URL(string: "https://example.com/page/\(index + 1)") : nil)
    }

    func details(repo: String) async throws -> CatalogDetails {
        CatalogDetails(repo: repo, revision: "r", weightBytes: 1_000_000_000, downloadBytes: 1_100_000_000, config: nil)
    }
}

@Suite("Browse screen model")
@MainActor
struct BrowseModelTests {
    let chat = CatalogEntry(repo: "mlx-community/Qwen3.5-9B-4bit", downloads: 10, task: "image-text-to-text",
                            modelType: "qwen3_5", bits: 4, license: "apache-2.0", parameters: ["U32": 8_952_741_888, "BF16": 457_071_088])
    let speech = CatalogEntry(repo: "mlx-community/whisper-large-v3-turbo", downloads: 9_000, task: "automatic-speech-recognition",
                              modelType: "whisper", license: "mit", parameters: ["F16": 800_000_000])
    let llama = CatalogEntry(repo: "mlx-community/Llama-3.3-70B-Instruct-4bit", downloads: 500, task: "text-generation",
                             modelType: "llama", bits: 4, license: "llama3.3", parameters: ["U32": 70_000_000_000, "BF16": 1_000_000_000])

    func make(_ workspace: Workspace, client: FakeCatalog, openInChat: @escaping @MainActor (String) -> Void = { _ in })
        -> BrowseModel {
        let queue = JobQueue(store: workspace.store, executor: FakeExecutor(), activity: NoActivity(),
                             calibrationURL: workspace.calibrationURL)
        let name = "locallab-browse-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name) ?? .standard
        defaults.removePersistentDomain(forName: name)
        return BrowseModel(
            client: client, cache: CatalogCache(directory: workspace.root.appendingPathComponent("cache")),
            store: workspace.store,
            installer: ModelInstaller(store: workspace.store, hub: FakeHubStub(), availableBytes: { 1 << 40 }),
            queue: queue, defaults: defaults, debounce: .milliseconds(1),
            detectHardware: { m2Max(availableGB: 19) },
            openInChat: openInChat
        )
    }

    @Test("Runnable models come first, graded; the rest say why not")
    func ordering() async {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        let browse = make(workspace, client: FakeCatalog(pages: [[speech, llama, chat]]))
        await browse.search()
        #expect(browse.rows.map(\.entry.repo) == [chat.repo, llama.repo, speech.repo])
        #expect(browse.rows[0].verdict.grade == .green)
        #expect(browse.rows[1].verdict.grade == .red, "70B doesn't fit 32 GB")
        guard case .notRunnable = browse.rows[2].verdict.grade else {
            Issue.record("whisper isn't runnable yet"); return
        }

        browse.query.sort = .downloads
        await browse.search()
        #expect(browse.rows.map(\.entry.repo) == [llama.repo, chat.repo, speech.repo],
                "by downloads within runnable — but runnable still first")
    }

    @Test("Offline, the last results are shown and labelled")
    func offline() async {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        let client = FakeCatalog(pages: [[chat]])
        let browse = make(workspace, client: client)
        await browse.search()
        client.failing.withLock { $0 = true }
        await browse.search()
        #expect(browse.rows.map(\.entry.repo) == [chat.repo])
        #expect(browse.offlineSince != nil)
    }

    @Test("More results load onto the end")
    func paging() async {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        let browse = make(workspace, client: FakeCatalog(pages: [[chat], [speech]]))
        await browse.search()
        #expect(browse.canLoadMore)
        await browse.loadMore()
        #expect(browse.rows.count == 2)
        #expect(!browse.canLoadMore)
    }

    @Test("A restrictive licence must be read before installing")
    func licenceGate() async {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        let browse = make(workspace, client: FakeCatalog(pages: [[llama, chat]]))
        await browse.search()
        let llamaRow = browse.rows.first { $0.entry.repo == llama.repo }
        let chatRow = browse.rows.first { $0.entry.repo == chat.repo }
        #expect(llamaRow.map { browse.installBlocker($0) } == "Read the licence first.")
        #expect(chatRow.map { browse.installBlocker($0) } ?? "x" == nil, "Apache 2.0 needs no gate")
        browse.acknowledge(llama.repo, true)
        #expect(llamaRow.map { browse.installBlocker($0) } ?? "x" == nil)
    }

    @Test("Opening a model refines its figures")
    func details() async {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        let other = CatalogEntry(repo: "someone/Custom-7B-4bit", task: "text-generation", modelType: "llama", bits: 4,
                                 parameters: ["U32": 7_000_000_000])
        let browse = make(workspace, client: FakeCatalog(pages: [[other]]))
        await browse.search()
        #expect(browse.rows.first?.verdict.isEstimate == true)
        browse.select(other.repo)
        for _ in 0 ..< 100 where browse.details[other.repo] == nil { try? await Task.sleep(for: .milliseconds(5)) }
        #expect(browse.rows.first?.verdict.weightBytes == 1_000_000_000)
    }

    @Test("An installed chat model opens in Chat, and Chat can use it even uncurated")
    func openInChat() async throws {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        let repo = "someone/Custom-7B-4bit"
        let directory = workspace.store.directory(forRepo: repo)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(#"{"model_type":"llama","num_hidden_layers":28,"num_attention_heads":24,"num_key_value_heads":8,"hidden_size":3072,"max_position_embeddings":131072,"quantization":{"bits":4,"group_size":64}}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        FileManager.default.createFile(atPath: workspace.store.installedMarker(forRepo: repo).path, contents: nil)
        var registry = InstallRegistry()
        registry.models[repo] = try JSONDecoder.iso.decode(InstalledModel.self, from: Data("""
        {"repo":"\(repo)","revision":"r","installedAt":"2026-09-18T00:00:00Z","sizeBytes":1800000000,
         "files":[{"path":"model.safetensors","size":1800000000,"checksum":null}]}
        """.utf8))
        try registry.save(workspace.store)

        let specs = InstalledChatModels.specs(in: workspace.store)
        #expect(specs.map(\.repo) == [repo])
        #expect(specs.first?.isCurated == false)

        // Chat offers it — Browse's "Runs in Chat" is a promise Chat keeps.
        let queue = JobQueue(store: workspace.store, executor: FakeExecutor(), activity: NoActivity(),
                             calibrationURL: workspace.calibrationURL)
        let chat = ChatModel(
            store: workspace.store,
            installer: ModelInstaller(store: workspace.store, hub: FakeHubStub(), availableBytes: { 1 << 40 }),
            queue: queue, backend: FakeChatBackend(),
            conversationStore: ConversationStore(directory: workspace.root.appendingPathComponent("conversations")),
            defaults: UserDefaults(suiteName: "locallab-browse-chat-\(UUID().uuidString)") ?? .standard,
            detectHardware: { m2Max(availableGB: 19) }
        )
        #expect(chat.installed.contains(repo))
        #expect(chat.current?.spec.repo == repo, "Smart Fit picks it: it's the only chat model installed")

        let opened = Locked<String?>(nil)
        let browse = make(workspace, client: FakeCatalog(pages: [[]]), openInChat: { chosen in opened.withLock { $0 = chosen } })
        browse.chat(with: repo)
        #expect(opened.withLock { $0 } == repo)
    }
}

extension JSONDecoder {
    static var iso: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
