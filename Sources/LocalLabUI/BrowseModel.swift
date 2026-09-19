import Foundation
import LocalLabCore
import Observation

/// Browse (plan §5.2, §9.1): MLX models from Hugging Face, each graded by Smart Fit for this
/// Mac, runnable ones first. Installs only when asked, and only after a restrictive licence
/// has been read.
@MainActor
@Observable
public final class BrowseModel {
    public struct Row: Identifiable, Equatable {
        public let entry: CatalogEntry
        public let verdict: ModelVerdict
        public var id: String { entry.repo }
    }

    public var query: CatalogQuery {
        didSet { if query != oldValue { scheduleSearch() } }
    }
    public private(set) var rows: [Row] = []
    public private(set) var isLoading = false
    public private(set) var errorMessage: String?
    /// Set when showing cached results because the search couldn't reach Hugging Face.
    public private(set) var offlineSince: Date?
    public private(set) var canLoadMore = false
    public private(set) var selected: String?
    public private(set) var details: [String: CatalogDetails] = [:]
    public private(set) var detailsError: String?
    public private(set) var installs: [String: Double] = [:]
    public private(set) var installErrors: [String: String] = [:]
    public private(set) var installed: Set<String> = []

    private let client: CatalogClient
    private let cache: CatalogCache
    private let store: ModelStore
    private let installer: ModelInstaller
    private let queue: JobQueue
    private let defaults: UserDefaults
    private let detectHardware: @Sendable () -> HardwareProfile
    private let libraryChanged: @MainActor () async -> Void
    private let openInChat: @MainActor (String) -> Void
    private let openCreateImage: @MainActor () -> Void
    private let debounce: Duration
    @ObservationIgnored private var entries: [CatalogEntry] = []
    @ObservationIgnored private var nextPage: URL?
    @ObservationIgnored private var searchTask: Task<Void, Never>?
    @ObservationIgnored private var generation = 0

    static let acknowledgedKey = "LocalLab.acknowledgedLicenses"
    static let pageSize = 50

    public init(
        client: CatalogClient = HuggingFaceCatalog(),
        cache: CatalogCache = CatalogCache(),
        store: ModelStore,
        installer: ModelInstaller,
        queue: JobQueue,
        defaults: UserDefaults = .standard,
        debounce: Duration = .milliseconds(350),
        detectHardware: @Sendable @escaping () -> HardwareProfile = { HardwareProfile.detect() },
        libraryChanged: @escaping @MainActor () async -> Void = {},
        openInChat: @escaping @MainActor (String) -> Void = { _ in },
        openCreateImage: @escaping @MainActor () -> Void = {}
    ) {
        self.client = client
        self.cache = cache
        self.store = store
        self.installer = installer
        self.queue = queue
        self.defaults = defaults
        self.debounce = debounce
        self.detectHardware = detectHardware
        self.libraryChanged = libraryChanged
        self.openInChat = openInChat
        self.openCreateImage = openCreateImage
        query = CatalogQuery()
        refreshInstalled()
    }

    public var selectedRow: Row? { rows.first { $0.entry.repo == selected } }

    // MARK: - Searching

    /// Search now — on appear, or when a search is retried. Supersedes a search that was
    /// waiting out the typing pause.
    public func search() async {
        searchTask?.cancel()
        await performSearch()
    }

    /// The search itself. Never cancels anything: it runs *inside* the scheduled task, and
    /// cancelling that task from here cancelled its own request — every filter change showed
    /// "Couldn't reach Hugging Face: cancelled".
    private func performSearch() async {
        generation += 1
        let mine = generation
        isLoading = true
        errorMessage = nil
        defer { if mine == generation { isLoading = false } }
        do {
            let page = try await client.search(query, pageSize: Self.pageSize)
            guard mine == generation else { return }
            entries = page.entries
            nextPage = page.next
            offlineSince = nil
            cache.save(entries, for: query)
        } catch {
            // Superseded by a newer search: not a failure, and not worth a word.
            guard mine == generation, !Self.isCancellation(error) else { return }
            if let cached = cache.load(query) {
                entries = cached.entries
                offlineSince = cached.fetched
            } else {
                entries = []
                errorMessage = "Couldn't reach Hugging Face: \(error.localizedDescription)"
            }
            nextPage = nil
        }
        regrade()
    }

    public func loadMore() async {
        guard let next = nextPage, !isLoading else { return }
        let mine = generation
        isLoading = true
        defer { if mine == generation { isLoading = false } }
        guard let page = try? await client.page(at: next), mine == generation else { return }
        let known = Set(entries.map(\.repo))
        entries += page.entries.filter { !known.contains($0.repo) }
        nextPage = page.next
        regrade()
    }

    private func scheduleSearch() {
        searchTask?.cancel()
        let delay = debounce
        searchTask = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            await self?.performSearch()
        }
    }

    /// Cancellation arrives as Swift's `CancellationError` or, from `URLSession`, as
    /// `URLError.cancelled`.
    static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        return (error as? URLError)?.code == .cancelled
    }

    /// Grade every loaded model for this Mac as it is now, and order them: runnable first
    /// always; within that, by Smart Fit for "Best fit", else in the search's own order.
    public func regrade() {
        let hardware = detectHardware()
        let calibration = queue.calibration
        let graded = entries.map { entry in
            (entry, ModelGrader.verdict(for: entry, details: details[entry.repo], hardware: hardware, calibration: calibration))
        }
        let ordered: [(CatalogEntry, ModelVerdict)]
        if query.sort == .bestFit {
            ordered = graded.sorted(by: ModelGrader.bestFitOrder)
        } else {
            ordered = graded.filter { $0.1.runner.isRunnable } + graded.filter { !$0.1.runner.isRunnable }
        }
        rows = ordered.map { Row(entry: $0.0, verdict: $0.1) }
        canLoadMore = nextPage != nil
        refreshInstalled()
    }

    // MARK: - One model

    public func select(_ repo: String?) {
        selected = repo
        detailsError = nil
        guard let repo, details[repo] == nil else { return }
        Task { [weak self] in
            guard let self else { return }
            do {
                let fetched = try await self.client.details(repo: repo)
                self.details[repo] = fetched
                self.regrade()
            } catch {
                if self.selected == repo { self.detailsError = error.localizedDescription }
            }
        }
    }

    public func isAcknowledged(_ repo: String) -> Bool {
        (defaults.stringArray(forKey: Self.acknowledgedKey) ?? []).contains(repo)
    }

    public func acknowledge(_ repo: String, _ yes: Bool) {
        var list = Set(defaults.stringArray(forKey: Self.acknowledgedKey) ?? [])
        if yes { list.insert(repo) } else { list.remove(repo) }
        defaults.set(Array(list).sorted(), forKey: Self.acknowledgedKey)
    }

    /// Why Install is unavailable for a model, or nil.
    public func installBlocker(_ row: Row) -> String? {
        if installed.contains(row.entry.repo) { return "Installed." }
        if installs[row.entry.repo] != nil { return "Downloading…" }
        if LicenseInfo.needsAcknowledgement(row.entry.license), !isAcknowledged(row.entry.repo) {
            return "Read the licence first."
        }
        if case .red = row.verdict.grade { return nil }  // Allowed: a user may want it for another Mac.
        return nil
    }

    public func install(_ row: Row) async {
        let repo = row.entry.repo
        guard installBlocker(row) == nil else { return }
        installs[repo] = 0
        installErrors[repo] = nil
        let manifest = ComponentManifest.known(for: repo)
        do {
            try await installer.install(repo, manifest: manifest) { progress in
                Task { @MainActor [weak self] in
                    guard let self, self.installs[repo] != nil else { return }
                    self.installs[repo] = progress.fraction
                }
            }
        } catch is CancellationError {
        } catch {
            installErrors[repo] = String(describing: error)
        }
        installs[repo] = nil
        refreshInstalled()
        await libraryChanged()
    }

    public func cancelInstall(_ repo: String) async {
        await installer.cancel(repo)
        installs[repo] = nil
    }

    public func uninstall(_ repo: String) async {
        try? await installer.uninstall(repo)
        refreshInstalled()
        await libraryChanged()
    }

    public func chat(with repo: String) {
        openInChat(repo)
    }

    public func createImage() {
        openCreateImage()
    }

    public func refreshInstalled() {
        installed = Set(entries.map(\.repo).filter { store.isInstalled(repo: $0) })
    }

    public func huggingFaceURL(_ repo: String) -> URL? {
        URL(string: "https://huggingface.co/\(repo)")
    }
}
