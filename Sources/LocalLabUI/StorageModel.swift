import AppKit
import Foundation
import LocalLabCore
import Observation

/// Settings → Storage (plan §5.4): where the library is, what's in it, and moving it —
/// verified before anything is deleted.
@MainActor
@Observable
public final class StorageModel {
    public struct ModelRow: Identifiable, Equatable, Sendable {
        public let repo: String
        public let name: String
        public let kind: String
        public let bytes: Int64
        public var id: String { repo }
    }

    public enum MoveState: Equatable {
        case idle
        /// Ready to move, waiting for the user to confirm.
        case confirmMove(LibraryMover.Plan)
        /// The chosen folder already holds a library: switch to it, move nothing.
        case confirmUse(URL)
        case moving(LibraryMover.Progress)
        case failed(String)
        /// Done — LocalLab restarts to use the library at this location.
        case finished(URL)
    }

    public private(set) var location: URL
    public private(set) var isCustom: Bool
    /// The chosen library, when its drive isn't connected and the default is in use.
    public private(set) var unavailable: URL?
    public private(set) var pendingMove: URL?
    public private(set) var freeBytes: Int64 = 0
    public private(set) var models: [ModelRow] = []
    public private(set) var outputsBytes: Int64 = 0
    public private(set) var downloadsBytes: Int64 = 0
    public private(set) var jobsBytes: Int64 = 0
    public private(set) var isMeasuring = false
    public private(set) var moveState: MoveState = .idle
    public private(set) var deleteError: String?

    private let store: ModelStore
    private let installer: ModelInstaller
    private let queue: JobQueue
    private let chat: ChatModel?
    private let defaults: UserDefaults
    private let relaunch: @MainActor () -> Void
    /// Tell the other screens the library changed under them — a model deleted.
    private let libraryChanged: @MainActor () async -> Void
    /// Why the library can't move right now (a job running, an install in progress).
    private let busyReason: @MainActor () -> String?
    @ObservationIgnored private var moveTask: Task<Void, Never>?

    public init(
        resolution: LibraryLocation.Resolution,
        installer: ModelInstaller,
        queue: JobQueue,
        chat: ChatModel?,
        defaults: UserDefaults = .standard,
        busyReason: @escaping @MainActor () -> String? = { nil },
        libraryChanged: @escaping @MainActor () async -> Void = {},
        relaunch: @escaping @MainActor () -> Void = { Relaunch.now() }
    ) {
        store = resolution.store
        location = resolution.store.baseDirectory
        isCustom = resolution.isCustom
        unavailable = resolution.unavailable
        self.installer = installer
        self.queue = queue
        self.chat = chat
        self.defaults = defaults
        self.busyReason = busyReason
        self.libraryChanged = libraryChanged
        self.relaunch = relaunch
        pendingMove = LibraryLocation.pendingMove(defaults: defaults)
    }

    public var totalBytes: Int64 {
        models.reduce(0) { $0 + $1.bytes } + outputsBytes + downloadsBytes + jobsBytes
    }

    public var displayLocation: String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return location.path.hasPrefix(home) ? "~" + location.path.dropFirst(home.count) : location.path
    }

    public var volumeName: String { LibraryLocation.displayName(location) }

    /// Why the location can't change now, or nil.
    public var moveBlocker: String? {
        if case .moving = moveState { return "A move is already running." }
        if queue.isRunningJob { return "A job is running. Pause it or let it finish first." }
        if chat?.isGenerating == true { return "Chat is answering. Let it finish first." }
        if chat?.installs.isEmpty == false { return "A model is downloading. Let it finish or pause it first." }
        return busyReason()
    }

    // MARK: - Sizes

    public func refresh() async {
        isMeasuring = true
        let store = self.store
        let result = await Task.detached(priority: .utility) { Self.measure(store) }.value
        models = result.models
        outputsBytes = result.outputs
        downloadsBytes = result.downloads
        jobsBytes = result.jobs
        freeBytes = store.root.availableBytes()
        pendingMove = LibraryLocation.pendingMove(defaults: defaults)
        isMeasuring = false
    }

    nonisolated static func measure(_ store: ModelStore) -> (models: [ModelRow], outputs: Int64, downloads: Int64, jobs: Int64) {
        let registry = InstallRegistry.load(store)
        let slugs = (try? FileManager.default.contentsOfDirectory(atPath: store.modelsDirectory.path)) ?? []
        let rows = slugs.compactMap { slug -> ModelRow? in
            let repo = registry.models.keys.first { ModelStore.slug(for: $0) == slug }
                ?? slug.replacingOccurrences(of: "--", with: "/", options: [], range: slug.range(of: "--"))
            guard store.isInstalled(repo: repo) else { return nil }
            return ModelRow(
                repo: repo,
                name: ChatCatalog.spec(forRepo: repo)?.displayName
                    ?? SeedVR2Variant.allCases.first { $0.hfRepo == repo }?.displayName
                    ?? String(repo.split(separator: "/").last ?? Substring(repo)),
                kind: ChatCatalog.spec(forRepo: repo) != nil ? "Chat"
                    : SeedVR2Variant.allCases.contains { $0.hfRepo == repo } ? "Upscaler" : "Model",
                bytes: ModelStore.directorySize(at: store.directory(forRepo: repo))
            )
        }
        return (
            rows.sorted { $0.bytes > $1.bytes },
            ModelStore.directorySize(at: store.outputsDirectory),
            ModelStore.directorySize(at: store.downloadsDirectory),
            ModelStore.directorySize(at: store.jobsDirectory)
        )
    }

    // MARK: - Changing the location

    /// A folder the user picked: switch to it if it holds a library, otherwise plan a move
    /// into it (or into a `LocalLab` folder on it).
    public func choose(_ folder: URL) {
        if let blocker = moveBlocker {
            moveState = .failed(blocker)
            return
        }
        let root = LibraryLocation.libraryRoot(forChosenFolder: folder)
        let resuming = FileManager.default.fileExists(atPath: root.appendingPathComponent(".locallab-move.json").path)
        if LibraryLocation.isLibrary(root), !resuming,
           LibraryMover.canonical(root) != LibraryMover.canonical(location) {
            moveState = .confirmUse(root)
            return
        }
        do {
            moveState = .confirmMove(try LibraryMover.plan(from: store, to: root))
        } catch {
            moveState = .failed(error.localizedDescription)
        }
    }

    public func moveToDefaultLocation() {
        choose(StorageRoot.defaultRoot().url)
    }

    public func dismiss() {
        if case .moving = moveState { return }
        moveState = .idle
    }

    /// Switch to an existing library; the current one stays where it is.
    public func useLibrary(at root: URL) {
        LibraryLocation.save(root, defaults: defaults)
        moveState = .finished(root)
        relaunch()
    }

    public func startMove(_ plan: LibraryMover.Plan) {
        if let blocker = moveBlocker {
            moveState = .failed(blocker)
            return
        }
        LibraryLocation.setPendingMove(plan.destination, defaults: defaults)
        pendingMove = plan.destination
        moveState = .moving(LibraryMover.Progress(phase: .copying, bytesDone: 0, bytesTotal: plan.totalBytes, file: ""))
        moveTask = Task { [weak self] in
            guard let self else { return }
            // Nothing may hold library files open while they move.
            await self.chat?.eject()
            // Progress crosses back through a stream: the copy runs off the main actor.
            let (updates, report) = AsyncStream<LibraryMover.Progress>.makeStream()
            let work = Task.detached(priority: .userInitiated) {
                defer { report.finish() }
                try await LibraryMover.run(plan, progress: { report.yield($0) })
            }
            let shown = Task { @MainActor in
                for await progress in updates {
                    if case .moving = self.moveState { self.moveState = .moving(progress) }
                }
            }
            defer { shown.cancel() }
            do {
                try await withTaskCancellationHandler { try await work.value } onCancel: { work.cancel() }
                LibraryLocation.save(plan.destination, defaults: self.defaults)
                LibraryLocation.setPendingMove(nil, defaults: self.defaults)
                self.pendingMove = nil
                self.moveState = .finished(plan.destination)
                // The running app still holds paths into the old place: start fresh.
                try? await Task.sleep(for: .seconds(1.5))
                self.relaunch()
            } catch is CancellationError {
                self.moveState = .idle
            } catch {
                self.moveState = .failed(error.localizedDescription)
            }
        }
    }

    /// Stop a move. Verified files stay at the destination, so it can resume.
    public func stopMove() {
        moveTask?.cancel()
    }

    public func resumePendingMove() {
        guard let pendingMove else { return }
        do {
            startMove(try LibraryMover.plan(from: store, to: pendingMove))
        } catch {
            moveState = .failed(error.localizedDescription)
        }
    }

    public func discardPendingMove() {
        guard let pendingMove else { return }
        LibraryMover.discardPartialMove(at: pendingMove)
        LibraryLocation.setPendingMove(nil, defaults: defaults)
        self.pendingMove = nil
        moveState = .idle
    }

    // MARK: - Models

    /// Why a model can't be deleted now, or nil.
    public func deleteBlocker(_ repo: String) -> String? {
        if chat?.loadedRepo == repo, chat?.isGenerating == true { return "Chat is using it right now." }
        if chat?.installs[repo] != nil { return "It's still downloading." }
        let waiting = queue.jobs.filter { job in
            guard !job.state.isFinished, case .upscale(let spec) = job.kind else { return false }
            return spec.variant.hfRepo == repo
        }
        if !waiting.isEmpty {
            return "\(waiting.count) job\(waiting.count == 1 ? " needs" : "s need") it. Finish or remove \(waiting.count == 1 ? "it" : "them") first."
        }
        return nil
    }

    public func delete(_ repo: String) async {
        deleteError = nil
        if let blocker = deleteBlocker(repo) {
            deleteError = blocker
            return
        }
        if chat?.loadedRepo == repo { await chat?.eject() }
        do {
            try await installer.uninstall(repo)
        } catch {
            deleteError = error.localizedDescription
        }
        await libraryChanged()
        await refresh()
    }

    /// Throw away partial downloads — only when none is running.
    public func discardDownloads() async {
        guard chat?.installs.isEmpty != false, busyReason() == nil else { return }
        let fm = FileManager.default
        for entry in (try? fm.contentsOfDirectory(at: store.downloadsDirectory, includingPropertiesForKeys: nil)) ?? [] {
            try? fm.removeItem(at: entry)
        }
        await refresh()
    }

    public func reveal(_ url: URL? = nil) {
        NSWorkspace.shared.activateFileViewerSelecting([url ?? location])
    }

    public func url(forRepo repo: String) -> URL { store.directory(forRepo: repo) }
    public var outputsURL: URL { store.outputsDirectory }
}

/// Start a fresh copy of LocalLab and quit this one — after the library moves, every open
/// path points at the old place.
public enum Relaunch {
    @MainActor
    public static func now() {
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.createsNewApplicationInstance = true
        NSWorkspace.shared.openApplication(at: Bundle.main.bundleURL, configuration: configuration) { _, _ in
            Task { @MainActor in NSApp.terminate(nil) }
        }
    }
}
