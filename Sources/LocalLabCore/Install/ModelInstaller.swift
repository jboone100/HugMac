import Foundation

/// Where an install is, for a progress bar.
public struct InstallProgress: Sendable, Equatable {
    public enum Phase: String, Sendable {
        case resolving, downloading, verifying, finalizing, installed
    }

    public let phase: Phase
    /// The file being worked on, repo-relative.
    public let file: String
    public let bytesDone: Int64
    public let bytesTotal: Int64

    public var fraction: Double {
        guard bytesTotal > 0 else { return phase == .installed ? 1 : 0 }
        return min(max(Double(bytesDone) / Double(bytesTotal), 0), 1)
    }
}

public enum InstallError: Error, Equatable, CustomStringConvertible {
    case noFiles(repo: String)
    case insufficientDisk(requiredBytes: Int64, availableBytes: Int64)
    case verification(repo: String, detail: String)
    case incomplete(repo: String, file: String, detail: String)

    public var description: String {
        switch self {
        case .noFiles(let repo):
            return "\(repo) has no files LocalLab knows how to install."
        case .insufficientDisk(let required, let available):
            return String(
                format: "Not enough disk space where models are stored — this needs about %.1f GB free, and %.1f GB is.",
                Double(required) / 1e9, Double(available) / 1e9
            )
        case .verification(let repo, let detail):
            return "A file from \(repo) failed verification — \(detail). It has been removed; retry to download it again."
        case .incomplete(let repo, let file, let detail):
            return "\(file) from \(repo) is incomplete — \(detail)."
        }
    }
}

/// Installs models from Hugging Face into a `ModelStore`. Ported from MLXUI's
/// `InstallManager`, keeping its guarantees (sibling resolution, heuristic file selection,
/// sequential download, verify, atomic finish, `.installed` marker, reconciled registry,
/// gated-model handling) and changing what didn't hold up at multi-gigabyte sizes:
///
/// | MLXUI                                        | Here                                          |
/// |---|---|
/// | files copied temp → staging → models: 2× disk | streamed into staging, renamed into place: ~1× |
/// | a failed download restarts from zero          | partials survive and resume with `Range`       |
/// | staging wiped at the start of every install    | staging kept; a cancel is a pause              |
/// | size checked to ±1 KB, skipped if unknown      | SHA-256 / git blob SHA-1 on every file         |
/// | files fetched from `resolve/main`              | every file from one pinned commit              |
/// | safetensors never inspected                    | header validated; required tensors checked     |
/// | installs by catalog card, fanned out to repos  | installs by repo (no catalog cards yet)        |
///
/// An `actor`, so two requests for the same repo share one download.
public actor ModelInstaller {
    public nonisolated let store: ModelStore
    private let hub: HubClient
    private let availableBytes: @Sendable () -> Int64
    /// Headroom kept free beyond what the download needs.
    public static let safetyBufferBytes: Int64 = 2_000_000_000

    private var inFlight: [String: Task<InstalledModel, Error>] = [:]

    public init(
        store: ModelStore = ModelStore(),
        hub: HubClient = HuggingFaceHub(),
        availableBytes: (@Sendable () -> Int64)? = nil
    ) {
        self.store = store
        self.hub = hub
        let root = store.root
        self.availableBytes = availableBytes ?? { root.availableBytes() }
    }

    // MARK: - Queries

    public func installedModels() -> [InstalledModel] {
        InstallRegistry.load(store).models.values.sorted { $0.repo < $1.repo }
    }

    public func isInstalled(_ repo: String) -> Bool {
        store.isInstalled(repo: repo)
    }

    /// Bytes already staged for `repo` — what a resumed install won't have to fetch again.
    public func stagedBytes(_ repo: String) -> Int64 {
        ModelStore.directorySize(at: store.stagingDirectory(forRepo: repo))
    }

    // MARK: - Install

    /// Install `repo`, or join an install of it already running.
    @discardableResult
    public func install(
        _ repo: String,
        manifest: ComponentManifest? = nil,
        progress: @Sendable @escaping (InstallProgress) -> Void = { _ in }
    ) async throws -> InstalledModel {
        if let existing = InstallRegistry.load(store).models[repo] {
            progress(InstallProgress(phase: .installed, file: "", bytesDone: existing.sizeBytes,
                                     bytesTotal: existing.sizeBytes))
            return existing
        }
        if let running = inFlight[repo] {
            return try await running.value
        }
        let manifest = manifest ?? .heuristic(for: repo)
        let store = self.store, hub = self.hub, available = self.availableBytes
        let task = Task {
            try await Self.perform(
                repo: repo, manifest: manifest, store: store, hub: hub,
                availableBytes: available, progress: progress
            )
        }
        inFlight[repo] = task
        defer { inFlight[repo] = nil }
        return try await task.value
    }

    /// Stop an install. Its partial files stay in staging, so installing again resumes.
    public func cancel(_ repo: String) {
        inFlight[repo]?.cancel()
        inFlight[repo] = nil
    }

    /// Throw away a paused install's partial files.
    public func discardPartial(_ repo: String) throws {
        cancel(repo)
        let staging = store.stagingDirectory(forRepo: repo)
        if FileManager.default.fileExists(atPath: staging.path) {
            try FileManager.default.removeItem(at: staging)
        }
    }

    public func uninstall(_ repo: String) throws {
        cancel(repo)
        let directory = store.directory(forRepo: repo)
        if FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
        var registry = InstallRegistry.load(store)
        registry.models[repo] = nil
        try registry.save(store)
    }

    // MARK: - The install itself

    /// One file to fetch: which repo and commit it comes from, and where it lands.
    struct PlannedFile {
        let repo: String
        let revision: String
        let remote: RemoteFile
        /// Path within the installed model directory (companions land in a subdirectory).
        let localPath: String
    }

    static func perform(
        repo: String,
        manifest: ComponentManifest,
        store: ModelStore,
        hub: HubClient,
        availableBytes: @Sendable () -> Int64,
        progress: @Sendable @escaping (InstallProgress) -> Void
    ) async throws -> InstalledModel {
        progress(InstallProgress(phase: .resolving, file: "", bytesDone: 0, bytesTotal: 0))
        try store.prepare()

        // 1. Resolve the snapshot and pick files.
        let snapshot = try await hub.snapshot(repo: repo)
        var planned = select(snapshot: snapshot, manifest: manifest, prefix: "")
        for companion in manifest.companions {
            let companionSnapshot = try await hub.snapshot(repo: companion.repo)
            planned += select(
                snapshot: companionSnapshot, manifest: .heuristic(for: companion.repo),
                prefix: companion.subdirectory + "/"
            )
        }
        guard !planned.isEmpty else { throw InstallError.noFiles(repo: repo) }
        // Small files first, so a bad token or a missing file surfaces in seconds, not after
        // a 4 GB transfer.
        planned.sort { $0.remote.size < $1.remote.size }

        let staging = store.stagingDirectory(forRepo: repo)
        let final = store.directory(forRepo: repo)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)

        // 2. Adopt anything already sitting in the model directory without a marker — a
        //    copy made by hand, or an install interrupted after its files landed. Those
        //    files are moved into staging and verified like any other; nothing is fetched
        //    twice if it checks out.
        if FileManager.default.fileExists(atPath: final.path), !store.isInstalled(repo: repo) {
            for file in planned {
                let adopted = final.appendingPathComponent(file.localPath)
                let staged = staging.appendingPathComponent(file.localPath)
                guard FileManager.default.fileExists(atPath: adopted.path),
                      !FileManager.default.fileExists(atPath: staged.path) else { continue }
                try FileManager.default.createDirectory(
                    at: staged.deletingLastPathComponent(), withIntermediateDirectories: true
                )
                try FileManager.default.moveItem(at: adopted, to: staged)
            }
        }

        // 3. Disk pre-flight on what's left to fetch — refuse now rather than fail
        //    gigabytes in. Renaming into place needs no second copy, so no 2× factor.
        let total = planned.reduce(Int64(0)) { $0 + $1.remote.size }
        let alreadyHere = planned.reduce(Int64(0)) { sum, file in
            let staged = staging.appendingPathComponent(file.localPath)
            let partial = staged.appendingPathExtension("partial")
            let have = max(Integrity.fileSize(staged), Integrity.fileSize(partial))
            return sum + min(have, file.remote.size)
        }
        let remaining = total - alreadyHere
        let free = availableBytes()
        if remaining > 0, remaining + safetyBufferBytes > free {
            throw InstallError.insufficientDisk(
                requiredBytes: remaining + safetyBufferBytes, availableBytes: free
            )
        }

        // 4. Fetch and verify each file.
        var completedBytes: Int64 = 0
        var recorded: [InstalledModel.File] = []
        for file in planned {
            try Task.checkCancellation()
            let staged = staging.appendingPathComponent(file.localPath)
            let partial = staged.appendingPathExtension("partial")
            let base = completedBytes

            // Already staged in full: verify rather than re-download.
            if FileManager.default.fileExists(atPath: staged.path) {
                progress(InstallProgress(phase: .verifying, file: file.localPath,
                                         bytesDone: base, bytesTotal: total))
                do {
                    try Integrity.verify(file: staged, displayPath: file.localPath,
                                         size: file.remote.size, expected: file.remote.expected)
                    completedBytes += file.remote.size
                    recorded.append(record(file))
                    continue
                } catch {
                    try? FileManager.default.removeItem(at: staged)
                }
            }

            // A partial longer than the file is not a partial of this file.
            if Integrity.fileSize(partial) > file.remote.size {
                try? FileManager.default.removeItem(at: partial)
            }
            try FileManager.default.createDirectory(
                at: partial.deletingLastPathComponent(), withIntermediateDirectories: true
            )
            if Integrity.fileSize(partial) < file.remote.size || file.remote.size == 0 {
                try await hub.download(
                    repo: file.repo, revision: file.revision, path: file.remote.path, to: partial
                ) { onDisk in
                    progress(InstallProgress(phase: .downloading, file: file.localPath,
                                             bytesDone: base + onDisk, bytesTotal: total))
                }
            }

            progress(InstallProgress(phase: .verifying, file: file.localPath,
                                     bytesDone: base + file.remote.size, bytesTotal: total))
            do {
                try Integrity.verify(file: partial, displayPath: file.localPath,
                                     size: file.remote.size, expected: file.remote.expected)
            } catch {
                // A file that fails its checksum is not resumable — start it fresh next time.
                try? FileManager.default.removeItem(at: partial)
                throw InstallError.verification(repo: repo, detail: String(describing: error))
            }
            try FileManager.default.moveItem(at: partial, to: staged)
            completedBytes += file.remote.size
            recorded.append(record(file))
        }

        // 5. Structural checks on every safetensors file, and the manifest's required tensors.
        progress(InstallProgress(phase: .finalizing, file: "", bytesDone: total, bytesTotal: total))
        for file in planned where file.localPath.hasSuffix(".safetensors") {
            let staged = staging.appendingPathComponent(file.localPath)
            do {
                let header = try SafetensorsHeader.read(staged)
                if let required = manifest.requiredTensors[file.localPath] {
                    try header.require(required)
                }
            } catch {
                throw InstallError.incomplete(
                    repo: repo, file: file.localPath, detail: String(describing: error)
                )
            }
        }

        // 6. Swap staging into place — one rename on the same volume — then write the marker.
        //    The marker is last: its presence means every step above succeeded.
        try Task.checkCancellation()
        if FileManager.default.fileExists(atPath: final.path) {
            try FileManager.default.removeItem(at: final)
        }
        try FileManager.default.moveItem(at: staging, to: final)
        FileManager.default.createFile(atPath: store.installedMarker(forRepo: repo).path, contents: nil)

        // Whole seconds: the registry stores ISO-8601, which drops fractions, and the model
        // returned here should equal the one a later load reads back.
        let installedAt = Date(timeIntervalSince1970: Date().timeIntervalSince1970.rounded(.down))
        let model = InstalledModel(
            repo: repo,
            revision: snapshot.revision,
            installedAt: installedAt,
            sizeBytes: total,
            files: recorded.sorted { $0.path < $1.path }
        )
        var registry = InstallRegistry.load(store)
        registry.models[repo] = model
        try registry.save(store)
        progress(InstallProgress(phase: .installed, file: "", bytesDone: total, bytesTotal: total))
        return model
    }

    static func select(snapshot: RepoSnapshot, manifest: ComponentManifest, prefix: String) -> [PlannedFile] {
        let chosen = Set(manifest.select(from: snapshot.files.map(\.path)))
        return snapshot.files
            .filter { chosen.contains($0.path) }
            .map { PlannedFile(repo: snapshot.repo, revision: snapshot.revision,
                               remote: $0, localPath: prefix + $0.path) }
    }

    static func record(_ file: PlannedFile) -> InstalledModel.File {
        let checksum: String?
        switch file.remote.expected {
        case .sha256(let hex): checksum = "sha256:" + hex
        case .gitBlobSHA1(let hex): checksum = "git-sha1:" + hex
        case .none: checksum = nil
        }
        return InstalledModel.File(path: file.localPath, size: file.remote.size, checksum: checksum)
    }
}
