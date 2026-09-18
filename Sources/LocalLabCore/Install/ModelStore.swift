import Foundation

/// The one place on-disk paths are built. Plan §5.4.
///
/// An **instance**, built from a `StorageRoot`, rather than MLXUI's `static let shared` —
/// so a changed library location, or a test's temporary directory, is a constructor argument
/// instead of a global. Layout, per root:
///
///     installed.json         registry of what is installed here
///     models/<repo-slug>/    a model's files, plus the `.installed` marker
///     downloads/<repo-slug>/ staging for an in-progress install; survives a cancel so the
///                            install can resume
///     outputs/               generated images, audio and video
///     jobs/                  per-job scratch (checkpoints, intermediates)
///
/// `installed.json` lives per root, so switching roots switches libraries coherently.
public struct ModelStore: Sendable, Equatable {
    public let root: StorageRoot

    public init(root: StorageRoot = .defaultRoot()) {
        self.root = root
    }

    public var baseDirectory: URL { root.url }
    public var modelsDirectory: URL { baseDirectory.appendingPathComponent("models", isDirectory: true) }
    public var downloadsDirectory: URL { baseDirectory.appendingPathComponent("downloads", isDirectory: true) }
    public var outputsDirectory: URL { baseDirectory.appendingPathComponent("outputs", isDirectory: true) }
    public var jobsDirectory: URL { baseDirectory.appendingPathComponent("jobs", isDirectory: true) }
    public var registryURL: URL { baseDirectory.appendingPathComponent("installed.json") }

    /// `mlx-community/SeedVR2-3B-mlx-int8` → `mlx-community--SeedVR2-3B-mlx-int8`.
    /// The one place this transform is computed.
    public static func slug(for repo: String) -> String {
        repo.replacingOccurrences(of: "/", with: "--")
    }

    public func directory(forRepo repo: String) -> URL {
        modelsDirectory.appendingPathComponent(Self.slug(for: repo), isDirectory: true)
    }

    public func stagingDirectory(forRepo repo: String) -> URL {
        downloadsDirectory.appendingPathComponent(Self.slug(for: repo), isDirectory: true)
    }

    /// The atomic "install succeeded" signal. Written last, only after every file verified.
    public func installedMarker(forRepo repo: String) -> URL {
        directory(forRepo: repo).appendingPathComponent(".installed")
    }

    public func isInstalled(repo: String) -> Bool {
        FileManager.default.fileExists(atPath: installedMarker(forRepo: repo).path)
    }

    /// Create the top-level directories. Idempotent.
    public func prepare() throws {
        for directory in [modelsDirectory, downloadsDirectory, outputsDirectory, jobsDirectory] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }

    /// `<directory>/<stem>.<ext>`, or `<stem> 2.<ext>`, `<stem> 3.<ext>`… — the first name
    /// that neither exists nor is `reserved` (claimed by a job that hasn't written it yet).
    /// The one place output names are chosen, so nothing writes over an earlier result.
    public static func uniqueOutputURL(
        in directory: URL, stem: String, extension ext: String, reserved: Set<String> = []
    ) -> URL {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var candidate = directory.appendingPathComponent("\(stem).\(ext)")
        var counter = 2
        while FileManager.default.fileExists(atPath: candidate.path) || reserved.contains(candidate.path) {
            candidate = directory.appendingPathComponent("\(stem) \(counter).\(ext)")
            counter += 1
        }
        return candidate
    }

    /// Total bytes under `url`, recursing. An unreadable child is skipped rather than
    /// aborting the count — a partial total beats a blank settings pane.
    public static func directorySize(at url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey],
            options: [],
            errorHandler: { _, _ in true }
        ) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            guard let values = try? file.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey]),
                  values.isDirectory != true else { continue }
            total += Int64(values.fileSize ?? 0)
        }
        return total
    }
}
