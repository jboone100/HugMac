import Foundation

/// A model that finished installing, as recorded in `installed.json`.
public struct InstalledModel: Codable, Sendable, Equatable {
    public struct File: Codable, Sendable, Equatable {
        public let path: String
        public let size: Int64
        /// SHA-256 for LFS files, git blob SHA-1 for small ones — kept so a later "verify
        /// installed models" can re-check a library offline.
        public let checksum: String?
    }

    public let repo: String
    /// The commit every file came from.
    public let revision: String
    public let installedAt: Date
    public let sizeBytes: Int64
    public let files: [File]
}

/// `installed.json`, one per storage root.
///
/// The `.installed` marker inside each model directory is the source of truth — it is written
/// last, after every file verified. The registry is reconciled against it on load, so an
/// entry whose files were deleted behind the app's back simply drops out.
public struct InstallRegistry: Codable, Sendable, Equatable {
    public var version = 1
    public var models: [String: InstalledModel] = [:]

    public init() {}

    public static func load(_ store: ModelStore) -> InstallRegistry {
        guard let data = try? Data(contentsOf: store.registryURL) else { return InstallRegistry() }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard var registry = try? decoder.decode(InstallRegistry.self, from: data) else {
            return InstallRegistry()
        }
        registry.models = registry.models.filter { store.isInstalled(repo: $0.key) }
        return registry
    }

    public func save(_ store: ModelStore) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try FileManager.default.createDirectory(
            at: store.baseDirectory, withIntermediateDirectories: true
        )
        try encoder.encode(self).write(to: store.registryURL, options: .atomic)
    }
}
