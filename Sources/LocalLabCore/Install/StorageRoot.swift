import Foundation

/// Where the model library lives. Plan §5.4.
///
/// MLXUI computed its root once from `.applicationSupportDirectory` inside a `static let`,
/// and 47 call sites reached for `ModelStore.shared`, which is what made the location
/// impossible to change. Here the root is a value the app resolves at launch — the default,
/// or a folder the user chose — and everything downstream is built from it.
///
/// A user-chosen folder outside the sandbox container arrives as a security-scoped URL;
/// `withAccess` brackets file work so the sandboxed edition can actually touch it. For the
/// default root, or the unsandboxed Direct edition, it is a no-op.
public struct StorageRoot: Sendable, Equatable {
    public let url: URL
    public let isSecurityScoped: Bool

    public init(url: URL, isSecurityScoped: Bool = false) {
        self.url = url
        self.isSecurityScoped = isSecurityScoped
    }

    /// `~/Library/Application Support/LocalLab`.
    public static func defaultRoot() -> StorageRoot {
        let support = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return StorageRoot(url: support.appendingPathComponent("LocalLab", isDirectory: true))
    }

    /// Resolve a root from a stored bookmark. Returns nil if the volume is gone — the caller
    /// falls back to the default root and says so, rather than treating a missing external
    /// drive as an empty library and re-downloading it.
    public static func resolve(bookmark: Data) -> StorageRoot? {
        var stale = false
        guard let url = try? URL(
            resolvingBookmarkData: bookmark,
            options: [.withSecurityScope],
            relativeTo: nil,
            bookmarkDataIsStale: &stale
        ) else { return nil }
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return StorageRoot(url: url, isSecurityScoped: true)
    }

    /// Run `body` with access to the root held open.
    public func withAccess<T>(_ body: () throws -> T) rethrows -> T {
        let started = isSecurityScoped && url.startAccessingSecurityScopedResource()
        defer { if started { url.stopAccessingSecurityScopedResource() } }
        return try body()
    }

    public func withAccess<T>(_ body: () async throws -> T) async rethrows -> T {
        let started = isSecurityScoped && url.startAccessingSecurityScopedResource()
        defer { if started { url.stopAccessingSecurityScopedResource() } }
        return try await body()
    }

    /// Free bytes on the volume holding this root — measured here, not on `$HOME`, since the
    /// library may be on another disk.
    public func availableBytes() -> Int64 {
        let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        if let important = values?.volumeAvailableCapacityForImportantUsage {
            return important
        }
        let fallback = try? url.resourceValues(forKeys: [.volumeAvailableCapacityKey])
        return Int64(fallback?.volumeAvailableCapacity ?? 0)
    }
}
