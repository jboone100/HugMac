import Foundation

/// Access to files and folders outside the app's container, which the sandbox grants only
/// when the user picks or drops them — and only until the app quits (plan §9.4).
///
/// - `hold` keeps a granted URL readable for the rest of the session: a preview, a plan and
///   the job that finally runs may be minutes or hours apart.
/// - `bookmark` records the grant so it survives a relaunch — a paused or queued job's
///   input, the library folder — and `resolve` redeems it.
///
/// Unsandboxed (tests, the bench) every call still works: access is simply always there, and
/// a plain bookmark stands in when a security-scoped one can't be made.
public enum FileAccess {
    private static let held = Locked<Set<String>>([])

    /// Start security-scoped access to `url` and keep it for the session. Idempotent.
    public static func hold(_ url: URL) {
        let path = url.standardizedFileURL.path
        guard held.withLock({ $0.insert(path).inserted }) else { return }
        _ = url.startAccessingSecurityScopedResource()
    }

    /// A bookmark that restores access to `url` after a relaunch.
    public static func bookmark(_ url: URL) -> Data? {
        (try? url.bookmarkData(options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil))
            ?? (try? url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil))
    }

    /// The URL a bookmark points at now — it follows a renamed or moved file — with access
    /// held. Nil when the file is gone.
    public static func resolve(_ bookmark: Data) -> URL? {
        var stale = false
        let url = (try? URL(resolvingBookmarkData: bookmark, options: [.withSecurityScope],
                            relativeTo: nil, bookmarkDataIsStale: &stale))
            ?? (try? URL(resolvingBookmarkData: bookmark, options: [], relativeTo: nil, bookmarkDataIsStale: &stale))
        guard let url else { return nil }
        hold(url)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// True when running in the App Sandbox.
    public static var isSandboxed: Bool {
        ProcessInfo.processInfo.environment["APP_SANDBOX_CONTAINER_ID"] != nil
    }
}

/// A value behind a lock — `Mutex` needs macOS 15, and the package supports 14.
public final class Locked<Value>: @unchecked Sendable {
    private var value: Value
    private let lock = NSLock()
    public init(_ value: Value) { self.value = value }
    public func withLock<R>(_ body: (inout Value) -> R) -> R {
        lock.lock()
        defer { lock.unlock() }
        return body(&value)
    }
}
