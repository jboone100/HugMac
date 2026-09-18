import Foundation

/// Which library LocalLab uses, remembered between launches. Plan §5.4.
///
/// Stored as a bookmark (which follows a folder the user renames or moves) plus its path
/// (for a readable message when the bookmark can't be resolved). **An unreachable library is
/// never treated as an empty one**: if the drive holding it isn't connected, LocalLab runs on
/// the default library and says which one is missing — rather than showing nothing
/// installed and inviting tens of GB of re-downloads — and keeps the choice for next time.
public enum LibraryLocation {
    static let pathKey = "LocalLab.libraryPath"
    static let bookmarkKey = "LocalLab.libraryBookmark"
    /// A move to this path started and hasn't finished — offered for resuming.
    static let pendingMoveKey = "LocalLab.pendingLibraryMove"

    public struct Resolution: Sendable {
        public let store: ModelStore
        /// The user chose this location (it isn't the default).
        public let isCustom: Bool
        /// The chosen library, when it couldn't be reached and the default is in use instead.
        public let unavailable: URL?

        public init(store: ModelStore, isCustom: Bool, unavailable: URL?) {
            self.store = store
            self.isCustom = isCustom
            self.unavailable = unavailable
        }

        /// "Models SSD" for `/Volumes/Models SSD/…`, else the folder's name.
        public var unavailableName: String? {
            unavailable.map(LibraryLocation.displayName)
        }
    }

    public static func resolve(defaults: UserDefaults = .standard) -> Resolution {
        guard let path = defaults.string(forKey: pathKey) else {
            return Resolution(store: ModelStore(), isCustom: false, unavailable: nil)
        }
        let saved = URL(fileURLWithPath: path, isDirectory: true)
        if let bookmark = defaults.data(forKey: bookmarkKey), let root = StorageRoot.resolve(bookmark: bookmark) {
            // Held for the app's lifetime: every model load and job reads under it.
            if root.isSecurityScoped { _ = root.url.startAccessingSecurityScopedResource() }
            return Resolution(store: ModelStore(root: root), isCustom: true, unavailable: nil)
        }
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: saved.path, isDirectory: &isDirectory), isDirectory.boolValue {
            return Resolution(store: ModelStore(root: StorageRoot(url: saved)), isCustom: true, unavailable: nil)
        }
        return Resolution(store: ModelStore(), isCustom: false, unavailable: saved)
    }

    /// Remember `url` as the library. The default location is stored as "no choice", so a
    /// later change of the default follows.
    public static func save(_ url: URL, defaults: UserDefaults = .standard) {
        let url = url.standardizedFileURL
        guard url != StorageRoot.defaultRoot().url.standardizedFileURL else {
            forget(defaults: defaults)
            return
        }
        defaults.set(url.path, forKey: pathKey)
        let bookmark = (try? url.bookmarkData(options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil))
            ?? (try? url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil))
        defaults.set(bookmark, forKey: bookmarkKey)
    }

    public static func forget(defaults: UserDefaults = .standard) {
        defaults.removeObject(forKey: pathKey)
        defaults.removeObject(forKey: bookmarkKey)
    }

    public static func pendingMove(defaults: UserDefaults = .standard) -> URL? {
        defaults.string(forKey: pendingMoveKey).map { URL(fileURLWithPath: $0, isDirectory: true) }
    }

    public static func setPendingMove(_ url: URL?, defaults: UserDefaults = .standard) {
        if let url { defaults.set(url.path, forKey: pendingMoveKey) } else { defaults.removeObject(forKey: pendingMoveKey) }
    }

    /// True when `url` already holds a LocalLab library.
    public static func isLibrary(_ url: URL) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: url.appendingPathComponent("installed.json").path)
            || fm.fileExists(atPath: url.appendingPathComponent("models").path)
    }

    /// The library folder for a folder the user picked: the folder itself when it is (or is
    /// named as) a library, otherwise a `LocalLab` folder inside it — picking a drive puts
    /// the library in `/Volumes/Drive/LocalLab`, not loose at its top level.
    public static func libraryRoot(forChosenFolder url: URL) -> URL {
        if isLibrary(url) || url.lastPathComponent == LegacyMigration.newFolderName { return url }
        return url.appendingPathComponent(LegacyMigration.newFolderName, isDirectory: true)
    }

    public static func displayName(_ url: URL) -> String {
        let parts = url.standardizedFileURL.pathComponents
        if parts.count > 2, parts[1] == "Volumes" { return parts[2] }
        return url.lastPathComponent
    }
}
