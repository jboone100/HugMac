import Foundation

/// The app was called HugMac until 2026-09-18. Its library, calibration and conversations
/// lived in `~/Library/Application Support/HugMac`; this moves them to `…/LocalLab` once.
///
/// A rename on the same volume, so tens of GB of models move instantly. A symbolic link is
/// left at the old path, so any absolute path saved before the rename — a job's output, a
/// checkpoint — still resolves.
public enum LegacyMigration {
    public static let oldFolderName = "HugMac"
    public static let newFolderName = "LocalLab"

    /// Settings stored before the bundle ID became `io.github.jboone100.locallab` live in the
    /// placeholder's domain. Copy LocalLab's own keys across once, never over newer ones.
    public static let oldBundleID = "com.locallab.app"
    static let defaultsMigratedKey = "LocalLab.settingsCopiedFromOldBundleID"

    @discardableResult
    public static func copySettingsIfNeeded(
        from old: UserDefaults? = UserDefaults(suiteName: oldBundleID), to current: UserDefaults = .standard
    ) -> Int {
        guard !current.bool(forKey: defaultsMigratedKey), let old else { return 0 }
        var copied = 0
        for (key, value) in old.dictionaryRepresentation()
        where key.hasPrefix("LocalLab.") && current.object(forKey: key) == nil {
            current.set(value, forKey: key)
            copied += 1
        }
        current.set(true, forKey: defaultsMigratedKey)
        return copied
    }

    /// Returns true if it moved something.
    @discardableResult
    public static func moveApplicationSupportIfNeeded(
        appSupport: URL? = nil, fileManager: FileManager = .default
    ) -> Bool {
        guard let base = appSupport ?? fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
        else { return false }
        let old = base.appendingPathComponent(oldFolderName, isDirectory: true)
        let new = base.appendingPathComponent(newFolderName, isDirectory: true)
        let oldIsRealFolder = (try? old.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == false
            && fileManager.fileExists(atPath: old.path)
        guard oldIsRealFolder, !fileManager.fileExists(atPath: new.path) else { return false }
        do {
            try fileManager.moveItem(at: old, to: new)
            try? fileManager.createSymbolicLink(at: old, withDestinationURL: new)
            return true
        } catch {
            return false
        }
    }
}
