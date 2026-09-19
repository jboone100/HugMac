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

    /// The per-user files that live beside a library in the unsandboxed layout — what this
    /// Mac measured and said — as opposed to the library's own models and outputs.
    public static let perUserItems = ["calibration.json", "probes.json", "conversations", "catalog-cache"]

    /// On the first sandboxed launch the container is empty. When the user points LocalLab
    /// at their old folder, copy its per-user files into the container (never over newer
    /// ones); the library itself is used in place.
    @discardableResult
    public static func adoptPerUserData(from oldFolder: URL, into container: URL, fileManager: FileManager = .default) -> [String] {
        var copied: [String] = []
        try? fileManager.createDirectory(at: container, withIntermediateDirectories: true)
        for item in perUserItems {
            let from = oldFolder.appendingPathComponent(item)
            let to = container.appendingPathComponent(item)
            guard fileManager.fileExists(atPath: from.path), !fileManager.fileExists(atPath: to.path) else { continue }
            if (try? fileManager.copyItem(at: from, to: to)) != nil { copied.append(item) }
        }
        return copied
    }

    /// The user's real home folder. Inside the sandbox `homeDirectoryForCurrentUser` is the
    /// container; this is where a folder picker should start to find the old library.
    public static var realHomeDirectory: URL {
        if let entry = getpwuid(getuid()), let dir = entry.pointee.pw_dir {
            return URL(fileURLWithPath: String(cString: dir), isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
    }

    /// Where the unsandboxed app kept everything.
    public static var unsandboxedFolder: URL {
        realHomeDirectory.appendingPathComponent("Library/Application Support/\(newFolderName)", isDirectory: true)
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
