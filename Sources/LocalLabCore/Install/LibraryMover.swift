import CryptoKit
import Darwin
import Foundation

/// Moves the library — models, downloads, outputs, jobs — to another folder, and proves the
/// copy before deleting anything. Plan §5.4.
///
/// - **Same volume:** a rename per top-level item — instant, whatever the size.
/// - **Another volume:** each file is copied while hashing the source, then read back from
///   the disk (bypassing the cache) and hashed again; the two must match. Only when every
///   file has verified are the originals deleted.
/// - **Resumable:** verified files are listed in `.locallab-move.json` at the destination;
///   a move stopped by a quit or a cancel skips them when it resumes.
/// - Saved job records are rewritten to point at the new place.
///
/// Per-user data — calibration, probe results, conversations — never moves: it describes
/// this Mac, not the library, and stays in `~/Library/Application Support/LocalLab`.
public enum LibraryMover {
    public static let libraryItems = ["installed.json", "models", "downloads", "outputs", "jobs"]
    static let manifestName = ".locallab-move.json"

    public struct Plan: Sendable, Equatable {
        public let source: URL
        public let destination: URL
        public let totalBytes: Int64
        public let fileCount: Int
        public let sameVolume: Bool
    }

    public struct Progress: Sendable, Equatable {
        public enum Phase: String, Sendable { case copying, verifying, finishing }
        public let phase: Phase
        public let bytesDone: Int64
        public let bytesTotal: Int64
        public let file: String

        public init(phase: Phase, bytesDone: Int64, bytesTotal: Int64, file: String) {
            self.phase = phase
            self.bytesDone = bytesDone
            self.bytesTotal = bytesTotal
            self.file = file
        }

        public var fraction: Double {
            bytesTotal > 0 ? min(Double(bytesDone) / Double(bytesTotal), 1) : (phase == .finishing ? 1 : 0)
        }
    }

    public enum MoveError: Error, Equatable, LocalizedError {
        case sameLocation
        case destinationInsideLibrary
        case libraryInsideDestination
        case destinationHasLibrary
        case notEnoughSpace(neededBytes: Int64, availableBytes: Int64)
        case verificationFailed(file: String)

        public var errorDescription: String? {
            switch self {
            case .sameLocation: "That's where the library already is."
            case .destinationInsideLibrary: "The new location is inside the current library."
            case .libraryInsideDestination: "The current library is inside that folder — choose a different one."
            case .destinationHasLibrary: "That folder already holds a LocalLab library. Use it instead, or choose an empty folder."
            case .notEnoughSpace(let needed, let available):
                String(format: "Not enough space there: the library needs %.1f GB and %.1f GB is free.",
                       Double(needed) / 1_073_741_824, Double(available) / 1_073_741_824)
            case .verificationFailed(let file): "\(file) didn't copy correctly, so nothing was deleted. Try again, or choose another disk."
            }
        }
    }

    // MARK: - Planning

    public static func plan(
        from store: ModelStore, to destination: URL,
        availableBytes: ((URL) -> Int64)? = nil
    ) throws -> Plan {
        let source = canonical(store.baseDirectory)
        let target = canonical(destination)
        guard source != target else { throw MoveError.sameLocation }
        if target.path.hasPrefix(source.path + "/") { throw MoveError.destinationInsideLibrary }
        if source.path.hasPrefix(target.path + "/") { throw MoveError.libraryInsideDestination }
        let resuming = FileManager.default.fileExists(atPath: target.appendingPathComponent(manifestName).path)
        if LibraryLocation.isLibrary(target), !resuming { throw MoveError.destinationHasLibrary }

        let files = regularFiles(under: source)
        let total = files.reduce(Int64(0)) { $0 + $1.size }
        let sameVolume = volumeID(of: source) != nil && volumeID(of: source) == volumeID(of: existingAncestor(of: target))
        if !sameVolume {
            let free = (availableBytes ?? { StorageRoot(url: existingAncestor(of: $0)).availableBytes() })(target)
            let already = resuming ? Manifest.load(at: target).bytesDone : 0
            // A little room beyond the library itself, so the disk isn't left full.
            let needed = total - already + 512 * 1_048_576
            guard needed <= free else { throw MoveError.notEnoughSpace(neededBytes: total, availableBytes: free) }
        }
        return Plan(source: source, destination: target, totalBytes: total, fileCount: files.count, sameVolume: sameVolume)
    }

    // MARK: - Moving

    /// Run a move. Throws `CancellationError` when cancelled — verified files are kept for a
    /// resume. `forceCopy` exercises the copy-and-verify path on one volume (tests);
    /// `tamper` lets a test damage a copy before it is verified.
    public static func run(
        _ plan: Plan,
        forceCopy: Bool = false,
        tamper: (@Sendable (URL) -> Void)? = nil,
        progress: @Sendable (Progress) -> Void = { _ in }
    ) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: plan.destination, withIntermediateDirectories: true)

        if plan.sameVolume && !forceCopy {
            for item in libraryItems {
                let from = plan.source.appendingPathComponent(item)
                guard fm.fileExists(atPath: from.path) else { continue }
                try fm.moveItem(at: from, to: plan.destination.appendingPathComponent(item))
            }
        } else {
            try await copyAndVerify(plan, tamper: tamper, progress: progress)
        }

        progress(Progress(phase: .finishing, bytesDone: plan.totalBytes, bytesTotal: plan.totalBytes, file: ""))
        remapJobs(in: plan.destination.appendingPathComponent("jobs"), from: plan.source, to: plan.destination)

        // Everything is at the destination and verified: now, and only now, remove the
        // originals.
        if !(plan.sameVolume && !forceCopy) {
            for item in libraryItems {
                try? fm.removeItem(at: plan.source.appendingPathComponent(item))
            }
        }
        try? fm.removeItem(at: plan.destination.appendingPathComponent(manifestName))
    }

    /// Delete what a stopped move had copied. The originals were never touched.
    public static func discardPartialMove(at destination: URL) {
        let fm = FileManager.default
        guard fm.fileExists(atPath: destination.appendingPathComponent(manifestName).path) else { return }
        for item in libraryItems {
            try? fm.removeItem(at: destination.appendingPathComponent(item))
        }
        try? fm.removeItem(at: destination.appendingPathComponent(manifestName))
        // Leave the folder itself only if something else is in it.
        if (try? fm.contentsOfDirectory(atPath: destination.path))?.isEmpty == true {
            try? fm.removeItem(at: destination)
        }
    }

    static func copyAndVerify(
        _ plan: Plan, tamper: (@Sendable (URL) -> Void)?, progress: @Sendable (Progress) -> Void
    ) async throws {
        let fm = FileManager.default
        var manifest = Manifest.load(at: plan.destination)
        manifest.source = plan.source.path
        var done: Int64 = 0

        for file in regularFiles(under: plan.source) {
            try Task.checkCancellation()
            let to = plan.destination.appendingPathComponent(file.relativePath)
            if manifest.verified[file.relativePath] == file.size, Integrity.fileSize(to) == file.size {
                done += file.size
                continue
            }
            try fm.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
            progress(Progress(phase: .copying, bytesDone: done, bytesTotal: plan.totalBytes, file: file.relativePath))
            let base = done
            let sourceHash = try copy(file.url, to: to) { copied in
                progress(Progress(phase: .copying, bytesDone: base + copied, bytesTotal: plan.totalBytes, file: file.relativePath))
            }
            tamper?(to)
            progress(Progress(phase: .verifying, bytesDone: base + file.size, bytesTotal: plan.totalBytes, file: file.relativePath))
            guard try hash(to, uncached: true) == sourceHash else {
                try? fm.removeItem(at: to)
                throw MoveError.verificationFailed(file: file.relativePath)
            }
            if let date = (try? file.url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate {
                try? fm.setAttributes([.modificationDate: date], ofItemAtPath: to.path)
            }
            done += file.size
            manifest.verified[file.relativePath] = file.size
            try manifest.save(at: plan.destination)
            await Task.yield()
        }
        // Empty folders — an `outputs/` with nothing in it yet — are part of the layout too.
        for item in libraryItems {
            let from = plan.source.appendingPathComponent(item)
            var isDirectory: ObjCBool = false
            if fm.fileExists(atPath: from.path, isDirectory: &isDirectory), isDirectory.boolValue {
                try fm.createDirectory(at: plan.destination.appendingPathComponent(item), withIntermediateDirectories: true)
            }
        }
    }

    // MARK: - Jobs

    /// Point saved jobs' inputs, outputs and results at the library's new place, so a finished
    /// job still shows its file and a paused one resumes.
    static func remapJobs(in jobsDirectory: URL, from source: URL, to destination: URL) {
        let fm = FileManager.default
        let prefix = source.path
        let map: (URL) -> URL = { url in
            let path = url.standardizedFileURL.path
            guard path == prefix || path.hasPrefix(prefix + "/") else { return url }
            return URL(fileURLWithPath: destination.path + path.dropFirst(prefix.count))
        }
        for entry in (try? fm.contentsOfDirectory(at: jobsDirectory, includingPropertiesForKeys: nil)) ?? [] {
            let file = entry.appendingPathComponent("job.json")
            guard let data = try? Data(contentsOf: file),
                  let job = try? JobQueue.decoder.decode(Job.self, from: data),
                  let encoded = try? JobQueue.encoder.encode(job.remappingURLs(map)) else { continue }
            try? encoded.write(to: file, options: .atomic)
        }
    }

    // MARK: - Files

    struct File {
        let url: URL
        let relativePath: String
        let size: Int64
    }

    static func regularFiles(under root: URL) -> [File] {
        var files: [File] = []
        for item in libraryItems {
            let top = root.appendingPathComponent(item)
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: top.path, isDirectory: &isDirectory) else { continue }
            if !isDirectory.boolValue {
                files.append(File(url: top, relativePath: item, size: Integrity.fileSize(top)))
                continue
            }
            // Relative paths straight from the enumerator: rebuilding them from absolute
            // paths breaks when part of the path is a symlink (`/var` → `/private/var`).
            guard let enumerator = FileManager.default.enumerator(atPath: top.path) else { continue }
            for case let relative as String in enumerator {
                let url = top.appendingPathComponent(relative)
                guard (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true else { continue }
                files.append(File(url: url, relativePath: item + "/" + relative, size: Integrity.fileSize(url)))
            }
        }
        return files.sorted { $0.relativePath < $1.relativePath }
    }

    static let chunk = 8 * 1_048_576

    /// Copy `from` to `to`, hashing what was read. Returns the source's SHA-256.
    static func copy(_ from: URL, to: URL, copied: (Int64) -> Void) throws -> String {
        let input = open(from.path, O_RDONLY)
        guard input >= 0 else { throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: from.path]) }
        defer { close(input) }
        let output = open(to.path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        guard output >= 0 else { throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: to.path]) }
        defer { close(output) }
        var hasher = SHA256()
        var buffer = [UInt8](repeating: 0, count: chunk)
        var total: Int64 = 0
        while true {
            try Task.checkCancellation()
            let count = buffer.withUnsafeMutableBytes { read(input, $0.baseAddress, chunk) }
            if count < 0 { throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: from.path]) }
            if count == 0 { break }
            try buffer.withUnsafeBytes { bytes in
                let slice = UnsafeRawBufferPointer(rebasing: bytes[0 ..< count])
                hasher.update(bufferPointer: slice)
                var offset = 0
                while offset < count {
                    let written = write(output, slice.baseAddress?.advanced(by: offset), count - offset)
                    guard written > 0 else { throw CocoaError(.fileWriteOutOfSpace, userInfo: [NSFilePathErrorKey: to.path]) }
                    offset += written
                }
            }
            total += Int64(count)
            copied(total)
        }
        guard fsync(output) == 0 else { throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: to.path]) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// SHA-256 of a file; `uncached` reads it from the disk rather than from memory, so a
    /// copy is checked as it was actually written.
    static func hash(_ url: URL, uncached: Bool) throws -> String {
        let input = open(url.path, O_RDONLY)
        guard input >= 0 else { throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: url.path]) }
        defer { close(input) }
        if uncached { _ = fcntl(input, F_NOCACHE, 1) }
        var hasher = SHA256()
        var buffer = [UInt8](repeating: 0, count: chunk)
        while true {
            try Task.checkCancellation()
            let count = buffer.withUnsafeMutableBytes { read(input, $0.baseAddress, chunk) }
            if count < 0 { throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: url.path]) }
            if count == 0 { break }
            buffer.withUnsafeBytes { hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: $0[0 ..< count])) }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func volumeID(of url: URL) -> String? {
        let values = try? url.resourceValues(forKeys: [.volumeIdentifierKey])
        return values?.volumeIdentifier.map { "\($0)" }
    }

    /// One spelling per folder, whether or not it exists yet: resolve the part that exists
    /// (following symlinks such as `/var` → `/private/var`) and append the rest. Resolving
    /// only an existing path would name the same destination two ways before and after
    /// it's created — and a resumed move wouldn't recognise its own work.
    public static func canonical(_ url: URL) -> URL {
        let standard = url.standardizedFileURL
        let ancestor = existingAncestor(of: standard)
        let rest = standard.pathComponents.dropFirst(ancestor.pathComponents.count)
        var resolved = ancestor.resolvingSymlinksInPath()
        for component in rest { resolved.appendPathComponent(component) }
        // One form for equality too: URLs with and without a trailing slash compare unequal.
        return URL(fileURLWithPath: resolved.path, isDirectory: true)
    }

    /// The nearest folder that exists — the destination may not be created yet.
    static func existingAncestor(of url: URL) -> URL {
        var current = url
        while !FileManager.default.fileExists(atPath: current.path), current.pathComponents.count > 1 {
            current.deleteLastPathComponent()
        }
        return current
    }

    struct Manifest: Codable {
        var source = ""
        var verified: [String: Int64] = [:]
        var bytesDone: Int64 { verified.values.reduce(0, +) }

        static func load(at destination: URL) -> Manifest {
            guard let data = try? Data(contentsOf: destination.appendingPathComponent(manifestName)),
                  let manifest = try? JSONDecoder().decode(Manifest.self, from: data) else { return Manifest() }
            return manifest
        }

        func save(at destination: URL) throws {
            try JSONEncoder().encode(self).write(to: destination.appendingPathComponent(manifestName), options: .atomic)
        }
    }
}
