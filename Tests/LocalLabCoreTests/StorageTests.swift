import CryptoKit
import Foundation
import Testing
@testable import LocalLabCore

private func tempDirectory(_ name: String = "storage") -> URL {
    FileManager.default.temporaryDirectory
        .appendingPathComponent("locallab-\(name)-\(UUID().uuidString)", isDirectory: true)
}

private func freshDefaults() -> UserDefaults {
    let name = "locallab-storage-tests-\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: name) ?? .standard
    defaults.removePersistentDomain(forName: name)
    return defaults
}

/// A small library: an installed model, a partial download, an output, and a finished job
/// whose record points at that output.
private func makeLibrary(at root: URL) throws -> (store: ModelStore, jobID: UUID) {
    let store = ModelStore(root: StorageRoot(url: root))
    try store.prepare()
    let model = store.directory(forRepo: "mlx-community/Tiny")
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    try Data(repeating: 7, count: 3 * 1_048_576 + 17).write(to: model.appendingPathComponent("model.safetensors"))
    FileManager.default.createFile(atPath: store.installedMarker(forRepo: "mlx-community/Tiny").path, contents: nil)
    try Data("{}".utf8).write(to: store.registryURL)
    let staging = store.stagingDirectory(forRepo: "mlx-community/Partial")
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
    try Data("half".utf8).write(to: staging.appendingPathComponent("a.partial"))
    let output = store.outputsDirectory.appendingPathComponent("clip 2x.mp4")
    try Data("video".utf8).write(to: output)

    var job = Job(title: "clip", kind: .upscale(UpscaleJobSpec(
        source: .image(url: URL(fileURLWithPath: "/Users/someone/Pictures/in.png"), width: 10, height: 10),
        target: .scale(2), quality: .balanced, variant: .threeBInt8, outputURL: output
    )))
    job.state = .completed
    job.outcome = JobOutcome(outputURL: output, seconds: 1, peakBytes: 1, samples: [])
    let jobDirectory = store.jobsDirectory.appendingPathComponent(job.id.uuidString)
    try FileManager.default.createDirectory(at: jobDirectory, withIntermediateDirectories: true)
    try JobQueue.encoder.encode(job).write(to: jobDirectory.appendingPathComponent("job.json"))
    return (store, job.id)
}

@Suite("Library location")
struct LibraryLocationTests {
    @Test func noChoiceMeansTheDefault() {
        let resolution = LibraryLocation.resolve(defaults: freshDefaults())
        #expect(!resolution.isCustom)
        #expect(resolution.unavailable == nil)
        #expect(resolution.store.baseDirectory == StorageRoot.defaultRoot().url)
    }

    @Test func aChosenFolderIsRemembered() throws {
        let defaults = freshDefaults()
        let folder = tempDirectory()
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        LibraryLocation.save(folder, defaults: defaults)
        let resolution = LibraryLocation.resolve(defaults: defaults)
        #expect(resolution.isCustom)
        #expect(resolution.store.baseDirectory.resolvingSymlinksInPath() == folder.resolvingSymlinksInPath())
    }

    @Test func aMissingDriveIsNeverTreatedAsAnEmptyLibrary() {
        let defaults = freshDefaults()
        defaults.set("/Volumes/Models SSD/LocalLab", forKey: LibraryLocation.pathKey)
        let resolution = LibraryLocation.resolve(defaults: defaults)
        #expect(resolution.store.baseDirectory == StorageRoot.defaultRoot().url, "runs on the default library meanwhile")
        #expect(resolution.unavailableName == "Models SSD")
        #expect(defaults.string(forKey: LibraryLocation.pathKey) != nil, "the choice is kept for next launch")
    }

    @Test func choosingTheDefaultForgetsTheChoice() {
        let defaults = freshDefaults()
        defaults.set("/somewhere", forKey: LibraryLocation.pathKey)
        LibraryLocation.save(StorageRoot.defaultRoot().url, defaults: defaults)
        #expect(defaults.string(forKey: LibraryLocation.pathKey) == nil)
    }

    @Test func pickingADriveMakesAFolderOnIt() throws {
        let drive = tempDirectory("drive")
        defer { try? FileManager.default.removeItem(at: drive) }
        try FileManager.default.createDirectory(at: drive, withIntermediateDirectories: true)
        #expect(LibraryLocation.libraryRoot(forChosenFolder: drive).lastPathComponent == "LocalLab")
        _ = try makeLibrary(at: drive)
        #expect(LibraryLocation.libraryRoot(forChosenFolder: drive) == drive, "an existing library is used as is")
    }
}

@Suite("Moving the library")
struct LibraryMoverTests {
    @Test func refusesMovesThatWouldGoWrong() throws {
        let root = tempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let (store, _) = try makeLibrary(at: root)
        #expect(throws: LibraryMover.MoveError.sameLocation) { try LibraryMover.plan(from: store, to: root) }
        #expect(throws: LibraryMover.MoveError.destinationInsideLibrary) {
            try LibraryMover.plan(from: store, to: root.appendingPathComponent("models/inside"))
        }
        let other = tempDirectory("other")
        defer { try? FileManager.default.removeItem(at: other) }
        _ = try makeLibrary(at: other)
        #expect(throws: LibraryMover.MoveError.destinationHasLibrary) { try LibraryMover.plan(from: store, to: other) }
    }

    @Test func refusesWhenTheDiskIsTooSmall() throws {
        let root = tempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let (store, _) = try makeLibrary(at: root)
        // Pretend the destination is on another volume with 1 MB free.
        let plan = try LibraryMover.plan(from: store, to: tempDirectory("dest"), availableBytes: { _ in 1_048_576 })
        #expect(plan.sameVolume, "temp folders share a volume, so no space check applies")
    }

    @Test func aSameVolumeMoveIsARenameAndJobsFollow() async throws {
        let root = tempDirectory()
        let destination = tempDirectory("dest")
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: destination) }
        let (store, jobID) = try makeLibrary(at: root)
        let plan = try LibraryMover.plan(from: store, to: destination)
        #expect(plan.sameVolume)
        #expect(plan.fileCount == 6)
        try await LibraryMover.run(plan)
        try expectMoved(from: root, to: destination, jobID: jobID)
    }

    @Test func aCrossVolumeMoveCopiesVerifiesThenDeletes() async throws {
        let root = tempDirectory()
        let destination = tempDirectory("dest")
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: destination) }
        let (store, jobID) = try makeLibrary(at: root)
        let original = try Data(contentsOf: store.directory(forRepo: "mlx-community/Tiny").appendingPathComponent("model.safetensors"))
        let plan = try LibraryMover.plan(from: store, to: destination)
        let phases = Locked<Set<String>>([])
        try await LibraryMover.run(plan, forceCopy: true, progress: { progress in phases.withLock { $0.insert(progress.phase.rawValue) } })
        try expectMoved(from: root, to: destination, jobID: jobID)
        let copied = try Data(contentsOf: destination.appendingPathComponent("models/mlx-community--Tiny/model.safetensors"))
        #expect(copied == original)
        #expect(phases.withLock { $0 } == ["copying", "verifying", "finishing"])
        #expect(!FileManager.default.fileExists(atPath: destination.appendingPathComponent(".locallab-move.json").path))
    }

    @Test func aBadCopyDeletesNothing() async throws {
        let root = tempDirectory()
        let destination = tempDirectory("dest")
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: destination) }
        let (store, _) = try makeLibrary(at: root)
        let plan = try LibraryMover.plan(from: store, to: destination)
        await #expect(throws: LibraryMover.MoveError.self) {
            try await LibraryMover.run(plan, forceCopy: true, tamper: { url in
                guard url.lastPathComponent == "model.safetensors",
                      let handle = try? FileHandle(forWritingTo: url) else { return }
                try? handle.seek(toOffset: 1_000)
                try? handle.write(contentsOf: Data([0xFF]))
                try? handle.close()
            })
        }
        // Every original is still where it was.
        #expect(FileManager.default.fileExists(atPath: store.installedMarker(forRepo: "mlx-community/Tiny").path))
        #expect(FileManager.default.fileExists(atPath: store.directory(forRepo: "mlx-community/Tiny").appendingPathComponent("model.safetensors").path))
    }

    @Test func aStoppedMoveResumesWithoutRecopying() async throws {
        let root = tempDirectory()
        let destination = tempDirectory("dest")
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: destination) }
        let (store, jobID) = try makeLibrary(at: root)
        // A first attempt verified the model file, then stopped.
        let plan = try LibraryMover.plan(from: store, to: destination)
        let modelPath = "models/mlx-community--Tiny/model.safetensors"
        let target = destination.appendingPathComponent(modelPath)
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: root.appendingPathComponent(modelPath), to: target)
        var manifest = LibraryMover.Manifest()
        manifest.verified[modelPath] = Integrity.fileSize(target)
        try manifest.save(at: destination)

        let resumed = try LibraryMover.plan(from: store, to: destination)
        #expect(resumed.destination == plan.destination, "a partial move is recognised, not refused as a library")
        let copiedFiles = Locked<Set<String>>([])
        try await LibraryMover.run(resumed, forceCopy: true, progress: { progress in
            if progress.phase == .copying, !progress.file.isEmpty { copiedFiles.withLock { $0.insert(progress.file) } }
        })
        #expect(!copiedFiles.withLock { $0 }.contains(modelPath))
        try expectMoved(from: root, to: destination, jobID: jobID)
    }

    @Test func discardingAPartialMoveLeavesTheOriginal() throws {
        let root = tempDirectory()
        let destination = tempDirectory("dest")
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: destination) }
        let (store, _) = try makeLibrary(at: root)
        try FileManager.default.createDirectory(at: destination.appendingPathComponent("models"), withIntermediateDirectories: true)
        try LibraryMover.Manifest().save(at: destination)
        LibraryMover.discardPartialMove(at: destination)
        #expect(!FileManager.default.fileExists(atPath: destination.path))
        #expect(store.isInstalled(repo: "mlx-community/Tiny"))
    }

    private func expectMoved(from root: URL, to destination: URL, jobID: UUID) throws {
        let moved = ModelStore(root: StorageRoot(url: destination))
        #expect(moved.isInstalled(repo: "mlx-community/Tiny"))
        #expect(FileManager.default.fileExists(atPath: moved.registryURL.path))
        #expect(FileManager.default.fileExists(atPath: moved.stagingDirectory(forRepo: "mlx-community/Partial").appendingPathComponent("a.partial").path), "a partial download can still resume")
        for item in LibraryMover.libraryItems {
            #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent(item).path), "\(item) left behind")
        }
        let data = try Data(contentsOf: moved.jobsDirectory.appendingPathComponent("\(jobID.uuidString)/job.json"))
        let job = try JobQueue.decoder.decode(Job.self, from: data)
        let expected = moved.outputsDirectory.appendingPathComponent("clip 2x.mp4").path
        #expect(job.outcome?.outputURL.path == expected)
        #expect(job.kind.outputURL.path == expected)
        guard case .upscale(let spec) = job.kind, case .image(let input, _, _) = spec.source else {
            Issue.record("kind changed")
            return
        }
        #expect(input.path == "/Users/someone/Pictures/in.png", "files outside the library are left alone")
    }
}

/// A value behind a lock — `Mutex` needs macOS 15, and the package supports 14.
final class Locked<Value>: @unchecked Sendable {
    private var value: Value
    private let lock = NSLock()
    init(_ value: Value) { self.value = value }
    func withLock<R>(_ body: (inout Value) -> R) -> R {
        lock.lock()
        defer { lock.unlock() }
        return body(&value)
    }
}
