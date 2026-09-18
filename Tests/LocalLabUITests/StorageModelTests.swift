import Foundation
import LocalLabCore
import Testing
@testable import LocalLabUI

@Suite("Storage settings model")
@MainActor
struct StorageModelTests {
    func defaults() -> UserDefaults {
        let name = "locallab-storage-ui-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name) ?? .standard
        defaults.removePersistentDomain(forName: name)
        return defaults
    }

    func install(_ repo: String, bytes: Int = 2_048, in workspace: Workspace) {
        let directory = workspace.store.directory(forRepo: repo)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? Data(repeating: 1, count: bytes).write(to: directory.appendingPathComponent("model.safetensors"))
        FileManager.default.createFile(atPath: workspace.store.installedMarker(forRepo: repo).path, contents: nil)
    }

    func make(_ workspace: Workspace, defaults: UserDefaults, relaunched: Locked<Int> = Locked(0),
              changed: Locked<Int> = Locked(0)) -> (StorageModel, JobQueue) {
        let queue = JobQueue(store: workspace.store, executor: FakeExecutor(), activity: NoActivity(),
                             calibrationURL: workspace.calibrationURL)
        let model = StorageModel(
            resolution: LibraryLocation.Resolution(store: workspace.store, isCustom: false, unavailable: nil),
            installer: ModelInstaller(store: workspace.store, hub: FakeHubStub(), availableBytes: { 1 << 40 }),
            queue: queue, chat: nil, defaults: defaults,
            libraryChanged: { changed.withLock { $0 += 1 } },
            relaunch: { relaunched.withLock { $0 += 1 } }
        )
        return (model, queue)
    }

    @Test("It lists what's installed, by size, with a kind people recognise")
    func listsModels() async {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install("mlx-community/Qwen3.5-9B-4bit", bytes: 4_096, in: workspace)
        install(SeedVR2Variant.threeBInt8.hfRepo, bytes: 1_024, in: workspace)
        let (model, _) = make(workspace, defaults: defaults())
        await model.refresh()
        #expect(model.models.map(\.name) == ["Qwen3.5 9B", "SeedVR2 3B int8"])
        #expect(model.models.map(\.kind) == ["Chat", "Upscaler"])
        #expect(model.totalBytes >= 5_120)
    }

    @Test("A model a waiting job needs can't be deleted; others can")
    func deleteRules() async throws {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install(SeedVR2Variant.threeBInt8.hfRepo, in: workspace)
        install("mlx-community/Qwen3.5-0.8B-4bit", in: workspace)
        let changed = Locked(0)
        let (model, queue) = make(workspace, defaults: defaults(), changed: changed)
        queue.pauseQueue()
        queue.enqueue(title: "waiting", kind: .upscale(UpscaleJobSpec(
            source: .image(url: try workspace.makeImage(), width: 320, height: 200),
            target: .scale(2), quality: .balanced, variant: .threeBInt8,
            outputURL: workspace.root.appendingPathComponent("out.png")
        )))
        #expect(model.deleteBlocker(SeedVR2Variant.threeBInt8.hfRepo)?.contains("1 job needs it") == true)

        await model.delete("mlx-community/Qwen3.5-0.8B-4bit")
        #expect(!workspace.store.isInstalled(repo: "mlx-community/Qwen3.5-0.8B-4bit"))
        #expect(changed.withLock { $0 } == 1, "the other screens are told")
        #expect(model.models.map(\.repo) == [SeedVR2Variant.threeBInt8.hfRepo])
    }

    @Test("Picking a folder that holds a library offers to switch to it, moving nothing")
    func useExisting() throws {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        let other = workspace.root.appendingPathComponent("other-library")
        try ModelStore(root: StorageRoot(url: other)).prepare()
        let settings = defaults(), relaunched = Locked(0)
        let (model, _) = make(workspace, defaults: settings, relaunched: relaunched)
        model.choose(other)
        #expect(model.moveState == .confirmUse(other))
        model.useLibrary(at: other)
        #expect(settings.string(forKey: "LocalLab.libraryPath") == other.standardizedFileURL.path)
        #expect(relaunched.withLock { $0 } == 1)
    }

    @Test("Moving to an empty folder: confirm, move, remember, restart")
    func moveFlow() async throws {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        install("mlx-community/Qwen3.5-0.8B-4bit", in: workspace)
        let drive = workspace.root.appendingPathComponent("Drive")
        try FileManager.default.createDirectory(at: drive, withIntermediateDirectories: true)
        let settings = defaults(), relaunched = Locked(0)
        let (model, _) = make(workspace, defaults: settings, relaunched: relaunched)

        model.choose(drive)
        guard case .confirmMove(let plan) = model.moveState else {
            Issue.record("expected a move to confirm, got \(model.moveState)")
            return
        }
        #expect(plan.destination.lastPathComponent == "LocalLab", "a drive gets a LocalLab folder, not loose files")
        model.startMove(plan)
        for _ in 0 ..< 400 where relaunched.withLock({ $0 }) == 0 { try? await Task.sleep(for: .milliseconds(10)) }
        #expect(model.moveState == .finished(plan.destination))
        #expect(relaunched.withLock { $0 } == 1)
        #expect(settings.string(forKey: "LocalLab.libraryPath") == plan.destination.path)
        #expect(settings.string(forKey: "LocalLab.pendingLibraryMove") == nil)
        #expect(ModelStore(root: StorageRoot(url: plan.destination)).isInstalled(repo: "mlx-community/Qwen3.5-0.8B-4bit"))
        #expect(!workspace.store.isInstalled(repo: "mlx-community/Qwen3.5-0.8B-4bit"))
    }
}
