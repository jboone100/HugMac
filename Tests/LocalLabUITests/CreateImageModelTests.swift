import Foundation
import LocalLabCore
import Testing
@testable import LocalLabUI

/// Writes a placeholder file where the image would go.
struct FakeImageExecutor: JobExecutor {
    func run(_ job: Job, context: JobContext,
             progress: @Sendable @escaping (StageProgress) -> Void) async throws -> JobOutcome {
        guard case .createImage(let spec) = job.kind else { throw StageError.cancelled }
        progress(StageProgress(fraction: 0.5, phase: "transformer", unitsDone: 2, unitsTotal: 4))
        try Data("png".utf8).write(to: spec.outputURL)
        return JobOutcome(outputURL: spec.outputURL, seconds: 12, peakBytes: 7_000_000_000, samples: [])
    }
}

@MainActor
@Suite("Create Image screen", .serialized)
struct CreateImageModelTests {
    let workspace = Workspace()
    let defaults: UserDefaults

    init() {
        let suite = "LocalLab.tests.createImage.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suite) ?? .standard
    }

    func makeModel(installed: Bool = true, sent: SentURLs = SentURLs()) -> CreateImageModel {
        let spec = ImageModelCatalog.fluxSchnell4bit
        if installed {
            let directory = workspace.store.directory(forRepo: spec.repo)
            try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: workspace.store.installedMarker(forRepo: spec.repo).path, contents: nil)
        }
        // The bundled reference Macs, as the app loads them — the source of a first estimate.
        let queue = JobQueue(store: workspace.store, executor: FakeImageExecutor(), activity: NoActivity(),
                             calibration: CalibrationStore(reference: ReferenceMachine.all),
                             calibrationURL: workspace.calibrationURL)
        return CreateImageModel(
            store: workspace.store,
            installer: ModelInstaller(store: workspace.store, hub: FakeHubStub(), availableBytes: { 1 << 40 }),
            queue: queue, defaults: defaults,
            detectHardware: { m2Max(availableGB: 19) },
            sendToUpscale: { sent.urls.append($0) }
        )
    }

    final class SentURLs { var urls: [URL] = [] }

    func waitForIdle(_ model: CreateImageModel) async {
        for _ in 0 ..< 200 where model.queue.activeCount > 0 {
            try? await Task.sleep(for: .milliseconds(10))
        }
    }

    @Test("Create says what's missing, one thing at a time")
    func blockers() async {
        let missing = makeModel(installed: false)
        await missing.refreshModelState()
        #expect(missing.startBlocker == "Install the model first.")

        let model = makeModel()
        await model.refreshModelState()
        #expect(model.startBlocker == "Describe the image you want.")
        model.prompt = "   "
        #expect(model.startBlocker == "Describe the image you want.")
        model.prompt = "a fox"
        model.seedText = "twelve"
        #expect(model.startBlocker?.contains("whole number") == true)
        model.seedText = ""
        #expect(model.startBlocker == nil)
        workspace.cleanUp()
    }

    @Test("an image is a job: named from the prompt, and listed when it's done")
    func createsAJob() async throws {
        let model = makeModel()
        await model.refreshModelState()
        model.prompt = "  A red fox, in the snow!  "
        model.size = .landscape
        model.seedText = "42"
        model.start()
        let job = try #require(model.currentJob)
        guard case .createImage(let spec) = job.kind else { Issue.record("wrong kind"); return }
        #expect(spec.prompt == "A red fox, in the snow!")
        #expect((spec.width, spec.height) == (1344, 768))
        #expect(spec.seed == 42 && spec.steps == 4)
        #expect(spec.outputURL.lastPathComponent == "a-red-fox-in-the-snow.png")
        #expect(spec.outputURL.deletingLastPathComponent().path == workspace.store.outputsDirectory.path)

        // A second press while the first waits gets its own file.
        model.start()
        let second = try #require(model.currentJob)
        #expect(second.kind.outputURL.lastPathComponent == "a-red-fox-in-the-snow 2.png")

        await waitForIdle(model)
        #expect(model.results.count == 2)
        workspace.cleanUp()
    }

    @Test("a blank seed picks a new one each time")
    func randomSeeds() async {
        let model = makeModel()
        await model.refreshModelState()
        model.prompt = "a fox"
        var seeds = Set<UInt64>()
        for _ in 0 ..< 3 {
            model.start()
            if case .createImage(let spec) = model.currentJob?.kind { seeds.insert(spec.seed) }
        }
        #expect(seeds.count == 3)
        await waitForIdle(model)
        workspace.cleanUp()
    }

    @Test("a result can be made again, or sent to Upscale")
    func reuseAndUpscale() async throws {
        let sent = SentURLs()
        let model = makeModel(sent: sent)
        await model.refreshModelState()
        model.prompt = "a lighthouse"
        model.size = .portrait
        model.seedText = "7"
        model.start()
        await waitForIdle(model)
        let result = try #require(model.results.first)

        model.prompt = ""
        model.seedText = ""
        model.size = .square512
        model.reuse(result)
        #expect(model.prompt == "a lighthouse" && model.seedText == "7" && model.size == .portrait)

        model.upscale(result)
        #expect(sent.urls == [result.outcome?.outputURL].compactMap { $0 })
        workspace.cleanUp()
    }

    @Test("the chosen size is remembered")
    func remembersSize() {
        let first = makeModel()
        first.size = .square768
        let second = makeModel()
        #expect(second.size == .square768)
        workspace.cleanUp()
    }

    @Test("Smart Fit grades the chosen size for this Mac")
    func grades() {
        let model = makeModel()
        model.size = .square512
        let small = model.fit?.time.seconds
        model.size = .square1024
        let large = model.fit?.time.seconds
        #expect(model.fit?.grade == .green)
        if let small, let large { #expect(small < large) } else { Issue.record("no time estimate") }
        workspace.cleanUp()
    }

    @Test("prompts become short, safe titles")
    func titles() {
        #expect(CreateImageModel.title(for: "one two three") == "one two three")
        #expect(CreateImageModel.title(for: "one two three four five six seven eight nine") == "one two three four five six seven eight…")
    }
}
