import CoreGraphics
import Foundation
import HugMacCore
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import HugMacUI

// MARK: - Fixtures

final class FakeRunner: UpscaleRunner, @unchecked Sendable {
    enum Behaviour { case succeed, fail, waitForCancel }
    let behaviour: Behaviour
    init(_ behaviour: Behaviour = .succeed) { self.behaviour = behaviour }

    func upscaleVideo(_ video: VideoMedia, plan: SeedVR2Plan, output: URL, scratch: URL,
                      progress: @Sendable @escaping (StageProgress) -> Void) async throws -> UpscaleOutcome {
        try await work(plan: plan, output: output, progress: progress)
    }

    func upscaleImage(at url: URL, plan: SeedVR2Plan, output: URL,
                      progress: @Sendable @escaping (StageProgress) -> Void) async throws -> UpscaleOutcome {
        try await work(plan: plan, output: output, progress: progress)
    }

    private func work(plan: SeedVR2Plan, output: URL,
                      progress: @Sendable @escaping (StageProgress) -> Void) async throws -> UpscaleOutcome {
        progress(StageProgress(fraction: 0.5, phase: "vae-decode", unitsDone: 1, unitsTotal: plan.chunks.count))
        switch behaviour {
        case .fail:
            throw StageError.engineFailure(stage: "SeedVR2", detail: "simulated")
        case .waitForCancel:
            while true { try await Task.sleep(for: .milliseconds(20)) }
        case .succeed:
            try Data("out".utf8).write(to: output)
            let samples = plan.phases.map {
                CalibrationSample(engineID: SeedVR2Resolver.mlxEngineID, phase: $0.phase,
                                  workUnits: $0.workUnits, peakUnits: $0.peakUnits, seconds: 10,
                                  peakBytes: $0.peakBytes, weightBytes: $0.weightBytes,
                                  chipName: "Apple M2 Max")
            }
            return UpscaleOutcome(outputURL: output, samples: samples,
                                  peakBytes: plan.peakBytes, seconds: 30)
        }
    }
}

func m2Max(availableGB: Double = 19) -> HardwareProfile {
    HardwareProfile(
        chipName: "Apple M2 Max", generation: 2, tier: .max, gpuCoreCount: 30,
        memoryBandwidthGBps: 400, totalMemoryBytes: 34_359_738_368,
        availableMemoryBytes: Int64(availableGB * 1_073_741_824),
        gpuWiredLimitBytes: Int64(0.75 * 34_359_738_368),
        macOSVersion: .init(majorVersion: 26, minorVersion: 0, patchVersion: 0)
    )
}

struct Workspace {
    let root: URL
    let store: ModelStore
    let calibrationURL: URL

    init() {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("hugmac-ui-tests-\(UUID().uuidString)", isDirectory: true)
        store = ModelStore(root: StorageRoot(url: root.appendingPathComponent("library")))
        calibrationURL = root.appendingPathComponent("calibration.json")
        try? store.prepare()
    }

    func markInstalled(_ variant: SeedVR2Variant = .threeBInt8) {
        let directory = store.directory(forRepo: variant.hfRepo)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: store.installedMarker(forRepo: variant.hfRepo).path, contents: nil)
    }

    /// A real, tiny H.264/HEVC clip, written with the same writer the engine uses.
    func makeVideo(width: Int = 96, height: Int = 64, frames: Int = 9) async throws -> URL {
        let url = root.appendingPathComponent("clip.mp4")
        let writer = try VideoIO.Writer(url: url, width: width, height: height, fps: 24)
        for index in 0 ..< frames {
            try writer.append(solidImage(width: width, height: height, shade: CGFloat(index) / CGFloat(frames)))
        }
        try await writer.finish()
        return url
    }

    func makeImage(width: Int = 320, height: Int = 200) throws -> URL {
        let url = root.appendingPathComponent("photo.png")
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(destination, solidImage(width: width, height: height, shade: 0.4), nil)
        guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
        return url
    }

    func solidImage(width: Int, height: Int, shade: CGFloat) -> CGImage {
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        context.setFillColor(CGColor(red: shade, green: 0.5, blue: 1 - shade, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()!
    }

    @MainActor
    func model(runner: UpscaleRunner = FakeRunner(), availableGB: Double = 19,
               installer: ModelInstaller? = nil) -> UpscaleModel {
        UpscaleModel(
            store: store,
            installer: installer ?? ModelInstaller(store: store, hub: FakeHubStub(), availableBytes: { 1 << 40 }),
            runner: runner,
            calibration: CalibrationStore(),
            calibrationURL: calibrationURL,
            detectHardware: { m2Max(availableGB: availableGB) }
        )
    }

    func cleanUp() { try? FileManager.default.removeItem(at: root) }
}

/// The screen never downloads in these tests; a hub that refuses keeps it honest.
struct FakeHubStub: HubClient {
    func snapshot(repo: String) async throws -> RepoSnapshot { throw HubError.notFound(repo: repo) }
    func download(repo: String, revision: String, path: String, to destination: URL,
                  progress: @Sendable @escaping (Int64) -> Void) async throws {
        throw HubError.notFound(repo: repo)
    }
}

@MainActor
func waitUntil(_ condition: @MainActor () -> Bool, timeout: Double = 5) async {
    let deadline = Date().addingTimeInterval(timeout)
    while !condition(), Date() < deadline {
        try? await Task.sleep(for: .milliseconds(20))
    }
}

// MARK: - Tests

@Suite("Upscale screen model")
@MainActor
struct UpscaleModelTests {

    @Test("Loading a video probes it, plans it, and offers only sizes that enlarge it")
    func loadsVideo() async throws {
        let workspace = Workspace(); defer { workspace.cleanUp() }
        let model = workspace.model()
        let url = try await workspace.makeVideo()
        await model.load(url)

        guard case .video(let video) = model.input else {
            Issue.record("expected a video input"); return
        }
        #expect(video.width == 96 && video.height == 64)
        #expect(model.thumbnail != nil)
        let plan = try #require(model.plan)
        #expect(plan.outputWidth == 192 && plan.outputHeight == 128)
        #expect(model.availableOutputSizes == UpscaleModel.OutputSize.allCases)
    }

    @Test("A 1080p source isn't offered 1080p — that would be a copy, not an upscale")
    func onlyEnlargingSizes() {
        #expect(!UpscaleModel.OutputSize.p1080.enlarges(shortSide: 1080))
        #expect(UpscaleModel.OutputSize.p1440.enlarges(shortSide: 1080))
        #expect(UpscaleModel.OutputSize.double.enlarges(shortSide: 4000))
    }

    @Test("An image plans as a single frame and writes a PNG")
    func loadsImage() async throws {
        let workspace = Workspace(); defer { workspace.cleanUp() }
        workspace.markInstalled()
        let model = workspace.model()
        await model.refreshModelState()
        await model.load(try workspace.makeImage())

        let plan = try #require(model.plan)
        #expect(plan.isSingleImage)
        #expect(plan.outputWidth == 640 && plan.outputHeight == 400)
        model.start()
        await waitUntil { !model.isRunning }
        #expect(model.outcome?.outputURL.pathExtension == "png")
    }

    @Test("Start stays disabled until the model is installed")
    func requiresInstall() async throws {
        let workspace = Workspace(); defer { workspace.cleanUp() }
        let model = workspace.model()
        await model.refreshModelState()
        await model.load(try await workspace.makeVideo())
        #expect(model.modelState == .notInstalled)
        #expect(model.plan != nil, "the plan still shows, so the user sees the cost before installing")
        #expect(!model.canStart)

        workspace.markInstalled()
        await model.refreshModelState()
        #expect(model.canStart)
    }

    @Test("When nothing fits, the plan card explains with numbers instead of offering Start")
    func refusesWithReason() async throws {
        let workspace = Workspace(); defer { workspace.cleanUp() }
        workspace.markInstalled()
        let model = workspace.model(availableGB: 2)
        await model.refreshModelState()
        await model.load(try await workspace.makeVideo(width: 640, height: 360))
        #expect(model.plan == nil)
        let refusal = try #require(model.refusal)
        #expect(refusal.contains("GB"))
        #expect(!model.canStart)
    }

    @Test("A run reports progress, finishes, and records its measurements for next time")
    func runsAndCalibrates() async throws {
        let workspace = Workspace(); defer { workspace.cleanUp() }
        workspace.markInstalled()
        let model = workspace.model()
        await model.refreshModelState()
        await model.load(try await workspace.makeVideo())
        #expect(model.plan?.totalTime == .unknown, "no measurements yet")

        model.start()
        #expect(model.isRunning)
        await waitUntil { !model.isRunning }

        let outcome = try #require(model.outcome)
        #expect(FileManager.default.fileExists(atPath: outcome.outputURL.path))
        #expect(outcome.outputURL.path.hasPrefix(workspace.store.outputsDirectory.path))
        #expect(FileManager.default.fileExists(atPath: workspace.calibrationURL.path))
        #expect(model.plan?.totalTime != .unknown, "the next plan is predicted from this run")
    }

    @Test("Cancel stops the run and says so")
    func cancels() async throws {
        let workspace = Workspace(); defer { workspace.cleanUp() }
        workspace.markInstalled()
        let model = workspace.model(runner: FakeRunner(.waitForCancel))
        await model.refreshModelState()
        await model.load(try await workspace.makeVideo())
        model.start()
        await waitUntil { model.run?.phase == "Decoding" }
        model.cancel()
        await waitUntil { !model.isRunning }
        #expect(model.errorMessage == "Cancelled.")
        #expect(model.outcome == nil)
    }

    @Test("An engine failure surfaces as a sentence, and the screen is usable again")
    func surfacesFailure() async throws {
        let workspace = Workspace(); defer { workspace.cleanUp() }
        workspace.markInstalled()
        let model = workspace.model(runner: FakeRunner(.fail))
        await model.refreshModelState()
        await model.load(try await workspace.makeVideo())
        model.start()
        await waitUntil { !model.isRunning }
        #expect(model.errorMessage?.contains("simulated") == true)
        #expect(model.canStart)
    }

    @Test("Results never overwrite an earlier one")
    func uniqueOutputNames() async throws {
        let workspace = Workspace(); defer { workspace.cleanUp() }
        workspace.markInstalled()
        let model = workspace.model()
        await model.refreshModelState()
        await model.load(try await workspace.makeVideo())
        model.start(); await waitUntil { !model.isRunning }
        let first = try #require(model.outcome?.outputURL)
        model.start(); await waitUntil { !model.isRunning }
        let second = try #require(model.outcome?.outputURL)
        #expect(first != second)
        #expect(second.lastPathComponent == "clip-192x128 2.mp4")
    }

    @Test("Time remaining comes from the plan early on, then from the observed rate")
    func remainingTime() async throws {
        let workspace = Workspace(); defer { workspace.cleanUp() }
        workspace.markInstalled()
        let model = workspace.model(runner: FakeRunner(.waitForCancel))
        await model.refreshModelState()
        await model.load(try await workspace.makeVideo())
        model.start()
        await waitUntil { model.run?.fraction == 0.5 }
        let run = try #require(model.run)
        // Halfway after 60 s → about 60 s left.
        let remaining = try #require(model.remainingSeconds(now: run.startedAt.addingTimeInterval(60)))
        #expect(abs(remaining - 60) < 1)
        model.cancel()
        await waitUntil { !model.isRunning }
    }

    @Test("Output size estimate scales the source file by pixel count")
    func outputEstimate() async throws {
        let workspace = Workspace(); defer { workspace.cleanUp() }
        let model = workspace.model()
        let url = try await workspace.makeVideo()
        await model.load(url)
        let source = Integrity.fileSize(url)
        #expect(model.estimatedOutputBytes == source * 4)
        #expect((model.estimatedScratchBytes ?? 0) > 0)
    }
}
