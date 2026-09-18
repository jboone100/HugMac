import CoreGraphics
import Foundation
import HugMacCore
import ImageIO
import Observation
import UniformTypeIdentifiers

/// The Upscale screen's state and actions (plan §5.11).
///
/// The screen asks for three things — a file, an output size, a quality — and derives
/// everything else. The owner's ComfyUI graph for the same job had 31 settings across
/// three nodes plus plumbing; `plan.reasons` is where those settings went.
@MainActor
@Observable
public final class UpscaleModel {

    // MARK: - Input

    public enum Input: Equatable, Sendable {
        case video(VideoMedia)
        case image(url: URL, width: Int, height: Int)

        public var url: URL {
            switch self {
            case .video(let video): video.url
            case .image(let url, _, _): url
            }
        }
        public var width: Int {
            switch self {
            case .video(let video): video.width
            case .image(_, let width, _): width
            }
        }
        public var height: Int {
            switch self {
            case .video(let video): video.height
            case .image(_, _, let height): height
            }
        }
        public var isVideo: Bool {
            if case .video = self { return true }
            return false
        }
    }

    /// What the user asks for. Only sizes larger than the source are offered.
    public enum OutputSize: String, CaseIterable, Identifiable, Sendable {
        case double, p1080, p1440, p2160

        public var id: String { rawValue }

        public var label: String {
            switch self {
            case .double: "2×"
            case .p1080: "1080p"
            case .p1440: "1440p"
            case .p2160: "4K"
            }
        }

        public var target: UpscaleTarget {
            switch self {
            case .double: .scale(2)
            case .p1080: .shortSide(1080)
            case .p1440: .shortSide(1440)
            case .p2160: .shortSide(2160)
            }
        }

        /// Whether this size would actually enlarge a source of this short side.
        public func enlarges(shortSide: Int) -> Bool {
            switch self {
            case .double: true
            case .p1080: shortSide < 1080
            case .p1440: shortSide < 1440
            case .p2160: shortSide < 2160
            }
        }
    }

    public enum ModelState: Equatable {
        case checking
        case notInstalled
        case installing(fraction: Double, detail: String)
        case installed
        case failed(String)
    }

    public struct RunState: Equatable {
        public var phase: String
        public var unitsDone: Int
        public var unitsTotal: Int
        public var fraction: Double
        public var startedAt: Date
    }

    // MARK: - Observable state

    public private(set) var input: Input?
    public private(set) var thumbnail: CGImage?
    public private(set) var isLoadingInput = false
    public var outputSize: OutputSize = .double { didSet { replan() } }
    public var quality: QualityPreset = .balanced { didSet { replan() } }

    public private(set) var plan: SeedVR2Plan?
    /// Why no plan: set when the resolver refuses, with its numbers.
    public private(set) var refusal: String?
    public private(set) var modelState: ModelState = .checking
    public private(set) var run: RunState?
    public private(set) var outcome: UpscaleOutcome?
    public private(set) var errorMessage: String?
    public private(set) var hardware: HardwareProfile

    /// The checkpoint the screen installs and plans against. int8 3B is the one measured on
    /// a 32 GB Mac; other variants are a later choice on the plan card.
    public let variant: SeedVR2Variant = .threeBInt8

    // MARK: - Dependencies

    private let store: ModelStore
    private let installer: ModelInstaller
    private let runner: UpscaleRunner
    private var calibration: CalibrationStore
    private let calibrationURL: URL?
    private let detectHardware: @Sendable () -> HardwareProfile
    private var runTask: Task<Void, Never>?
    private var sleepActivity: NSObjectProtocol?

    public init(
        store: ModelStore,
        installer: ModelInstaller,
        runner: UpscaleRunner,
        calibration: CalibrationStore,
        calibrationURL: URL? = CalibrationStore.defaultURL(),
        detectHardware: @Sendable @escaping () -> HardwareProfile = { HardwareProfile.detect() }
    ) {
        self.store = store
        self.installer = installer
        self.runner = runner
        self.calibration = calibration
        self.calibrationURL = calibrationURL
        self.detectHardware = detectHardware
        self.hardware = detectHardware()
    }

    /// The production wiring: the default library, the real installer and engine, and this
    /// Mac's saved measurements.
    public static func live() -> UpscaleModel {
        let store = ModelStore()
        let hardware = HardwareProfile.detect()
        return UpscaleModel(
            store: store,
            installer: ModelInstaller(store: store),
            runner: SeedVR2Runner(store: store, chipName: hardware.chipName),
            calibration: CalibrationStore.load()
        )
    }

    // MARK: - Derived

    public var availableOutputSizes: [OutputSize] {
        guard let input else { return OutputSize.allCases }
        let shortSide = min(input.width, input.height)
        return OutputSize.allCases.filter { $0.enlarges(shortSide: shortSide) }
    }

    public var canStart: Bool {
        plan != nil && modelState == .installed && run == nil && input != nil
    }

    public var isRunning: Bool { run != nil }

    /// Rough output size: source bytes scaled by the pixel ratio. On the owner's clip this
    /// predicts 4.6 MB against 4.4 MB measured.
    public var estimatedOutputBytes: Int64? {
        guard let input, let plan else { return nil }
        let sourceBytes = Integrity.fileSize(input.url)
        guard sourceBytes > 0 else { return nil }
        let ratio = Double(plan.outputWidth * plan.outputHeight) / Double(max(input.width * input.height, 1))
        return Int64(Double(sourceBytes) * ratio)
    }

    /// Checkpointed latents held between phases: two tensors per chunk, 16 channels, bf16.
    public var estimatedScratchBytes: Int64? {
        guard let plan, input?.isVideo == true else { return nil }
        return plan.chunks.reduce(Int64(0)) { sum, chunk in
            sum + Int64(16 * chunk.latentFrames * plan.latentWidth * plan.latentHeight * 2 * 2)
        }
    }

    public var freeDiskBytes: Int64 { store.root.availableBytes() }

    /// Seconds remaining: the plan's calibrated prediction until the run has enough progress
    /// to extrapolate from, then the observed rate.
    public func remainingSeconds(now: Date = Date()) -> Double? {
        guard let run else { return nil }
        let elapsed = now.timeIntervalSince(run.startedAt)
        if run.fraction > 0.05 {
            return max(elapsed / run.fraction - elapsed, 0)
        }
        if let total = plan?.totalTime.seconds {
            return max(total - elapsed, 0)
        }
        return nil
    }

    // MARK: - Actions

    /// Check whether the model is installed. Called when the screen appears.
    public func refreshModelState() async {
        let installed = await installer.isInstalled(variant.hfRepo)
        if case .installing = modelState { return }
        modelState = installed ? .installed : .notInstalled
        replan()
    }

    /// Load a video or image. Anything else is refused with a sentence, not a crash.
    public func load(_ url: URL) async {
        errorMessage = nil
        outcome = nil
        isLoadingInput = true
        defer { isLoadingInput = false }

        let type = UTType(filenameExtension: url.pathExtension.lowercased())
        do {
            if type?.conforms(to: .movie) == true || type?.conforms(to: .video) == true {
                let video = try await VideoIO.probe(url)
                input = .video(video)
                thumbnail = try? await VideoIO.readFrames(from: url, startIndex: 0, count: 1, fps: video.fps).first
            } else if type?.conforms(to: .image) == true,
                      let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) {
                input = .image(url: url, width: image.width, height: image.height)
                thumbnail = image
            } else {
                errorMessage = "\(url.lastPathComponent) isn't a video or image HugMac can read."
                return
            }
        } catch {
            errorMessage = String(describing: error)
            return
        }
        if !availableOutputSizes.contains(outputSize) {
            outputSize = availableOutputSizes.first ?? .double
        }
        replan()
    }

    public func clearInput() {
        guard run == nil else { return }
        input = nil
        thumbnail = nil
        plan = nil
        refusal = nil
        outcome = nil
        errorMessage = nil
    }

    /// Re-resolve the plan. Cheap — the resolver is arithmetic — so it runs on every change,
    /// and samples available memory afresh each time, since that is what it plans against.
    public func replan() {
        guard let input, run == nil else { return }
        hardware = detectHardware()
        let resolver = SeedVR2Resolver(hardware: hardware, calibration: calibration)
        let source: SeedVR2Resolver.Source = switch input {
        case .video(let video): .video(video)
        case .image(_, let width, let height): .image(width: width, height: height)
        }
        do {
            plan = try resolver.plan(
                source: source, target: outputSize.target, quality: quality,
                installedVariants: [variant]
            )
            refusal = nil
        } catch {
            plan = nil
            refusal = String(describing: error)
        }
    }

    public func installModel() async {
        guard modelState != .installed else { return }
        modelState = .installing(fraction: 0, detail: "Resolving…")
        do {
            try await installer.install(variant.hfRepo, manifest: SeedVR2Variant.manifest) { progress in
                Task { @MainActor [weak self] in
                    guard let self, case .installing = self.modelState else { return }
                    let detail: String = switch progress.phase {
                    case .resolving: "Resolving…"
                    case .downloading: "Downloading \(progress.file)"
                    case .verifying: "Verifying \(progress.file)"
                    case .finalizing: "Finishing…"
                    case .installed: "Installed"
                    }
                    self.modelState = .installing(fraction: progress.fraction, detail: detail)
                }
            }
            modelState = .installed
            replan()
        } catch is CancellationError {
            modelState = .notInstalled
        } catch {
            modelState = .failed(String(describing: error))
        }
    }

    public func cancelInstall() async {
        await installer.cancel(variant.hfRepo)
        modelState = .notInstalled
    }

    public func start() {
        guard canStart, let input, let plan else { return }
        errorMessage = nil
        outcome = nil
        run = RunState(phase: "Starting", unitsDone: 0, unitsTotal: plan.chunks.count,
                       fraction: 0, startedAt: Date())
        // Hold the Mac awake. The owner's ComfyUI run lost ~2.6 hours to what looks like the
        // Mac sleeping mid-decode; a job this long has to say it's working.
        sleepActivity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .idleSystemSleepDisabled],
            reason: "Upscaling \(input.url.lastPathComponent)"
        )

        let runner = self.runner
        let output = outputURL(for: input, plan: plan)
        let scratch = store.jobsDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let progress: @Sendable (StageProgress) -> Void = { update in
            Task { @MainActor [weak self] in self?.apply(update) }
        }

        runTask = Task { [weak self] in
            do {
                // Detached: MLX evaluation blocks its thread, and that thread must not be
                // the main one.
                let work = Task.detached(priority: .userInitiated) {
                    switch input {
                    case .video(let video):
                        try await runner.upscaleVideo(video, plan: plan, output: output,
                                                      scratch: scratch, progress: progress)
                    case .image(let url, _, _):
                        try await runner.upscaleImage(at: url, plan: plan, output: output,
                                                      progress: progress)
                    }
                }
                // A detached task does not inherit cancellation from whoever awaits it, so
                // Cancel has to be forwarded by hand — without this, Cancel changed nothing
                // and the upscale ran on.
                let outcome = try await withTaskCancellationHandler {
                    try await work.value
                } onCancel: {
                    work.cancel()
                }
                self?.finish(.success(outcome))
            } catch {
                self?.finish(.failure(error))
            }
        }
    }

    public func cancel() {
        runTask?.cancel()
    }

    // MARK: - Internals

    private func apply(_ update: StageProgress) {
        guard var current = run else { return }
        current.phase = Self.phaseLabel(update.phase)
        current.unitsDone = update.unitsDone
        current.unitsTotal = max(update.unitsTotal, current.unitsTotal)
        current.fraction = update.fraction
        run = current
    }

    private func finish(_ result: Result<UpscaleOutcome, Error>) {
        if let sleepActivity { ProcessInfo.processInfo.endActivity(sleepActivity) }
        sleepActivity = nil
        runTask = nil
        run = nil
        switch result {
        case .success(let outcome):
            self.outcome = outcome
            // Feed what this run measured back into the next plan.
            calibration.merge(outcome.samples)
            if let calibrationURL { try? calibration.save(to: calibrationURL) }
            replan()
        case .failure(let error):
            errorMessage = (error is CancellationError) ? "Cancelled." : String(describing: error)
        }
    }

    static func phaseLabel(_ phase: String) -> String {
        switch phase {
        case "vae-encode": "Encoding"
        case "dit": "Upscaling"
        case "vae-decode": "Decoding"
        case "done": "Finishing"
        default: phase.isEmpty ? "Working" : phase.capitalized
        }
    }

    /// `<name>-<w>x<h>.<ext>` in the library's outputs folder, never overwriting a previous
    /// result.
    func outputURL(for input: Input, plan: SeedVR2Plan) -> URL {
        let stem = input.url.deletingPathExtension().lastPathComponent
            + "-\(plan.outputWidth)x\(plan.outputHeight)"
        let ext = input.isVideo ? "mp4" : "png"
        try? FileManager.default.createDirectory(at: store.outputsDirectory, withIntermediateDirectories: true)
        var candidate = store.outputsDirectory.appendingPathComponent("\(stem).\(ext)")
        var counter = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = store.outputsDirectory.appendingPathComponent("\(stem) \(counter).\(ext)")
            counter += 1
        }
        return candidate
    }
}
