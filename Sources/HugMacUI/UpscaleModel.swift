import CoreGraphics
import Foundation
import HugMacCore
import ImageIO
import Observation
import UniformTypeIdentifiers

/// The Upscale screen's state and actions (plan §5.11).
///
/// The screen asks for three things — a file, an output size, a quality — and derives
/// everything else. It *previews* a plan; the job it starts is planned again at the moment it
/// actually runs (see `UpscaleJobSpec`), and runs on the app's `JobQueue`, so it outlives
/// this screen and its window.
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
        var jobSource: UpscaleJobSpec.Source {
            switch self {
            case .video(let video): .video(video)
            case .image(let url, let width, let height): .image(url: url, width: width, height: height)
            }
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

    // MARK: - Observable state

    public private(set) var input: Input?
    public private(set) var thumbnail: CGImage?
    public private(set) var isLoadingInput = false
    public var outputSize: OutputSize = .double { didSet { replan() } }
    public var quality: QualityPreset = .balanced { didSet { replan() } }

    /// The preview shown on the plan card.
    public private(set) var plan: SeedVR2Plan?
    /// Why no plan: set when the resolver refuses, with its numbers.
    public private(set) var refusal: String?
    public private(set) var modelState: ModelState = .checking
    /// The job this screen most recently started.
    public private(set) var currentJobID: UUID?
    private var loadError: String?
    public private(set) var hardware: HardwareProfile

    /// The checkpoint the screen installs and plans against. int8 3B is the one measured on
    /// a 32 GB Mac; other variants are a later choice on the plan card.
    public let variant: SeedVR2Variant = .threeBInt8

    // MARK: - Dependencies

    public let queue: JobQueue
    private let store: ModelStore
    private let installer: ModelInstaller
    private let detectHardware: @Sendable () -> HardwareProfile

    public init(
        store: ModelStore,
        installer: ModelInstaller,
        queue: JobQueue,
        detectHardware: @Sendable @escaping () -> HardwareProfile = { HardwareProfile.detect() }
    ) {
        self.store = store
        self.installer = installer
        self.queue = queue
        self.detectHardware = detectHardware
        self.hardware = detectHardware()
        // A finished job has just recorded measurements; the preview should use them now,
        // not the next time a setting changes.
        queue.onFinished { [weak self] _ in self?.replan() }
    }

    // MARK: - Derived from the job

    public var currentJob: Job? { currentJobID.flatMap(queue.job) }

    /// Queued or running — this screen's job is in flight.
    public var isRunning: Bool {
        guard let state = currentJob?.state else { return false }
        return state == .queued || state == .running
    }

    public var outcome: JobOutcome? {
        currentJob?.state == .completed ? currentJob?.outcome : nil
    }

    public var errorMessage: String? {
        if let loadError { return loadError }
        switch currentJob?.state {
        case .failed: return currentJob?.failure ?? "The job failed."
        case .cancelled: return "Cancelled."
        default: return nil
        }
    }

    /// Another job is running, so starting this one queues it behind.
    public var willQueue: Bool { queue.running != nil }

    public var availableOutputSizes: [OutputSize] {
        guard let input else { return OutputSize.allCases }
        let shortSide = min(input.width, input.height)
        return OutputSize.allCases.filter { $0.enlarges(shortSide: shortSide) }
    }

    public var canStart: Bool {
        plan != nil && modelState == .installed && input != nil && !isRunning
    }

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

    /// Seconds remaining on this screen's job: the queue's observed rate once there is one,
    /// else the plan's calibrated prediction scaled by what's left.
    public func remainingSeconds(now: Date = Date()) -> Double? {
        guard let job = currentJob, job.state == .running else { return nil }
        if queue.running?.id == job.id, let observed = queue.remainingSeconds(now: now) {
            return observed
        }
        if let total = plan?.totalTime.seconds {
            return max(total * (1 - job.progress.fraction), 0)
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
        loadError = nil
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
                loadError = "\(url.lastPathComponent) isn't a video or image HugMac can read."
                return
            }
        } catch {
            loadError = String(describing: error)
            return
        }
        // A new file starts fresh; the previous job carries on in the queue.
        currentJobID = nil
        if !availableOutputSizes.contains(outputSize) {
            outputSize = availableOutputSizes.first ?? .double
        }
        replan()
    }

    public func clearInput() {
        input = nil
        thumbnail = nil
        plan = nil
        refusal = nil
        loadError = nil
        currentJobID = nil
    }

    /// Re-resolve the preview plan. Cheap — the resolver is arithmetic — so it runs on every
    /// change, and samples available memory afresh each time.
    public func replan() {
        guard let input else { return }
        hardware = detectHardware()
        let resolver = SeedVR2Resolver(hardware: hardware, calibration: queue.calibration)
        do {
            plan = try resolver.plan(
                source: input.jobSource.resolverSource, target: outputSize.target,
                quality: quality, installedVariants: [variant]
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

    /// Put an upscale on the queue. It starts now if nothing else is running.
    public func start() {
        guard canStart, let input, let plan else { return }
        loadError = nil
        let spec = UpscaleJobSpec(
            source: input.jobSource, target: outputSize.target, quality: quality,
            variant: variant, outputURL: outputURL(for: input, plan: plan)
        )
        let title = "\(input.url.lastPathComponent) → \(plan.outputWidth)×\(plan.outputHeight)"
        currentJobID = queue.enqueue(title: title, kind: .upscale(spec)).id
    }

    /// Stop at the next checkpoint, keeping the work done so far.
    public func pause() {
        if let currentJobID { queue.pause(currentJobID) }
    }

    public func resume() {
        if let currentJobID { queue.resume(currentJobID) }
    }

    /// Stop for good and discard the work done so far.
    public func cancel() {
        if let currentJobID { queue.cancel(currentJobID) }
    }

    // MARK: - Internals

    static func phaseLabel(_ phase: String) -> String {
        switch phase {
        case "vae-encode": "Encoding"
        case "dit": "Upscaling"
        case "vae-decode": "Decoding"
        case "done": "Finishing"
        default: phase.isEmpty ? "Starting" : phase.capitalized
        }
    }

    /// `<name>-<w>x<h>.<ext>` in the library's outputs folder, never overwriting an earlier
    /// result or one a queued job has already claimed.
    func outputURL(for input: Input, plan: SeedVR2Plan) -> URL {
        let stem = input.url.deletingPathExtension().lastPathComponent
            + "-\(plan.outputWidth)x\(plan.outputHeight)"
        let claimed = Set(queue.jobs.compactMap { job -> String? in
            // Unfinished jobs reserve their name; finished ones already exist on disk.
            guard case .upscale(let spec) = job.kind, !job.state.isFinished else { return nil }
            return spec.outputURL.path
        })
        return ModelStore.uniqueOutputURL(
            in: store.outputsDirectory, stem: stem,
            extension: input.isVideo ? "mp4" : "png", reserved: claimed
        )
    }
}
