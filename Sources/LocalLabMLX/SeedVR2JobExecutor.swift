import CoreGraphics
import Foundation
import LocalLabCore
import ImageIO
import UniformTypeIdentifiers

/// Runs upscale jobs on SeedVR2. The queue decides when; this decides nothing, it only
/// runs — resuming from whatever checkpoints are already in `workDirectory`.
public struct SeedVR2JobExecutor: JobExecutor {
    let store: ModelStore
    let calibrationURL: URL

    public init(store: ModelStore, calibrationURL: URL = CalibrationStore.defaultURL()) {
        self.store = store
        self.calibrationURL = calibrationURL
    }

    public func run(
        _ job: Job, context: JobContext,
        progress: @Sendable @escaping (StageProgress) -> Void
    ) async throws -> JobOutcome {
        guard case .upscale(let requested) = job.kind else {
            throw StageError.unsupportedSetting("job kind")
        }
        let workDirectory = context.workDirectory
        let spec = try await Self.resolve(requested, context: context)
        let hardware = HardwareProfile.detect()
        let machine = hardware.machineKey
        let plan = try Self.plan(for: spec, in: workDirectory, hardware: hardware,
                                 calibrationURL: calibrationURL)
        let components = SeedVR2Components(directory: store.directory(forRepo: plan.variant.hfRepo))
        let residency = SeedVR2Residency()
        try FileManager.default.createDirectory(
            at: spec.outputURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )

        switch spec.source {
        case .video(let video):
            let result = try await SeedVR2Engine.upscale(
                video: video, plan: plan, components: components, residency: residency,
                machine: machine, outputURL: spec.outputURL, scratch: workDirectory,
                resume: true, progress: progress
            )
            return JobOutcome(
                outputURL: spec.outputURL, seconds: result.measurements.totalSeconds,
                peakBytes: result.measurements.peakBytes, samples: result.measurements.samples
            )

        case .outputOf:
            // `resolve` always replaces this with a concrete video or image.
            throw StageError.engineFailure(stage: "SeedVR2", detail: "unresolved input")

        case .image(let url, _, _):
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw StageError.engineFailure(stage: "SeedVR2", detail: "couldn't read \(url.lastPathComponent)")
            }
            let result = try await SeedVR2Engine.upscale(
                image: image, plan: plan, components: components, residency: residency,
                machine: machine, progress: progress
            )
            guard let destination = CGImageDestinationCreateWithURL(
                spec.outputURL as CFURL, UTType.png.identifier as CFString, 1, nil
            ) else {
                throw StageError.engineFailure(stage: "SeedVR2", detail: "couldn't create \(spec.outputURL.lastPathComponent)")
            }
            CGImageDestinationAddImage(destination, result.image, nil)
            guard CGImageDestinationFinalize(destination) else {
                throw StageError.engineFailure(stage: "SeedVR2", detail: "couldn't write \(spec.outputURL.lastPathComponent)")
            }
            return JobOutcome(
                outputURL: spec.outputURL, seconds: result.measurements.totalSeconds,
                peakBytes: result.measurements.peakBytes, samples: result.measurements.samples
            )
        }
    }

    /// Turn "the output of job X" into the video or image it actually is — known only now,
    /// when X has finished.
    static func resolve(_ spec: UpscaleJobSpec, context: JobContext) async throws -> UpscaleJobSpec {
        guard case .outputOf(let jobID) = spec.source else { return spec }
        guard let url = context.dependencyOutputs[jobID] else {
            throw StageError.missingInput(slot: "input", kind: .video)
        }
        let type = UTType(filenameExtension: url.pathExtension.lowercased())
        let source: UpscaleJobSpec.Source
        if type?.conforms(to: .movie) == true || type?.conforms(to: .video) == true {
            source = .video(try await VideoIO.probe(url))
        } else if let image = CGImageSourceCreateWithURL(url as CFURL, nil)
                    .flatMap({ CGImageSourceCreateImageAtIndex($0, 0, nil) }) {
            source = .image(url: url, width: image.width, height: image.height)
        } else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "can't read \(url.lastPathComponent)")
        }
        return UpscaleJobSpec(source: source, target: spec.target, quality: spec.quality,
                              variant: spec.variant, outputURL: spec.outputURL)
    }

    /// The plan this job runs with: the one saved in its work directory if it has started
    /// before, else a fresh one made now — against memory as it is *now*, with this Mac's
    /// latest measurements — and saved for any later resume.
    static func plan(
        for spec: UpscaleJobSpec, in workDirectory: URL,
        hardware: HardwareProfile, calibrationURL: URL
    ) throws -> SeedVR2Plan {
        let saved = workDirectory.appendingPathComponent("plan.json")
        if let data = try? Data(contentsOf: saved),
           let plan = try? JSONDecoder().decode(SeedVR2Plan.self, from: data) {
            return plan
        }
        let resolver = SeedVR2Resolver(
            hardware: hardware, calibration: CalibrationStore.load(from: calibrationURL)
        )
        guard let source = spec.source.resolverSource else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "unresolved input")
        }
        let plan = try resolver.plan(
            source: source, target: spec.target, quality: spec.quality,
            installedVariants: [spec.variant]
        )
        try FileManager.default.createDirectory(at: workDirectory, withIntermediateDirectories: true)
        try JSONEncoder().encode(plan).write(to: saved, options: .atomic)
        return plan
    }
}

