import CoreGraphics
import Foundation
import HugMacCore
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
        _ job: Job, workDirectory: URL,
        progress: @Sendable @escaping (StageProgress) -> Void
    ) async throws -> JobOutcome {
        guard case .upscale(let spec) = job.kind else {
            throw StageError.unsupportedSetting("job kind")
        }
        let hardware = HardwareProfile.detect()
        let chipName = hardware.chipName
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
                chipName: chipName, outputURL: spec.outputURL, scratch: workDirectory,
                resume: true, progress: progress
            )
            return JobOutcome(
                outputURL: spec.outputURL, seconds: result.measurements.totalSeconds,
                peakBytes: result.measurements.peakBytes, samples: result.measurements.samples
            )

        case .image(let url, _, _):
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw StageError.engineFailure(stage: "SeedVR2", detail: "couldn't read \(url.lastPathComponent)")
            }
            let result = try await SeedVR2Engine.upscale(
                image: image, plan: plan, components: components, residency: residency,
                chipName: chipName, progress: progress
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
        let plan = try resolver.plan(
            source: spec.source.resolverSource, target: spec.target, quality: spec.quality,
            installedVariants: [spec.variant]
        )
        try FileManager.default.createDirectory(at: workDirectory, withIntermediateDirectories: true)
        try JSONEncoder().encode(plan).write(to: saved, options: .atomic)
        return plan
    }
}

