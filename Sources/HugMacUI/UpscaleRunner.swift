import CoreGraphics
import Foundation
import HugMacCore
import HugMacMLX
import ImageIO
import UniformTypeIdentifiers

/// What a finished upscale produced and measured.
public struct UpscaleOutcome: Sendable, Equatable {
    public let outputURL: URL
    public let samples: [CalibrationSample]
    public let peakBytes: Int64
    public let seconds: Double
}

/// The boundary between the screen and the engine, so the screen's logic can be tested
/// without loading 4 GB of weights.
public protocol UpscaleRunner: Sendable {
    func upscaleVideo(
        _ video: VideoMedia, plan: SeedVR2Plan, output: URL, scratch: URL,
        progress: @Sendable @escaping (StageProgress) -> Void
    ) async throws -> UpscaleOutcome

    func upscaleImage(
        at url: URL, plan: SeedVR2Plan, output: URL,
        progress: @Sendable @escaping (StageProgress) -> Void
    ) async throws -> UpscaleOutcome
}

/// The real thing: SeedVR2 on MLX.
public struct SeedVR2Runner: UpscaleRunner {
    let store: ModelStore
    let chipName: String

    public init(store: ModelStore, chipName: String) {
        self.store = store
        self.chipName = chipName
    }

    public func upscaleVideo(
        _ video: VideoMedia, plan: SeedVR2Plan, output: URL, scratch: URL,
        progress: @Sendable @escaping (StageProgress) -> Void
    ) async throws -> UpscaleOutcome {
        let components = SeedVR2Components(directory: store.directory(forRepo: plan.variant.hfRepo))
        let residency = SeedVR2Residency()
        let result = try await SeedVR2Engine.upscale(
            video: video, plan: plan, components: components, residency: residency,
            chipName: chipName, outputURL: output, scratch: scratch, resume: true,
            progress: progress
        )
        try? FileManager.default.removeItem(at: scratch)
        return UpscaleOutcome(
            outputURL: output, samples: result.measurements.samples,
            peakBytes: result.measurements.peakBytes, seconds: result.measurements.totalSeconds
        )
    }

    public func upscaleImage(
        at url: URL, plan: SeedVR2Plan, output: URL,
        progress: @Sendable @escaping (StageProgress) -> Void
    ) async throws -> UpscaleOutcome {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "couldn't read \(url.lastPathComponent)")
        }
        let components = SeedVR2Components(directory: store.directory(forRepo: plan.variant.hfRepo))
        let residency = SeedVR2Residency()
        let result = try await SeedVR2Engine.upscale(
            image: image, plan: plan, components: components, residency: residency,
            chipName: chipName, progress: progress
        )
        guard let destination = CGImageDestinationCreateWithURL(
            output as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "couldn't create \(output.lastPathComponent)")
        }
        CGImageDestinationAddImage(destination, result.image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "couldn't write \(output.lastPathComponent)")
        }
        return UpscaleOutcome(
            outputURL: output, samples: result.measurements.samples,
            peakBytes: result.measurements.peakBytes, seconds: result.measurements.totalSeconds
        )
    }
}
