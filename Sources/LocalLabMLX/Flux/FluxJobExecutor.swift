import CoreGraphics
import Foundation
import ImageIO
import LocalLabCore
import UniformTypeIdentifiers

/// Runs create-image jobs. A generation is a minute or so, so there's nothing to resume: an
/// interrupted one simply starts again.
public struct FluxJobExecutor: JobExecutor {
    let store: ModelStore

    public init(store: ModelStore) {
        self.store = store
    }

    public func run(
        _ job: Job, context: JobContext,
        progress: @Sendable @escaping (StageProgress) -> Void
    ) async throws -> JobOutcome {
        guard case .createImage(let spec) = job.kind else {
            throw StageError.unsupportedSetting("job kind")
        }
        guard let model = ImageModelCatalog.spec(forRepo: spec.model) else {
            throw StageError.modelNotInstalled(id: spec.model)
        }
        let request = ImageRequest(prompt: spec.prompt, width: spec.width, height: spec.height,
                                   steps: spec.steps, seed: spec.seed)
        let result = try await FluxEngine.generate(
            request, spec: model, directory: store.directory(forRepo: model.repo), progress: progress
        )
        try FileManager.default.createDirectory(
            at: spec.outputURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try Self.writePNG(result.image.cgImage, to: spec.outputURL, prompt: spec.prompt)
        return JobOutcome(outputURL: spec.outputURL, seconds: result.seconds,
                          peakBytes: result.peakBytes, samples: result.samples)
    }

    /// PNG, with the prompt in its description — so the file says how it was made.
    static func writePNG(_ image: CGImage, to url: URL, prompt: String) throws {
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw StageError.engineFailure(stage: "FLUX", detail: "couldn't create \(url.lastPathComponent)")
        }
        let properties: [CFString: Any] = [
            kCGImagePropertyPNGDictionary: [kCGImagePropertyPNGDescription: prompt],
        ]
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw StageError.engineFailure(stage: "FLUX", detail: "couldn't write \(url.lastPathComponent)")
        }
    }
}
