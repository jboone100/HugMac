import CoreGraphics
import Foundation
import LocalLabCore

/// The registry-facing stage: one engine, two input shapes.
///
/// `image → image` and `video → video` are the same model and the same code path — the only
/// difference is how many frames a chunk holds and whether frames come from a decoder. This
/// is why the named-slot contract matters: the stage declares what it takes, and a Run UI
/// builds its form from that rather than from a hand-wired graph.
public struct SeedVR2UpscaleStage: PipelineStage {
    public let id: String
    public let name: String
    public let inputs: [InputSlot]
    public let produces: MediaKind

    let plan: SeedVR2Plan
    let components: SeedVR2Components
    let residency: SeedVR2Residency
    let machine: MachineKey
    let outputDirectory: URL
    let scratchDirectory: URL

    public init(
        plan: SeedVR2Plan,
        components: SeedVR2Components,
        residency: SeedVR2Residency,
        machine: MachineKey,
        outputDirectory: URL,
        scratchDirectory: URL
    ) {
        self.plan = plan
        self.components = components
        self.residency = residency
        self.machine = machine
        self.outputDirectory = outputDirectory
        self.scratchDirectory = scratchDirectory
        let kind: MediaKind = plan.isSingleImage ? .image : .video
        id = "seedvr2.\(plan.variant.rawValue)"
        name = "SeedVR2 \(plan.variant.parameterCountB.formatted())B"
        inputs = [InputSlot(name: plan.isSingleImage ? "image" : "video", kind: kind)]
        produces = kind
    }

    /// The measurements from the most recent run are reported through this callback rather
    /// than mutating the stage, so the stage stays `Sendable`.
    public var onMeasured: (@Sendable (SeedVR2Engine.Measurements) -> Void)?

    public func run(
        _ inputs: [String: Media],
        progress: @Sendable @escaping (StageProgress) -> Void
    ) async throws -> Media {
        try validate(inputs)

        if plan.isSingleImage {
            let media = try require(inputs, "image", .image)
            guard case let .image(image) = media else {
                throw StageError.kindMismatch(slot: "image", expected: .image, got: media.kind)
            }
            let result = try await SeedVR2Engine.upscale(
                image: image.cgImage, plan: plan, components: components,
                residency: residency, machine: machine, progress: progress
            )
            onMeasured?(result.measurements)
            return .image(ImageMedia(cgImage: result.image))
        }

        let media = try require(inputs, "video", .video)
        guard case let .video(video) = media else {
            throw StageError.kindMismatch(slot: "video", expected: .video, got: media.kind)
        }
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let output = outputDirectory.appendingPathComponent(
            video.url.deletingPathExtension().lastPathComponent
                + "-\(plan.outputWidth)x\(plan.outputHeight).mp4"
        )
        let result = try await SeedVR2Engine.upscale(
            video: video, plan: plan, components: components, residency: residency,
            machine: machine, outputURL: output, scratch: scratchDirectory, progress: progress
        )
        onMeasured?(result.measurements)
        return .video(result.video)
    }
}
