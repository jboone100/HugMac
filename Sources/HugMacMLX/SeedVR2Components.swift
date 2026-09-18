import Foundation
import HugMacCore
import MLX

/// An installed SeedVR2 checkpoint on disk, verified before anything loads.
///
/// Node-graph tools show what the alternative costs: runs that fail on a model folder
/// that isn't where the loader looks, on a missing `processor` directory,
/// and — worst — on a *present but incomplete* file (`preview TAE is missing decoder
/// tensors`). So completeness is checked **by tensor name**, not by file existence, and it is
/// checked before a multi-hour job starts rather than partway in.
public struct SeedVR2Components: Sendable {
    public let directory: URL
    public let transformerURL: URL
    public let vaeURL: URL
    public let positionEmbeddingURL: URL
    public let configURL: URL?

    public init(directory: URL) {
        self.directory = directory
        transformerURL = directory.appendingPathComponent("transformer.safetensors")
        vaeURL = directory.appendingPathComponent("vae.safetensors")
        positionEmbeddingURL = directory.appendingPathComponent("pos_emb.safetensors")
        let config = directory.appendingPathComponent("config.json")
        configURL = FileManager.default.fileExists(atPath: config.path) ? config : nil
    }

    /// Tensors that must exist for each component to be loadable at all.
    /// Names read from the published checkpoint's safetensors header, not guessed: the VAE's
    /// convolutions are stored flat (`encoder.conv_in.weight`), without the `.conv.` segment
    /// the Swift module nesting might suggest.
    static let requiredVAETensors = [
        "encoder.conv_in.weight",
        "decoder.conv_out.weight",
    ]
    static let requiredTransformerTensors = [
        "vid_in.proj.weight",
        "vid_out.proj.weight",
    ]

    /// Throws `StageError.componentIncomplete` naming what is missing.
    public func verify() throws {
        let fileManager = FileManager.default
        for url in [transformerURL, vaeURL, positionEmbeddingURL] {
            guard fileManager.fileExists(atPath: url.path) else {
                throw StageError.componentIncomplete(
                    model: directory.lastPathComponent,
                    detail: "\(url.lastPathComponent) is missing"
                )
            }
        }
        try Self.verifyTensors(in: vaeURL, expecting: Self.requiredVAETensors, label: "VAE")
        try Self.verifyTensors(
            in: transformerURL, expecting: Self.requiredTransformerTensors, label: "transformer"
        )
    }

    static func verifyTensors(in url: URL, expecting: [String], label: String) throws {
        let arrays: [String: MLXArray]
        do {
            arrays = try MLX.loadArrays(url: url)
        } catch {
            throw StageError.componentIncomplete(
                model: url.deletingLastPathComponent().lastPathComponent,
                detail: "\(url.lastPathComponent) could not be read (\(error.localizedDescription))"
            )
        }
        guard !arrays.isEmpty else {
            throw StageError.componentIncomplete(
                model: url.deletingLastPathComponent().lastPathComponent,
                detail: "\(url.lastPathComponent) contains no tensors"
            )
        }
        // Prefixes rather than exact keys: a quantized checkpoint appends `.scales`/`.biases`
        // and some conversions nest differently, but the stem is stable.
        for tensor in expecting {
            let stem = tensor.replacingOccurrences(of: ".weight", with: "")
            let found = arrays.keys.contains { $0 == tensor || $0.hasPrefix(stem) }
            guard found else {
                throw StageError.componentIncomplete(
                    model: url.deletingLastPathComponent().lastPathComponent,
                    detail: "the \(label) in \(url.lastPathComponent) has no '\(tensor)'"
                )
            }
        }
    }
}
