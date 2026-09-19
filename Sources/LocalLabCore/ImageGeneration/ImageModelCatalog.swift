import Foundation

/// A text-to-image model LocalLab can run, with the figures Smart Fit needs — read from the
/// published repo, not derived: component bytes are the sums of each folder's `.safetensors`.
public struct ImageModelSpec: Sendable, Equatable, Identifiable {
    public enum Family: String, Sendable, Codable {
        /// FLUX.1 [schnell]: 19 double + 38 single blocks, no guidance embedding, a 4-step
        /// schedule without the time shift, and a 256-token T5 window.
        case fluxSchnell
    }

    public let repo: String
    public let displayName: String
    public let family: Family
    public let license: String
    /// The MMDiT transformer — the part resident while denoising.
    public let transformerBytes: Int64
    /// T5 plus CLIP — resident only while the prompt is encoded.
    public let textEncoderBytes: Int64
    public let vaeBytes: Int64
    public let defaultSteps: Int
    public let stepRange: ClosedRange<Int>
    public let manifest: ComponentManifest

    public var id: String { repo }
    public var downloadBytes: Int64 { transformerBytes + textEncoderBytes + vaeBytes }
}

public enum ImageModelCatalog {
    /// FLUX.1 [schnell], 4-bit (group 64), converted with `mzbac/flux.swift` — Apache 2.0, as
    /// the original. Its tensor names are diffusers', which the engine reads directly; the
    /// VAE file carries an encoder too, which generation never loads.
    public static let fluxSchnell4bit = ImageModelSpec(
        repo: "mzbac/flux1.schnell.4bit.mlx",
        displayName: "FLUX.1 schnell",
        family: .fluxSchnell,
        license: "apache-2.0",
        transformerBytes: 6_693_752_375,
        textEncoderBytes: 2_992_397_467,
        vaeBytes: 164_654_313,
        defaultSteps: 4,
        stepRange: 1 ... 8,
        manifest: ComponentManifest(
            include: [
                "transformer/*.safetensors",
                "text_encoder/*.safetensors",
                "text_encoder_2/*.safetensors",
                "vae/*.safetensors",
                "tokenizer/vocab.json",
                "tokenizer/merges.txt",
                "tokenizer_2/spiece.model",
            ],
            requiredTensors: [
                "transformer/flux1_mlx_model-00001-of-00004.safetensors": ["context_embedder.weight", "proj_out.weight"],
                "text_encoder/flux1_mlx_model.safetensors": ["text_model.final_layer_norm.weight"],
                "transformer/flux1_mlx_model-00004-of-00004.safetensors": ["x_embedder.weight"],
                "text_encoder_2/flux1_mlx_model-00002-of-00002.safetensors": [
                    "shared.weight", "relative_attention_bias.weight", "encoder.final_layer_norm.weight",
                ],
                "vae/flux1_mlx_model.safetensors": ["decoder.conv_in.weight", "decoder.conv_out.weight"],
            ]
        )
    )

    public static let models: [ImageModelSpec] = [fluxSchnell4bit]

    public static func spec(forRepo repo: String) -> ImageModelSpec? {
        models.first { $0.repo == repo }
    }
}

extension ComponentManifest {
    /// The manifest an install of `repo` should use: an engine's own, where it has one, else
    /// `nil` for the heuristics.
    public static func known(for repo: String) -> ComponentManifest? {
        if SeedVR2Variant.allCases.contains(where: { $0.hfRepo == repo }) { return SeedVR2Variant.manifest }
        return ImageModelCatalog.spec(forRepo: repo)?.manifest
    }
}
