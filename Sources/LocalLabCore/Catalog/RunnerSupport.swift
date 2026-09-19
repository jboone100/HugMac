import Foundation

/// Which of LocalLab's engines can run a model — decided from its `model_type` and task,
/// never guessed from its name.
public enum Runner: Sendable, Equatable {
    case chat
    case upscale(SeedVR2Variant)
    /// LocalLab can't run it yet; the reason says what's missing.
    case notYet(String)

    public var isRunnable: Bool {
        if case .notYet = self { return false }
        return true
    }

    public var label: String {
        switch self {
        case .chat: "Chat"
        case .upscale: "Upscale"
        case .notYet: "Not yet"
        }
    }
}

public enum RunnerSupport {
    /// `model_type`s `mlx-swift-lm` 3.31.3's `LLMModelFactory` loads — copied from its
    /// registry. A VLM whose type is here (Qwen3.5, Gemma 3) chats with its text half.
    public static let chatModelTypes: Set<String> = [
        "mistral", "llama", "phi", "phi3", "phimoe", "gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n",
        "gemma4", "gemma4_text", "qwen2", "qwen3", "qwen3_moe", "qwen3_next", "qwen3_5", "qwen3_5_moe",
        "qwen3_5_text", "minicpm", "starcoder2", "cohere", "openelm", "internlm2", "deepseek_v3", "granite",
        "granitemoehybrid", "mimo", "mimo_v2_flash", "minimax", "glm4", "glm4_moe", "glm4_moe_lite",
        "acereason", "falcon_h1", "bitnet", "smollm3", "ernie4_5", "lfm2", "baichuan_m1", "exaone4",
        "olmoe", "olmo2", "olmo3", "bailing_moe", "lfm2_moe", "nanochat", "nemotron_h", "afmoe",
        "jamba_3b", "mistral3", "apertus",
    ]

    /// `model_type`s `mlx-swift-lm` 3.31.3's `VLMModelFactory` loads with their vision half —
    /// copied from its registry. These chat about images as well as text.
    public static let visionModelTypes: Set<String> = [
        "paligemma", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_5", "qwen3_5_moe", "idefics3",
        "gemma3", "gemma4", "smolvlm", "fastvlm", "llava_qwen2", "pixtral", "mistral3", "lfm2_vl", "glm_ocr",
    ]

    /// Whether a model can be asked about images: its architecture has a vision half the
    /// runtime loads, and its task says it takes images.
    public static func seesImages(_ entry: CatalogEntry) -> Bool {
        guard let type = entry.modelType, visionModelTypes.contains(type) else { return false }
        return entry.task == "image-text-to-text" || entry.task == "any-to-any"
    }

    /// Loadable, but LocalLab can't show their replies properly yet.
    public static let chatExclusions: [String: String] = [
        "gpt_oss": "Its replies use the “harmony” format, which LocalLab doesn't read yet — they'd show as raw control tokens.",
    ]

    public static func runner(for entry: CatalogEntry) -> Runner {
        if let variant = SeedVR2Variant.allCases.first(where: { $0.hfRepo == entry.repo }) {
            return .upscale(variant)
        }
        if let type = entry.modelType {
            if let reason = chatExclusions[type] { return .notYet(reason) }
            let textTasks: Set<String?> = ["text-generation", "image-text-to-text", "conversational", nil]
            if chatModelTypes.contains(type), textTasks.contains(entry.task) { return .chat }
            // Vision-only architectures (Qwen2.5-VL, SmolVLM, …) chat through their vision load.
            if seesImages(entry) { return .chat }
        }
        return .notYet(reason(for: entry.task))
    }

    static func reason(for task: String?) -> String {
        switch task {
        case "image-text-to-text": "LocalLab's vision engine doesn't support this architecture yet."
        case "automatic-speech-recognition": "Speech to text comes in a later phase."
        case "text-to-speech", "text-to-audio": "Speech and audio generation come in a later phase."
        case "text-to-image": "Image generation comes in a later phase."
        case "text-to-video", "image-to-video", "image-text-to-video": "Video generation comes in a later phase."
        case "feature-extraction", "sentence-similarity": "Embeddings come in a later phase."
        case "text-generation": "This architecture isn't one LocalLab's chat engine can load."
        default: "LocalLab has no engine for this kind of model yet."
        }
    }
}

/// A licence, and whether installing it needs the user to read it first (plan §10: licences
/// are data, and are honoured).
public enum LicenseInfo {
    /// Licences with no use restrictions beyond attribution.
    public static let permissive: Set<String> = [
        "apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0", "unlicense", "isc",
    ]

    public static func needsAcknowledgement(_ license: String?) -> Bool {
        guard let license = license?.lowercased() else { return true }
        return !permissive.contains(license)
    }

    public static func displayName(_ license: String?) -> String {
        guard let license, !license.isEmpty else { return "No licence stated" }
        switch license.lowercased() {
        case "apache-2.0": return "Apache 2.0"
        case "mit": return "MIT"
        case "other": return "Custom licence"
        default: return license
        }
    }
}
