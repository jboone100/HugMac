import Foundation

/// One model as the Hugging Face search API describes it — enough to grade it for this Mac
/// before anything is downloaded (plan §5.2, §9.1).
public struct CatalogEntry: Sendable, Equatable, Identifiable, Codable {
    public let repo: String
    public let downloads: Int
    public let likes: Int
    public let lastModified: Date?
    /// `text-generation`, `image-text-to-text`, `automatic-speech-recognition`, …
    public let task: String?
    /// `config.json`'s `model_type` — what decides which engine could run it.
    public let modelType: String?
    public let bits: Int?
    public let license: String?
    public let gated: Bool
    public let baseModel: String?
    /// Parameter counts by stored type, as the API reports them. For MLX-quantized weights
    /// the `U32` count is the number of *parameters* packed into them, not of U32 words.
    public let parameters: [String: Int64]

    public var id: String { repo }
    public var publisher: String { String(repo.split(separator: "/").first ?? "") }
    public var name: String { String(repo.split(separator: "/").last ?? Substring(repo)) }

    public init(
        repo: String, downloads: Int = 0, likes: Int = 0, lastModified: Date? = nil, task: String? = nil,
        modelType: String? = nil, bits: Int? = nil, license: String? = nil, gated: Bool = false,
        baseModel: String? = nil, parameters: [String: Int64] = [:]
    ) {
        self.repo = repo
        self.downloads = downloads
        self.likes = likes
        self.lastModified = lastModified
        self.task = task
        self.modelType = modelType
        self.bits = bits
        self.license = license
        self.gated = gated
        self.baseModel = baseModel
        self.parameters = parameters
    }

    public var parameterCount: Int64 { parameters.values.reduce(0, +) }

    /// Bytes on disk, estimated from the parameter counts: quantized parameters at `bits`
    /// each plus the per-group scale and bias MLX stores beside them (4 bytes per 64), and the
    /// rest at their own width. Matched real repos to a few MB for 4- and 8-bit affine
    /// quantization; other schemes (MXFP4) run a few percent low. Nil when the API gave no
    /// counts — the exact size comes with the model's file list.
    public var estimatedWeightBytes: Int64? {
        guard !parameters.isEmpty else { return nil }
        var bytes = 0.0
        for (type, count) in parameters {
            let n = Double(count)
            switch type {
            case "U32": bytes += n * (Double(bits ?? 4) / 8 + 4.0 / 64)
            case "F32", "I32": bytes += n * 4
            case "BF16", "F16", "I16": bytes += n * 2
            case "U8", "I8", "F8_E4M3", "F8_E5M2": bytes += n
            case "F64", "I64": bytes += n * 8
            default: bytes += n * 2
            }
        }
        return Int64(bytes)
    }

    // MARK: - Parsing

    /// One element of `GET /api/models?…&expand[]=…`. Tolerant: any field may be missing, and
    /// `gated` is `false` or a string.
    public static func parse(_ object: [String: Any]) -> CatalogEntry? {
        guard let repo = object["id"] as? String else { return nil }
        let card = object["cardData"] as? [String: Any] ?? [:]
        let config = object["config"] as? [String: Any] ?? [:]
        let quantization = (config["quantization_config"] as? [String: Any]) ?? (config["quantization"] as? [String: Any])
        let tags = object["tags"] as? [String] ?? []
        let license = (card["license"] as? String)
            ?? tags.first { $0.hasPrefix("license:") }.map { String($0.dropFirst("license:".count)) }
        let base: String? = (card["base_model"] as? String) ?? (card["base_model"] as? [String])?.first
        var parameters: [String: Int64] = [:]
        if let counts = (object["safetensors"] as? [String: Any])?["parameters"] as? [String: Any] {
            for (type, value) in counts {
                if let number = value as? NSNumber { parameters[type] = number.int64Value }
            }
        }
        let gated: Bool = (object["gated"] as? Bool) ?? ((object["gated"] as? String).map { $0 != "false" } ?? false)
        return CatalogEntry(
            repo: repo,
            downloads: (object["downloads"] as? NSNumber)?.intValue ?? 0,
            likes: (object["likes"] as? NSNumber)?.intValue ?? 0,
            lastModified: (object["lastModified"] as? String).flatMap(Self.date),
            task: (object["pipeline_tag"] as? String) ?? (card["pipeline_tag"] as? String),
            modelType: config["model_type"] as? String,
            bits: (quantization?["bits"] as? NSNumber)?.intValue,
            license: license,
            gated: gated,
            baseModel: base,
            parameters: parameters
        )
    }

    static func date(_ text: String) -> Date? {
        (try? Date.ISO8601FormatStyle(includingFractionalSeconds: true).parse(text))
            ?? (try? Date.ISO8601FormatStyle().parse(text))
    }
}

/// What opening a model adds: its exact size from the file list, and the attention shape
/// from `config.json` that the context-cache arithmetic needs.
public struct CatalogDetails: Sendable, Equatable, Codable {
    public let repo: String
    public let revision: String?
    public let weightBytes: Int64
    public let downloadBytes: Int64
    public let config: ModelConfigSummary?

    public init(repo: String, revision: String?, weightBytes: Int64, downloadBytes: Int64, config: ModelConfigSummary?) {
        self.repo = repo
        self.revision = revision
        self.weightBytes = weightBytes
        self.downloadBytes = downloadBytes
        self.config = config
    }
}

/// The parts of a model's `config.json` that decide memory: layers, the key/value shape,
/// experts, and quantization. Reads either the top level or a VLM's `text_config`.
public struct ModelConfigSummary: Sendable, Equatable, Codable {
    public let modelType: String?
    public let layers: Int
    /// Layers that keep a growing key/value cache — all of them, except in hybrid models
    /// (Qwen3.5) whose linear-attention layers keep a fixed-size state.
    public let cacheLayers: Int
    public let kvHeads: Int
    public let headDim: Int
    public let maxContext: Int
    public let experts: Int?
    public let expertsPerToken: Int?
    public let bits: Int?
    public let groupSize: Int?

    public init(modelType: String?, layers: Int, cacheLayers: Int, kvHeads: Int, headDim: Int, maxContext: Int,
                experts: Int?, expertsPerToken: Int?, bits: Int?, groupSize: Int?) {
        self.modelType = modelType
        self.layers = layers
        self.cacheLayers = cacheLayers
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.maxContext = maxContext
        self.experts = experts
        self.expertsPerToken = expertsPerToken
        self.bits = bits
        self.groupSize = groupSize
    }

    public static func parse(_ data: Data) -> ModelConfigSummary? {
        guard let top = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        let text = (top["text_config"] as? [String: Any]).map { top.merging($0) { _, inner in inner } } ?? top
        func int(_ key: String) -> Int? { (text[key] as? NSNumber)?.intValue }
        guard let layers = int("num_hidden_layers") ?? int("n_layer") ?? int("num_layers") else { return nil }
        let heads = int("num_attention_heads") ?? int("n_head") ?? 1
        let kvHeads = int("num_key_value_heads") ?? heads
        let headDim = int("head_dim") ?? ((int("hidden_size") ?? 0) / max(heads, 1))
        let layerTypes = text["layer_types"] as? [String]
        let fullAttention = layerTypes.map { $0.filter { $0.contains("full") || $0 == "attention" }.count }
        let quantization = (top["quantization"] as? [String: Any]) ?? (top["quantization_config"] as? [String: Any])
        return ModelConfigSummary(
            modelType: (top["model_type"] as? String) ?? (text["model_type"] as? String),
            layers: layers,
            cacheLayers: fullAttention.flatMap { $0 > 0 ? $0 : nil } ?? layers,
            kvHeads: kvHeads,
            headDim: max(headDim, 1),
            maxContext: int("max_position_embeddings") ?? 32_768,
            experts: int("num_experts") ?? int("num_local_experts") ?? int("n_routed_experts"),
            expertsPerToken: int("num_experts_per_tok") ?? int("num_experts_per_token"),
            bits: (quantization?["bits"] as? NSNumber)?.intValue,
            groupSize: (quantization?["group_size"] as? NSNumber)?.intValue
        )
    }
}
