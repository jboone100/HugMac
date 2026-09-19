import Foundation

/// A chat model LocalLab knows enough about to grade before it is downloaded (plan §5.12).
///
/// The figures are **read from the published repos**, not derived: weight bytes are the sum of
/// the repo's `.safetensors` files, and the attention shape comes from its `config.json` —
/// queried from the Hugging Face API on 2026-09-18 at the revision recorded here.
public struct ChatModelSpec: Sendable, Equatable, Identifiable, Hashable {
    public let repo: String
    public let displayName: String
    public let license: String
    /// The commit the figures were read at.
    public let revision: String
    /// Bytes on disk, and the upper bound on weights resident: the Qwen3.5 checkpoints also
    /// carry a vision tower that a text-only load skips.
    public let weightBytes: Int64
    public let totalParamsB: Double
    /// Parameters read per generated token — all of them for a dense model, the routed
    /// experts' share for a mixture of experts. Token generation streams these through
    /// memory once per token, so they set the speed.
    public let activeParamsB: Double
    /// Layers that keep a key/value cache. Qwen3.5 interleaves three linear-attention layers
    /// (fixed-size state) with one full-attention layer, so only a quarter of its layers grow
    /// with context.
    public let kvLayers: Int
    public let kvHeads: Int
    public let headDim: Int
    public let maxContext: Int
    /// A curated ordering, **not a benchmark** — higher is better. Public benchmark coverage of
    /// MLX quantizations is too sparse to rank on (plan §9.6); this is the implementer's call,
    /// pending the owner's confirmation, and is reviewed like any catalog fact.
    public let qualityScore: Int
    /// The model emits a `<think>` block before answering unless told not to.
    public let thinks: Bool
    /// Its figures and quality score were reviewed (the curated list), rather than
    /// estimated from its repo.
    public var isCurated = true

    public var id: String { repo }

    public var weightGB: Double { Double(weightBytes) / 1_073_741_824 }

    /// A mixture of experts: it reads only part of its weights per token.
    public var isMixtureOfExperts: Bool { activeParamsB < totalParamsB * 0.9 }

    /// Bytes read per generated token.
    public var activeWeightBytes: Double {
        Double(weightBytes) * activeParamsB / max(totalParamsB, 0.001)
    }

    /// fp16 keys and values for `context` tokens.
    public func kvCacheBytes(context: Int) -> Int64 {
        Int64(2 * kvLayers * kvHeads * headDim * 2) * Int64(context)
    }
}

public enum ChatCatalog {
    /// Chat v1: the Qwen3.5 family — seven sizes from 0.8B to 122B, one license (Apache-2.0),
    /// one prompt format, all supported by `mlx-swift-lm` 3.31.3.
    ///
    /// Left out on purpose: gpt-oss, whose "harmony" channels would reach the transcript as
    /// raw control tokens until they are parsed; older Qwen3, superseded here by Qwen3.5.
    public static let models: [ChatModelSpec] = [
        qwen35("0.8B", sha: "da28692b5f139cb0ec58a356b437486b7dac7462", bytes: 625_000_000,
               total: 0.8, active: 0.8, layers: 24, kvHeads: 2, score: 30),
        qwen35("2B", sha: "674aaa7240b91e8012fcad5d791b7dfe5ba90207", bytes: 1_722_000_000,
               total: 2, active: 2, layers: 24, kvHeads: 2, score: 45),
        qwen35("4B", sha: "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c", bytes: 3_034_000_000,
               total: 4, active: 4, layers: 32, kvHeads: 4, score: 60),
        qwen35("9B", sha: "8b2b98c00a6b4d291155e4890773ca8f769aee53", bytes: 5_950_000_000,
               total: 9, active: 9, layers: 32, kvHeads: 4, score: 72),
        qwen35("27B", sha: "45797d2985a12c55e6473686e9ea91b95e959553", bytes: 16_055_000_000,
               total: 27, active: 27, layers: 64, kvHeads: 4, score: 86),
        qwen35("35B-A3B", sha: "1e20fd8d42056f870933bf98ca6211024744f7ec", bytes: 20_392_000_000,
               total: 35, active: 3, layers: 40, kvHeads: 2, score: 82),
        qwen35("122B-A10B", sha: "e9c67b08899964be5fdd069bb1b4bc8907fe68f5", bytes: 69_594_000_000,
               total: 122, active: 10, layers: 48, kvHeads: 2, score: 94),
    ]

    public static func spec(forRepo repo: String) -> ChatModelSpec? {
        models.first { $0.repo == repo }
    }

    static func qwen35(
        _ size: String, sha: String, bytes: Int64, total: Double, active: Double,
        layers: Int, kvHeads: Int, score: Int
    ) -> ChatModelSpec {
        ChatModelSpec(
            repo: "mlx-community/Qwen3.5-\(size)-4bit",
            displayName: "Qwen3.5 \(size)",
            license: "apache-2.0",
            revision: sha,
            weightBytes: bytes,
            totalParamsB: total,
            activeParamsB: active,
            kvLayers: layers / 4,
            kvHeads: kvHeads,
            headDim: 256,
            maxContext: 262_144,
            qualityScore: score,
            thinks: true
        )
    }
}

// MARK: - Models from outside the curated list

extension ChatModelSpec {
    /// A spec for a chat model LocalLab knows only from its repo: its weights and
    /// `config.json`. Quality is **estimated** from size (`isCurated` false): larger models are
    /// usually better within a generation, which is all Smart Fit may assume without a
    /// reviewed score. A mixture of experts reads roughly its routed share per token, with a
    /// floor for the always-active attention and shared layers.
    public static func estimated(
        repo: String, weightBytes: Int64, config: ModelConfigSummary?, parameterCount: Int64? = nil,
        license: String? = nil, revision: String? = nil
    ) -> ChatModelSpec {
        let bits = Double(config?.bits ?? 4)
        let group = Double(config?.groupSize ?? 64)
        let params = parameterCount.map(Double.init)
            ?? Double(weightBytes) / (bits / 8 + 4 / group)
        let totalB = params / 1e9
        var activeB = totalB
        if let experts = config?.experts, let perToken = config?.expertsPerToken, experts > 0 {
            activeB = max(totalB * Double(perToken) / Double(experts), totalB * 0.1)
        }
        let score = Int(min(80, 25 + 12 * log2(max(totalB, 0.5)))) - 5
        let name = String(repo.split(separator: "/").last ?? Substring(repo))
        return ChatModelSpec(
            repo: repo,
            displayName: name,
            license: license ?? "unknown",
            revision: revision ?? "",
            weightBytes: weightBytes,
            totalParamsB: totalB,
            activeParamsB: activeB,
            kvLayers: config?.cacheLayers ?? 32,
            kvHeads: config?.kvHeads ?? 8,
            headDim: config?.headDim ?? 128,
            maxContext: config?.maxContext ?? 32_768,
            qualityScore: score,
            thinks: config?.modelType?.hasPrefix("qwen3") ?? false,
            isCurated: false
        )
    }
}

/// Chat models installed in a library, curated or not — what Chat can offer.
public enum InstalledChatModels {
    public static func specs(in store: ModelStore) -> [ChatModelSpec] {
        let registry = InstallRegistry.load(store)
        var specs = ChatCatalog.models.filter { store.isInstalled(repo: $0.repo) }
        let curated = Set(specs.map(\.repo))
        for (repo, installed) in registry.models where !curated.contains(repo) && store.isInstalled(repo: repo) {
            let directory = store.directory(forRepo: repo)
            guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
                  let config = ModelConfigSummary.parse(data),
                  let type = config.modelType,
                  RunnerSupport.chatModelTypes.contains(type), RunnerSupport.chatExclusions[type] == nil else { continue }
            let weights = installed.files.filter { $0.path.hasSuffix(".safetensors") }.reduce(Int64(0)) { $0 + $1.size }
            specs.append(.estimated(repo: repo, weightBytes: weights, config: config, revision: installed.revision))
        }
        return specs
    }
}
