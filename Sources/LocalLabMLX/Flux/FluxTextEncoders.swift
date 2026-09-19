//
// Ported from MLXUI (same author, MIT License); MLXUI's port follows mflux (MIT) and
// Hugging Face's T5EncoderModel / CLIPTextModel (Apache 2.0). Changed for LocalLab: the
// module tree follows the checkpoint's own names (`encoder.block.N.layer.M…`), with T5's one
// relative-position table stored at the top, and layers are plain until loading quantizes
// the ones the checkpoint stores quantized.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - T5 (text_encoder_2)

struct FluxT5Config: Sendable {
    var dModel = 4096
    var dFF = 10240
    var heads = 64
    var headDim = 64
    var vocab = 32128
    var layers = 24
    var buckets = 32
    var maxDistance = 128
}

/// T5's layer norm: RMS, a learned scale, no bias or mean.
final class FluxT5LayerNorm: Module {
    let weight: MLXArray

    init(_ dimensions: Int) {
        weight = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: 1e-6)
    }
}

final class FluxT5SelfAttention: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    let heads: Int
    let headDim: Int

    init(_ c: FluxT5Config) {
        heads = c.heads
        headDim = c.headDim
        let inner = c.heads * c.headDim
        _q.wrappedValue = Linear(c.dModel, inner, bias: false)
        _k.wrappedValue = Linear(c.dModel, inner, bias: false)
        _v.wrappedValue = Linear(c.dModel, inner, bias: false)
        _o.wrappedValue = Linear(inner, c.dModel, bias: false)
    }

    /// T5 folds the 1/√d scale into its weights, so attention is unscaled.
    func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        func shape(_ y: MLXArray) -> MLXArray { y.reshaped([b, l, heads, headDim]).transposed(0, 2, 1, 3) }
        let out = MLXFast.scaledDotProductAttention(
            queries: shape(q(x)), keys: shape(k(x)), values: shape(v(x)), scale: 1, mask: .array(bias)
        )
        return o(out.transposed(0, 2, 1, 3).reshaped([b, l, -1]))
    }
}

final class FluxT5AttentionLayer: Module {
    @ModuleInfo(key: "SelfAttention") var attention: FluxT5SelfAttention
    @ModuleInfo(key: "layer_norm") var norm: FluxT5LayerNorm

    init(_ c: FluxT5Config) {
        _attention.wrappedValue = FluxT5SelfAttention(c)
        _norm.wrappedValue = FluxT5LayerNorm(c.dModel)
    }
}

final class FluxT5DenseGated: Module {
    @ModuleInfo(key: "wi_0") var wi0: Linear
    @ModuleInfo(key: "wi_1") var wi1: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(_ c: FluxT5Config) {
        _wi0.wrappedValue = Linear(c.dModel, c.dFF, bias: false)
        _wi1.wrappedValue = Linear(c.dModel, c.dFF, bias: false)
        _wo.wrappedValue = Linear(c.dFF, c.dModel, bias: false)
    }

    /// "gated-gelu": gelu_new(wi_0 x) × wi_1 x.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        wo(geluApproximate(wi0(x)) * wi1(x))
    }
}

final class FluxT5FeedForwardLayer: Module {
    @ModuleInfo(key: "DenseReluDense") var dense: FluxT5DenseGated
    @ModuleInfo(key: "layer_norm") var norm: FluxT5LayerNorm

    init(_ c: FluxT5Config) {
        _dense.wrappedValue = FluxT5DenseGated(c)
        _norm.wrappedValue = FluxT5LayerNorm(c.dModel)
    }
}

/// `encoder.block.N`: `layer.0` is attention, `layer.1` the feed-forward — the checkpoint's
/// list, kept as a list of two differently-shaped modules.
final class FluxT5Block: Module {
    @ModuleInfo(key: "layer") var layer: [Module]

    init(_ c: FluxT5Config) {
        _layer.wrappedValue = [FluxT5AttentionLayer(c), FluxT5FeedForwardLayer(c)]
    }

    func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        guard let attention = layer[0] as? FluxT5AttentionLayer,
              let feedForward = layer[1] as? FluxT5FeedForwardLayer else { return x }
        var h = x + attention.attention(attention.norm(x), bias: bias)
        h = h + feedForward.dense(feedForward.norm(h))
        return h
    }
}

final class FluxT5Stack: Module {
    @ModuleInfo(key: "block") var blocks: [FluxT5Block]
    @ModuleInfo(key: "final_layer_norm") var finalNorm: FluxT5LayerNorm

    init(_ c: FluxT5Config) {
        _blocks.wrappedValue = (0 ..< c.layers).map { _ in FluxT5Block(c) }
        _finalNorm.wrappedValue = FluxT5LayerNorm(c.dModel)
    }
}

final class FluxT5Encoder: Module {
    let config: FluxT5Config
    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "encoder") var encoder: FluxT5Stack
    @ModuleInfo(key: "relative_attention_bias") var relativeBias: Embedding

    init(_ c: FluxT5Config = FluxT5Config()) {
        config = c
        _shared.wrappedValue = Embedding(embeddingCount: c.vocab, dimensions: c.dModel)
        _encoder.wrappedValue = FluxT5Stack(c)
        _relativeBias.wrappedValue = Embedding(embeddingCount: c.buckets, dimensions: c.heads)
    }

    /// Hugging Face's bidirectional `_relative_position_bucket`: half the buckets per
    /// direction, exact up to 8, logarithmic out to `maxDistance`.
    static func bucket(_ relative: Int, buckets: Int = 32, maxDistance: Int = 128) -> Int {
        let half = buckets / 2
        var result = relative > 0 ? half : 0
        let distance = abs(relative)
        let maxExact = half / 2
        if distance < maxExact { return result + distance }
        let scaled = log(Double(distance) / Double(maxExact)) / log(Double(maxDistance) / Double(maxExact))
            * Double(half - maxExact)
        result += min(maxExact + Int(scaled), half - 1)
        return result
    }

    /// (1, heads, L, L) — every layer shares it.
    func positionBias(length: Int) -> MLXArray {
        var indices = [Int32]()
        indices.reserveCapacity(length * length)
        for query in 0 ..< length {
            for key in 0 ..< length {
                indices.append(Int32(Self.bucket(key - query, buckets: config.buckets, maxDistance: config.maxDistance)))
            }
        }
        let table = relativeBias(MLXArray(indices, [length, length]))  // (L, L, heads)
        return table.transposed(2, 0, 1).expandedDimensions(axis: 0)
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var h = shared(tokens)
        let bias = positionBias(length: tokens.dim(1)).asType(h.dtype)
        for block in encoder.blocks { h = block(h, bias: bias) }
        return encoder.finalNorm(h)
    }
}

// MARK: - CLIP (text_encoder)

struct FluxCLIPConfig: Sendable {
    var dModel = 768
    var dFF = 3072
    var heads = 12
    var vocab = 49408
    var positions = 77
    var layers = 12
}

final class FluxCLIPAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var out: Linear
    let heads: Int

    init(_ c: FluxCLIPConfig) {
        heads = c.heads
        _q.wrappedValue = Linear(c.dModel, c.dModel)
        _k.wrappedValue = Linear(c.dModel, c.dModel)
        _v.wrappedValue = Linear(c.dModel, c.dModel)
        _out.wrappedValue = Linear(c.dModel, c.dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l, d) = (x.dim(0), x.dim(1), x.dim(2))
        func shape(_ y: MLXArray) -> MLXArray { y.reshaped([b, l, heads, d / heads]).transposed(0, 2, 1, 3) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: shape(q(x)), keys: shape(k(x)), values: shape(v(x)),
            scale: 1 / Float(d / heads).squareRoot(), mask: .causal
        )
        return out(attended.transposed(0, 2, 1, 3).reshaped([b, l, d]))
    }
}

final class FluxCLIPMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ c: FluxCLIPConfig) {
        _fc1.wrappedValue = Linear(c.dModel, c.dFF)
        _fc2.wrappedValue = Linear(c.dFF, c.dModel)
    }

    /// quick_gelu: x·σ(1.702x).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = fc1(x)
        return fc2(h * sigmoid(1.702 * h))
    }
}

final class FluxCLIPLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: FluxCLIPAttention
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: FluxCLIPMLP
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm

    init(_ c: FluxCLIPConfig) {
        _attention.wrappedValue = FluxCLIPAttention(c)
        _norm1.wrappedValue = LayerNorm(dimensions: c.dModel, eps: 1e-5)
        _mlp.wrappedValue = FluxCLIPMLP(c)
        _norm2.wrappedValue = LayerNorm(dimensions: c.dModel, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(norm1(x))
        return h + mlp(norm2(h))
    }
}

final class FluxCLIPEmbeddings: Module {
    @ModuleInfo(key: "token_embedding") var token: Embedding
    @ModuleInfo(key: "position_embedding") var position: Embedding

    init(_ c: FluxCLIPConfig) {
        _token.wrappedValue = Embedding(embeddingCount: c.vocab, dimensions: c.dModel)
        _position.wrappedValue = Embedding(embeddingCount: c.positions, dimensions: c.dModel)
    }
}

final class FluxCLIPEncoderLayers: Module {
    @ModuleInfo(key: "layers") var layers: [FluxCLIPLayer]

    init(_ c: FluxCLIPConfig) {
        _layers.wrappedValue = (0 ..< c.layers).map { _ in FluxCLIPLayer(c) }
    }
}

final class FluxCLIPTextModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: FluxCLIPEmbeddings
    @ModuleInfo(key: "encoder") var encoder: FluxCLIPEncoderLayers
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(_ c: FluxCLIPConfig) {
        _embeddings.wrappedValue = FluxCLIPEmbeddings(c)
        _encoder.wrappedValue = FluxCLIPEncoderLayers(c)
        _finalNorm.wrappedValue = LayerNorm(dimensions: c.dModel, eps: 1e-5)
    }
}

final class FluxCLIPEncoder: Module {
    @ModuleInfo(key: "text_model") var textModel: FluxCLIPTextModel

    init(_ c: FluxCLIPConfig = FluxCLIPConfig()) {
        _textModel.wrappedValue = FluxCLIPTextModel(c)
    }

    /// The pooled embedding: the final hidden state at the end-of-text token, which is the
    /// highest id in CLIP's vocabulary — so its position is the argmax.
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let length = tokens.dim(1)
        var h = textModel.embeddings.token(tokens)
            + textModel.embeddings.position(MLXArray(Int32(0) ..< Int32(length)).expandedDimensions(axis: 0))
        for layer in textModel.encoder.layers { h = layer(h) }
        h = textModel.finalNorm(h)
        let end = argMax(tokens[0], axis: -1).item(Int.self)
        return h[0 ..< 1, end]
    }
}
