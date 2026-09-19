//
// Ported from MLXUI (same author, MIT License); MLXUI's port follows mflux (MIT) and
// diffusers' FluxTransformer2DModel (Apache 2.0). Changed for LocalLab: the block count and
// guidance embedding come from `FluxTransformerConfig` (schnell has 19 double blocks and no
// guidance; MLXUI's Flux-1.lite had 8 and guidance), and layers are plain `Linear` until
// loading quantizes the ones the checkpoint stores quantized.
//

import Foundation
import MLX
import MLXFast
import MLXNN

struct FluxTransformerConfig: Sendable {
    var hidden = 3072
    var heads = 24
    var headDim = 128
    /// T5 hidden size, into `context_embedder`.
    var contextDim = 4096
    /// CLIP pooled size, into `text_embedder`.
    var pooledDim = 768
    /// 16 latent channels × a 2×2 patch.
    var latentChannels = 64
    var doubleBlocks = 19
    var singleBlocks = 38
    var guidanceEmbeds = false
    var theta: Float = 10_000
    var axesDim = [16, 56, 56]

    static let schnell = FluxTransformerConfig()
}

private let layerNormEps: Float = 1e-6

private func gelu(_ x: MLXArray) -> MLXArray { geluApproximate(x) }

// MARK: - Modulation

/// `AdaLayerNormZero`: six modulation vectors from the conditioning, and the first one
/// applied.
final class FluxAdaNormZero: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    let norm: LayerNorm
    let hidden: Int

    init(_ c: FluxTransformerConfig) {
        hidden = c.hidden
        _linear.wrappedValue = Linear(c.hidden, 6 * c.hidden)
        norm = LayerNorm(dimensions: c.hidden, eps: layerNormEps, affine: false)
    }

    func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> (MLXArray, [MLXArray]) {
        let chunks = split(linear(silu(conditioning)), parts: 6, axis: -1).map { $0.expandedDimensions(axis: 1) }
        // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        return (norm(x) * (1 + chunks[1]) + chunks[0], chunks)
    }
}

final class FluxAdaNormZeroSingle: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    let norm: LayerNorm

    init(_ c: FluxTransformerConfig) {
        _linear.wrappedValue = Linear(c.hidden, 3 * c.hidden)
        norm = LayerNorm(dimensions: c.hidden, eps: layerNormEps, affine: false)
    }

    func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> (MLXArray, MLXArray) {
        let chunks = split(linear(silu(conditioning)), parts: 3, axis: -1).map { $0.expandedDimensions(axis: 1) }
        return (norm(x) * (1 + chunks[1]) + chunks[0], chunks[2])
    }
}

/// The final `AdaLayerNormContinuous`: scale first, then shift.
final class FluxAdaNormContinuous: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    let norm: LayerNorm

    init(_ c: FluxTransformerConfig) {
        _linear.wrappedValue = Linear(c.hidden, 2 * c.hidden)
        norm = LayerNorm(dimensions: c.hidden, eps: layerNormEps, affine: false)
    }

    func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let chunks = split(linear(silu(conditioning)), parts: 2, axis: -1).map { $0.expandedDimensions(axis: 1) }
        return norm(x) * (1 + chunks[0]) + chunks[1]
    }
}

// MARK: - Attention

/// Rotate pairs of the last axis by the rope table `pe` (…, L, D/2, 2, 2).
func fluxApplyRope(_ x: MLXArray, _ pe: MLXArray) -> MLXArray {
    let shape = x.shape
    let pairs = x.asType(.float32).reshaped(Array(shape.dropLast()) + [-1, 1, 2])
    let out = pe[.ellipsis, 0] * pairs[.ellipsis, 0] + pe[.ellipsis, 1] * pairs[.ellipsis, 1]
    return out.reshaped(shape).asType(x.dtype)
}

final class FluxHeads: Module {
    @ModuleInfo(key: "to_q") var q: Linear
    @ModuleInfo(key: "to_k") var k: Linear
    @ModuleInfo(key: "to_v") var v: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    let heads: Int
    let headDim: Int

    init(_ c: FluxTransformerConfig) {
        heads = c.heads
        headDim = c.headDim
        _q.wrappedValue = Linear(c.hidden, c.hidden)
        _k.wrappedValue = Linear(c.hidden, c.hidden)
        _v.wrappedValue = Linear(c.hidden, c.hidden)
        _normQ.wrappedValue = RMSNorm(dimensions: c.headDim, eps: layerNormEps)
        _normK.wrappedValue = RMSNorm(dimensions: c.headDim, eps: layerNormEps)
    }

    /// (B, L, hidden) → three (B, heads, L, headDim), queries and keys normalised.
    func project(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (b, l) = (x.dim(0), x.dim(1))
        func shape(_ y: MLXArray) -> MLXArray {
            y.reshaped([b, l, heads, headDim]).transposed(0, 2, 1, 3)
        }
        return (normQ(shape(q(x))), normK(shape(k(x))), shape(v(x)))
    }
}

func fluxAttention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, rope: MLXArray) -> MLXArray {
    let q = fluxApplyRope(q, rope)
    let k = fluxApplyRope(k, rope)
    let out = MLXFast.scaledDotProductAttention(
        queries: q, keys: k, values: v, scale: 1 / Float(q.dim(3)).squareRoot(), mask: nil
    )
    let (b, l) = (out.dim(0), out.dim(2))
    return out.transposed(0, 2, 1, 3).reshaped([b, l, -1])
}

/// Joint attention over text and image tokens, each with its own projections.
final class FluxJointAttention: Module {
    @ModuleInfo(key: "to_q") var q: Linear
    @ModuleInfo(key: "to_k") var k: Linear
    @ModuleInfo(key: "to_v") var v: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "add_q_proj") var addQ: Linear
    @ModuleInfo(key: "add_k_proj") var addK: Linear
    @ModuleInfo(key: "add_v_proj") var addV: Linear
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear
    let heads: Int
    let headDim: Int

    init(_ c: FluxTransformerConfig) {
        heads = c.heads
        headDim = c.headDim
        _q.wrappedValue = Linear(c.hidden, c.hidden)
        _k.wrappedValue = Linear(c.hidden, c.hidden)
        _v.wrappedValue = Linear(c.hidden, c.hidden)
        _normQ.wrappedValue = RMSNorm(dimensions: c.headDim, eps: layerNormEps)
        _normK.wrappedValue = RMSNorm(dimensions: c.headDim, eps: layerNormEps)
        _toOut.wrappedValue = [Linear(c.hidden, c.hidden)]
        _addQ.wrappedValue = Linear(c.hidden, c.hidden)
        _addK.wrappedValue = Linear(c.hidden, c.hidden)
        _addV.wrappedValue = Linear(c.hidden, c.hidden)
        _normAddedQ.wrappedValue = RMSNorm(dimensions: c.headDim, eps: layerNormEps)
        _normAddedK.wrappedValue = RMSNorm(dimensions: c.headDim, eps: layerNormEps)
        _toAddOut.wrappedValue = Linear(c.hidden, c.hidden)
    }

    private func shape(_ y: MLXArray, _ b: Int, _ l: Int) -> MLXArray {
        y.reshaped([b, l, heads, headDim]).transposed(0, 2, 1, 3)
    }

    func callAsFunction(image: MLXArray, text: MLXArray, rope: MLXArray) -> (MLXArray, MLXArray) {
        let b = image.dim(0), li = image.dim(1), lt = text.dim(1)
        let qi = normQ(shape(q(image), b, li)), ki = normK(shape(k(image), b, li)), vi = shape(v(image), b, li)
        let qt = normAddedQ(shape(addQ(text), b, lt)), kt = normAddedK(shape(addK(text), b, lt))
        let vt = shape(addV(text), b, lt)
        // Text tokens first, as the rope table is laid out.
        let out = fluxAttention(
            concatenated([qt, qi], axis: 2), concatenated([kt, ki], axis: 2),
            concatenated([vt, vi], axis: 2), rope: rope
        )
        let textOut = out[0..., ..<lt]
        let imageOut = out[0..., lt...]
        return (toOut[0](imageOut), toAddOut(textOut))
    }
}

final class FluxFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(_ c: FluxTransformerConfig) {
        _linear1.wrappedValue = Linear(c.hidden, 4 * c.hidden)
        _linear2.wrappedValue = Linear(4 * c.hidden, c.hidden)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}

// MARK: - Blocks

final class FluxDoubleBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: FluxAdaNormZero
    @ModuleInfo(key: "norm1_context") var norm1Context: FluxAdaNormZero
    @ModuleInfo(key: "attn") var attn: FluxJointAttention
    @ModuleInfo(key: "ff") var ff: FluxFeedForward
    @ModuleInfo(key: "ff_context") var ffContext: FluxFeedForward
    let norm2: LayerNorm
    let norm2Context: LayerNorm

    init(_ c: FluxTransformerConfig) {
        _norm1.wrappedValue = FluxAdaNormZero(c)
        _norm1Context.wrappedValue = FluxAdaNormZero(c)
        _attn.wrappedValue = FluxJointAttention(c)
        _ff.wrappedValue = FluxFeedForward(c)
        _ffContext.wrappedValue = FluxFeedForward(c)
        norm2 = LayerNorm(dimensions: c.hidden, eps: layerNormEps, affine: false)
        norm2Context = LayerNorm(dimensions: c.hidden, eps: layerNormEps, affine: false)
    }

    func callAsFunction(
        image: MLXArray, text: MLXArray, conditioning: MLXArray, rope: MLXArray
    ) -> (image: MLXArray, text: MLXArray) {
        let (normImage, mi) = norm1(image, conditioning: conditioning)
        let (normText, mt) = norm1Context(text, conditioning: conditioning)
        let (imageAttn, textAttn) = attn(image: normImage, text: normText, rope: rope)

        func finish(_ x: MLXArray, _ attnOut: MLXArray, _ m: [MLXArray], _ norm: LayerNorm, _ ff: FluxFeedForward) -> MLXArray {
            let h = x + m[2] * attnOut
            return h + m[5] * ff(norm(h) * (1 + m[4]) + m[3])
        }
        return (finish(image, imageAttn, mi, norm2, ff), finish(text, textAttn, mt, norm2Context, ffContext))
    }
}

final class FluxSingleBlock: Module {
    @ModuleInfo(key: "norm") var norm: FluxAdaNormZeroSingle
    @ModuleInfo(key: "attn") var attn: FluxHeads
    @ModuleInfo(key: "proj_mlp") var projMLP: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(_ c: FluxTransformerConfig) {
        _norm.wrappedValue = FluxAdaNormZeroSingle(c)
        _attn.wrappedValue = FluxHeads(c)
        _projMLP.wrappedValue = Linear(c.hidden, 4 * c.hidden)
        _projOut.wrappedValue = Linear(5 * c.hidden, c.hidden)
    }

    func callAsFunction(_ x: MLXArray, conditioning: MLXArray, rope: MLXArray) -> MLXArray {
        let (normed, gate) = norm(x, conditioning: conditioning)
        let (q, k, v) = attn.project(normed)
        let attnOut = fluxAttention(q, k, v, rope: rope)
        let mlp = gelu(projMLP(normed))
        return x + gate * projOut(concatenated([attnOut, mlp], axis: -1))
    }
}

// MARK: - Embeddings

final class FluxEmbedMLP: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(input: Int, hidden: Int) {
        _linear1.wrappedValue = Linear(input, hidden)
        _linear2.wrappedValue = Linear(hidden, hidden)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

final class FluxTimeTextEmbed: Module {
    @ModuleInfo(key: "timestep_embedder") var timestep: FluxEmbedMLP
    @ModuleInfo(key: "guidance_embedder") var guidance: FluxEmbedMLP?
    @ModuleInfo(key: "text_embedder") var text: FluxEmbedMLP

    init(_ c: FluxTransformerConfig) {
        _timestep.wrappedValue = FluxEmbedMLP(input: 256, hidden: c.hidden)
        _guidance.wrappedValue = c.guidanceEmbeds ? FluxEmbedMLP(input: 256, hidden: c.hidden) : nil
        _text.wrappedValue = FluxEmbedMLP(input: c.pooledDim, hidden: c.hidden)
    }

    /// diffusers' `get_timestep_embedding(flip_sin_to_cos: true, downscale_freq_shift: 0)`:
    /// 128 frequencies, cosines first. `t` is already scaled to 0…1000.
    static func sinusoid(_ t: MLXArray, dim: Int = 256) -> MLXArray {
        let half = dim / 2
        let exponent = MLXArray(0 ..< half).asType(.float32) * (-log(Float(10_000)) / Float(half))
        let angles = t.asType(.float32).expandedDimensions(axis: -1) * exp(exponent).expandedDimensions(axis: 0)
        return concatenated([cos(angles), sin(angles)], axis: -1)
    }

    func callAsFunction(timestep t: MLXArray, pooled: MLXArray, guidance g: MLXArray?) -> MLXArray {
        var embedding = timestep(Self.sinusoid(t).asType(pooled.dtype))
        if let guidance, let g {
            embedding = embedding + guidance(Self.sinusoid(g).asType(pooled.dtype))
        }
        return embedding + text(pooled)
    }
}

// MARK: - Transformer

final class FluxTransformer: Module {
    let config: FluxTransformerConfig
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: FluxTimeTextEmbed
    @ModuleInfo(key: "transformer_blocks") var doubleBlocks: [FluxDoubleBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleBlocks: [FluxSingleBlock]
    @ModuleInfo(key: "norm_out") var normOut: FluxAdaNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(_ c: FluxTransformerConfig) {
        config = c
        _xEmbedder.wrappedValue = Linear(c.latentChannels, c.hidden)
        _contextEmbedder.wrappedValue = Linear(c.contextDim, c.hidden)
        _timeTextEmbed.wrappedValue = FluxTimeTextEmbed(c)
        _doubleBlocks.wrappedValue = (0 ..< c.doubleBlocks).map { _ in FluxDoubleBlock(c) }
        _singleBlocks.wrappedValue = (0 ..< c.singleBlocks).map { _ in FluxSingleBlock(c) }
        _normOut.wrappedValue = FluxAdaNormContinuous(c)
        _projOut.wrappedValue = Linear(c.hidden, c.latentChannels)
    }

    /// The rope table for `textLength` text tokens (all at position 0) followed by a
    /// `rows × columns` grid of image patches — (1, 1, L, 64, 2, 2). Built once per image:
    /// it depends only on the sizes.
    func rope(textLength: Int, rows: Int, columns: Int) -> MLXArray {
        let text = MLXArray.zeros([textLength, 3], dtype: .int32)
        let row = broadcast(MLXArray(Int32(0) ..< Int32(rows)).reshaped([rows, 1]), to: [rows, columns])
        let column = broadcast(MLXArray(Int32(0) ..< Int32(columns)).reshaped([1, columns]), to: [rows, columns])
        let image = stacked([MLXArray.zeros([rows, columns], dtype: .int32), row, column], axis: -1)
            .reshaped([rows * columns, 3])
        let ids = concatenated([text, image], axis: 0).asType(.float32)

        let tables = config.axesDim.enumerated().map { axis, dim in
            let scale = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
            let omega = 1 / pow(MLXArray(config.theta), scale)
            let angles = ids[0..., axis].expandedDimensions(axis: -1) * omega
            let (c, s) = (cos(angles), sin(angles))
            return stacked([c, -s, s, c], axis: -1).reshaped([ids.dim(0), dim / 2, 2, 2])
        }
        return concatenated(tables, axis: 1).reshaped([1, 1, ids.dim(0), config.headDim / 2, 2, 2])
    }

    /// One velocity prediction. `latents` are packed (1, L, 64); `timestep` is 0…1000.
    func callAsFunction(
        latents: MLXArray, text: MLXArray, pooled: MLXArray, timestep: MLXArray,
        guidance: MLXArray?, rope: MLXArray
    ) -> MLXArray {
        var image = xEmbedder(latents)
        var context = contextEmbedder(text)
        let conditioning = timeTextEmbed(timestep: timestep, pooled: pooled, guidance: guidance)
        for block in doubleBlocks {
            (image, context) = block(image: image, text: context, conditioning: conditioning, rope: rope)
        }
        let textLength = context.dim(1)
        var joined = concatenated([context, image], axis: 1)
        for block in singleBlocks {
            joined = block(joined, conditioning: conditioning, rope: rope)
        }
        image = joined[0..., textLength...]
        return projOut(normOut(image, conditioning: conditioning))
    }
}
