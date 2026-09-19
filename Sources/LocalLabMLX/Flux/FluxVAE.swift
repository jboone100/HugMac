//
// Ported from MLXUI (same author, MIT License); MLXUI's port follows mflux (MIT) and
// diffusers' AutoencoderKL (Apache 2.0). Decode only — the checkpoint's encoder is never
// loaded. Channels-last throughout, as MLX's `Conv2d` expects, and the checkpoint's conv
// weights are already stored that way.
//

import Foundation
import MLX
import MLXFast
import MLXNN

private func groupNorm(_ channels: Int) -> GroupNorm {
    GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, affine: true, pytorchCompatible: true)
}

final class FluxVAEResnet: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var shortcut: Conv2d?

    init(_ input: Int, _ output: Int) {
        _norm1.wrappedValue = groupNorm(input)
        _conv1.wrappedValue = Conv2d(inputChannels: input, outputChannels: output, kernelSize: 3, padding: 1)
        _norm2.wrappedValue = groupNorm(output)
        _conv2.wrappedValue = Conv2d(inputChannels: output, outputChannels: output, kernelSize: 3, padding: 1)
        _shortcut.wrappedValue = input == output ? nil : Conv2d(inputChannels: input, outputChannels: output, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(norm1(x)))
        h = conv2(silu(norm2(h)))
        return (shortcut?(x) ?? x) + h
    }
}

/// Single-head self-attention over every pixel of the mid-block feature map.
final class FluxVAEAttention: Module {
    @ModuleInfo(key: "group_norm") var norm: GroupNorm
    @ModuleInfo(key: "to_q") var q: Linear
    @ModuleInfo(key: "to_k") var k: Linear
    @ModuleInfo(key: "to_v") var v: Linear
    @ModuleInfo(key: "to_out") var out: [Linear]

    init(_ channels: Int) {
        _norm.wrappedValue = groupNorm(channels)
        _q.wrappedValue = Linear(channels, channels)
        _k.wrappedValue = Linear(channels, channels)
        _v.wrappedValue = Linear(channels, channels)
        _out.wrappedValue = [Linear(channels, channels)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let y = norm(x).reshaped([b, h * w, c])
        func head(_ z: MLXArray) -> MLXArray { z.reshaped([b, h * w, 1, c]).transposed(0, 2, 1, 3) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: head(q(y)), keys: head(k(y)), values: head(v(y)),
            scale: 1 / Float(c).squareRoot(), mask: .none
        )
        return x + out[0](attended.transposed(0, 2, 1, 3).reshaped([b, h, w, c]))
    }
}

final class FluxVAEMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [FluxVAEResnet]
    @ModuleInfo(key: "attentions") var attentions: [FluxVAEAttention]

    override init() {
        _resnets.wrappedValue = [FluxVAEResnet(512, 512), FluxVAEResnet(512, 512)]
        _attentions.wrappedValue = [FluxVAEAttention(512)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resnets[1](attentions[0](resnets[0](x)))
    }
}

final class FluxVAEUpsampler: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(_ channels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
    }

    /// Nearest-neighbour 2× (each pixel repeated along both axes), then a 3×3 conv.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let doubled = broadcast(x.reshaped([b, h, 1, w, 1, c]), to: [b, h, 2, w, 2, c])
            .reshaped([b, h * 2, w * 2, c])
        return conv(doubled)
    }
}

final class FluxVAEUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [FluxVAEResnet]
    @ModuleInfo(key: "upsamplers") var upsamplers: [FluxVAEUpsampler]

    init(_ input: Int, _ output: Int, upsample: Bool) {
        _resnets.wrappedValue = [FluxVAEResnet(input, output), FluxVAEResnet(output, output), FluxVAEResnet(output, output)]
        _upsamplers.wrappedValue = upsample ? [FluxVAEUpsampler(output)] : []
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for resnet in resnets { h = resnet(h) }
        if let upsampler = upsamplers.first { h = upsampler(h) }
        return h
    }
}

final class FluxVAEDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var mid: FluxVAEMidBlock
    @ModuleInfo(key: "up_blocks") var up: [FluxVAEUpBlock]
    @ModuleInfo(key: "conv_norm_out") var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    override init() {
        _convIn.wrappedValue = Conv2d(inputChannels: 16, outputChannels: 512, kernelSize: 3, padding: 1)
        _mid.wrappedValue = FluxVAEMidBlock()
        _up.wrappedValue = [
            FluxVAEUpBlock(512, 512, upsample: true),
            FluxVAEUpBlock(512, 512, upsample: true),
            FluxVAEUpBlock(512, 256, upsample: true),
            FluxVAEUpBlock(256, 128, upsample: false),
        ]
        _normOut.wrappedValue = groupNorm(128)
        _convOut.wrappedValue = Conv2d(inputChannels: 128, outputChannels: 3, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = mid(convIn(z))
        for block in up { h = block(h) }
        return convOut(silu(normOut(h)))
    }
}

final class FluxVAE: Module {
    static let scalingFactor: Float = 0.3611
    static let shiftFactor: Float = 0.1159

    @ModuleInfo(key: "decoder") var decoder: FluxVAEDecoder

    override init() {
        _decoder.wrappedValue = FluxVAEDecoder()
    }

    /// Latents (1, h, w, 16) → pixels (1, 8h, 8w, 3) in 0…1.
    func decode(_ latents: MLXArray) -> MLXArray {
        let z = latents / Self.scalingFactor + Self.shiftFactor
        return clip(decoder(z) / 2 + 0.5, min: 0, max: 1)
    }
}
