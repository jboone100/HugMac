import CoreGraphics
import Foundation
import LocalLabCore
import MLX
import MLXNN
import MLXRandom

/// What to generate.
public struct ImageRequest: Sendable, Equatable {
    public var prompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var seed: UInt64

    public init(prompt: String, width: Int, height: Int, steps: Int, seed: UInt64) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.seed = seed
    }
}

/// Text → image with FLUX.1, one phase at a time.
///
/// The three parts never share memory: the text encoders (3 GB at 4-bit) encode the prompt and
/// are released, the transformer (6.7 GB) denoises and is released, then the VAE decodes. The
/// peak is the transformer plus its activations — not the ~10 GB the whole model weighs —
/// which is what lets a 16 GB Mac run it.
public enum FluxEngine {
    public static let engineID = ImageModelFitter.engineID

    /// Phases, as the progress bar and the calibration store name them. The names follow
    /// `ProbeKind.forPhase`, so another Mac's timings scale by the right probe: the
    /// transformer by matmul throughput, the VAE by convolution.
    public enum Phase: String, Sendable {
        case encode = "text-encode"
        case denoise = "transformer"
        case decode = "vae-decode"
    }

    public struct Result: Sendable {
        public let image: ImageMedia
        public let seconds: Double
        public let peakBytes: Int64
        public let samples: [CalibrationSample]
    }

    /// Sizes are rounded to multiples of 16: the VAE halves three times and the transformer
    /// packs 2×2 latent patches.
    public static func snapped(_ side: Int) -> Int {
        max(256, (side + 8) / 16 * 16)
    }

    public static func generate(
        _ request: ImageRequest, spec: ImageModelSpec, directory: URL,
        progress: @Sendable @escaping (StageProgress) -> Void = { _ in }
    ) async throws -> Result {
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw StageError.modelNotInstalled(id: spec.repo)
        }
        let width = snapped(request.width), height = snapped(request.height)
        let steps = max(1, request.steps)
        let machine = HardwareProfile.detect().machineKey
        let started = Date()
        var samples: [CalibrationSample] = []
        var peak: Int64 = 0
        let previousLimit = Memory.cacheLimit
        Memory.cacheLimit = 1 << 30
        defer { MemoryRelease.returnCachedBuffers(restoringLimitTo: previousLimit) }

        // Progress: encoding ~10%, denoising ~80%, decoding ~10%.
        func report(_ phase: Phase, _ within: Double, done: Int = 0, total: Int = 0) {
            let (start, span): (Double, Double) = switch phase {
            case .encode: (0, 0.1)
            case .denoise: (0.1, 0.8)
            case .decode: (0.9, 0.1)
            }
            progress(StageProgress(fraction: start + span * min(max(within, 0), 1),
                                   phase: phase.rawValue, unitsDone: done, unitsTotal: total))
        }
        // `units` drive time and grow with steps; `peakUnits` drive memory and don't — one
        // step's activations are freed before the next.
        func record(_ phase: Phase, since: Date, units: Double, peakUnits: Double, weights: Int64) {
            let phasePeak = Int64(Memory.peakMemory)
            peak = max(peak, phasePeak)
            samples.append(CalibrationSample(
                engineID: engineID, phase: phase.rawValue, workUnits: units, peakUnits: peakUnits,
                seconds: Date().timeIntervalSince(since), peakBytes: phasePeak, weightBytes: weights,
                machine: machine, chunkFrames: 1, note: "\(width)×\(height), \(steps) steps"
            ))
            MemoryRelease.returnCachedBuffers(restoringLimitTo: 1 << 30)
            Memory.peakMemory = 0
        }

        // 1. The prompt.
        Memory.peakMemory = 0
        var phaseStart = Date()
        report(.encode, 0)
        let tokenizers = try FluxTokenizers(directory: directory)
        let t5Length = spec.family == .fluxSchnell ? 256 : 512
        let (text, pooled): (MLXArray, MLXArray) = try {
            let t5 = FluxT5Encoder()
            try load(t5, from: directory.appendingPathComponent("text_encoder_2"), model: spec.repo)
            let t5IDs = MLXArray(tokenizers.t5Tokens(request.prompt, maxLength: t5Length).map(Int32.init))
            let text = t5(t5IDs.expandedDimensions(axis: 0))
            eval(text)
            let clip = FluxCLIPEncoder()
            try load(clip, from: directory.appendingPathComponent("text_encoder"), model: spec.repo)
            let clipIDs = MLXArray(tokenizers.clipTokens(request.prompt).map(Int32.init))
            let pooled = clip(clipIDs.expandedDimensions(axis: 0))
            eval(pooled)
            return (text.asType(.bfloat16), pooled.asType(.bfloat16))
        }()
        record(.encode, since: phaseStart, units: 1, peakUnits: 1, weights: spec.textEncoderBytes)
        report(.encode, 1)
        try Task.checkCancellation()

        // 2. Denoise.
        phaseStart = Date()
        let rows = height / 16, columns = width / 16
        let latentTokens = rows * columns
        let latents: MLXArray = try {
            let transformer = FluxTransformer(.schnell)
            try load(transformer, from: directory.appendingPathComponent("transformer"), model: spec.repo)
            let rope = transformer.rope(textLength: text.dim(1), rows: rows, columns: columns)
            let key = MLXRandom.key(request.seed)
            // Noise in the unpacked latent shape, then packed: a seed then means the same
            // picture whatever the packing.
            let noise = MLXRandom.normal([1, height / 8, width / 8, 16], key: key).asType(.bfloat16)
            var x = pack(noise)
            let sigmas = schedule(steps: steps)
            for step in 0 ..< steps {
                try Task.checkCancellation()
                let t = MLXArray([Float(sigmas[step] * 1000)]).asType(.bfloat16)
                let velocity = transformer(latents: x, text: text, pooled: pooled, timestep: t,
                                           guidance: nil, rope: rope)
                x = x + Float(sigmas[step + 1] - sigmas[step]) * velocity
                eval(x)
                report(.denoise, Double(step + 1) / Double(steps), done: step + 1, total: steps)
            }
            return unpack(x, rows: rows, columns: columns)
        }()
        record(.denoise, since: phaseStart, units: Double(steps * latentTokens), peakUnits: Double(latentTokens),
               weights: spec.transformerBytes)
        try Task.checkCancellation()

        // 3. Pixels.
        phaseStart = Date()
        let pixels: MLXArray = try {
            let vae = FluxVAE()
            try load(vae, from: directory.appendingPathComponent("vae"), model: spec.repo, prefix: "decoder.")
            let decoded = vae.decode(latents)
            eval(decoded)
            return decoded
        }()
        let image = try cgImage(pixels)
        record(.decode, since: phaseStart, units: Double(width * height), peakUnits: Double(width * height),
               weights: spec.vaeBytes)
        report(.decode, 1)

        return Result(image: ImageMedia(cgImage: image), seconds: Date().timeIntervalSince(started),
                      peakBytes: peak, samples: samples)
    }

    // MARK: - Sampling

    /// schnell's schedule: evenly from 1 to 0, no time shift.
    static func schedule(steps: Int) -> [Double] {
        (0 ... steps).map { 1 - Double($0) / Double(steps) }
    }

    /// (1, H, W, 16) → (1, H/2 · W/2, 64): each 2×2 patch becomes one token, channels first
    /// within the patch — diffusers' `_pack_latents` layout, which the checkpoint was trained on.
    static func pack(_ x: MLXArray) -> MLXArray {
        let (h, w) = (x.dim(1), x.dim(2))
        return x.reshaped([1, h / 2, 2, w / 2, 2, 16])
            .transposed(0, 1, 3, 5, 2, 4)
            .reshaped([1, (h / 2) * (w / 2), 64])
    }

    /// The inverse of `pack`, to (1, H, W, 16) for the VAE.
    static func unpack(_ x: MLXArray, rows: Int, columns: Int) -> MLXArray {
        x.reshaped([1, rows, columns, 16, 2, 2])
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped([1, rows * 2, columns * 2, 16])
    }

    // MARK: - Weights

    /// Load every `.safetensors` in `folder` into `module`. Layers the checkpoint stores
    /// quantized — those with a `.scales` beside the weight — are quantized first, at the
    /// bit width their packed shape implies; the rest stay as they are.
    static func load(_ module: Module, from folder: URL, model: String, prefix: String? = nil) throws {
        let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        var weights: [String: MLXArray] = [:]
        for file in files where file.pathExtension == "safetensors" {
            for (key, value) in try MLX.loadArrays(url: file) where prefix.map(key.hasPrefix) ?? true {
                weights[key] = value
            }
        }
        guard !weights.isEmpty else {
            throw StageError.componentIncomplete(model: model, detail: "\(folder.lastPathComponent)/ has no weights")
        }
        quantize(model: module) { path, layer in
            guard let scales = weights["\(path).scales"], let packed = weights["\(path).weight"] else { return nil }
            let groups = scales.dim(-1)
            let inputs: Int = switch layer {
            case let linear as Linear: linear.weight.dim(-1)
            case let embedding as Embedding: embedding.weight.dim(-1)
            default: 0
            }
            guard groups > 0, inputs > 0 else { return nil }
            let bits = packed.dim(-1) * 32 / inputs
            return (inputs / groups, bits, .affine)
        }
        do {
            try module.update(parameters: ModuleParameters.unflattened(weights), verify: [.allModelKeysSet, .shapeMismatch])
        } catch {
            throw StageError.componentIncomplete(model: model, detail: "\(folder.lastPathComponent)/: \(error)")
        }
        eval(module)
    }

    // MARK: - Pixels

    static func cgImage(_ pixels: MLXArray) throws -> CGImage {
        let (h, w) = (pixels.dim(1), pixels.dim(2))
        let rgb = (pixels[0] * 255).round().asType(.uint8).asArray(UInt8.self)
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        for pixel in 0 ..< w * h {
            rgba[pixel * 4] = rgb[pixel * 3]
            rgba[pixel * 4 + 1] = rgb[pixel * 3 + 1]
            rgba[pixel * 4 + 2] = rgb[pixel * 3 + 2]
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let image = CGImage(
                width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
              ) else {
            throw StageError.engineFailure(stage: "FLUX", detail: "couldn't make an image from the decoded pixels")
        }
        return image
    }
}
