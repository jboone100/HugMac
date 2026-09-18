import CoreGraphics
import Foundation
import HugMacCore
import MLX
import MLXNN
import MLXRandom

/// SeedVR2 one-step diffusion super-resolution, for a single image **or** a video clip.
///
/// The architecture needed no change to handle video: the VAE is a 3D causal autoencoder
/// (4× temporal compression, hence 4n+1 frame counts) and the transformer does 3-D windowed
/// attention over (t, h, w). MLXUI only ever fed them one frame. What this adds is the work
/// around them — chunking, tiling, blending, staging, checkpointing — all decided in advance
/// by `SeedVR2Resolver` rather than typed in by the user.
///
/// **Phase-major, not chunk-major.** Weights are loaded three times for a whole job (VAE,
/// transformer, VAE) instead of three times per chunk. Latents are tiny — a 5-frame chunk at
/// 1344×768 is 2×96×168×16 at 2 bytes, about 1 MB — so holding every chunk's latents between
/// phases costs megabytes and saves an hour of reloading on a 61-chunk job. It is also the
/// phase structure the reference implementation reports, which makes the two directly
/// comparable.
public enum SeedVR2Engine {

    public static let engineID = SeedVR2Resolver.mlxEngineID

    /// What a finished run measured, for `CalibrationStore`.
    public struct Measurements: Sendable {
        public var samples: [CalibrationSample]
        public var peakBytes: Int64
        public var totalSeconds: Double
    }

    public struct VideoResult: Sendable {
        public let video: VideoMedia
        public let measurements: Measurements
    }

    public struct ImageResult: @unchecked Sendable {
        public let image: CGImage
        public let measurements: Measurements
    }

    // MARK: - Single image

    public static func upscale(
        image: CGImage,
        plan: SeedVR2Plan,
        components: SeedVR2Components,
        residency: SeedVR2Residency,
        machine: MachineKey,
        progress: @Sendable @escaping (StageProgress) -> Void = { _ in }
    ) async throws -> ImageResult {
        try components.verify()
        var samples: [CalibrationSample] = []
        SeedVR2Residency.resetPeak()
        let started = Date()

        let prepared = try prepareFrames(
            [image], plan: plan, sourceWidth: image.width, sourceHeight: image.height
        )

        progress(StageProgress(fraction: 0.05, phase: "vae-encode", unitsDone: 0, unitsTotal: 1))
        beginPhase()
        let encodeStart = Date()
        // `eval` inside each phase: MLX is lazy, and without it the encode ran inside the
        // transformer phase — both models resident at once, and every phase mis-measured.
        let latent = try withVAE(components: components) { vae in
            let encoded = try encode(prepared, vae: vae, tiling: plan.encodeTiling, plan: plan)
            eval(encoded)
            return encoded
        }
        await residency.evict(.vae)
        samples.append(sample(
            "vae-encode", plan: plan, seconds: Date().timeIntervalSince(encodeStart),
            units: workUnits(for: "vae-encode", plan: plan), machine: machine
        ))

        progress(StageProgress(fraction: 0.35, phase: "dit", unitsDone: 0, unitsTotal: 1))
        beginPhase()
        let ditStart = Date()
        let denoised = try withTransformer(components: components) { transformer, textEmbedding in
            try denoise(latent, transformer: transformer, textEmbedding: textEmbedding, seed: plan.seed)
        }
        await residency.evict(.transformer)
        samples.append(sample(
            "dit", plan: plan, seconds: Date().timeIntervalSince(ditStart),
            units: workUnits(for: "dit", plan: plan), machine: machine
        ))

        progress(StageProgress(fraction: 0.6, phase: "vae-decode", unitsDone: 0, unitsTotal: 1))
        beginPhase()
        let decodeStart = Date()
        let pixels = try withVAE(components: components) { vae in
            let decoded = try decode(denoised, vae: vae, tiling: plan.decodeTiling, plan: plan)
            eval(decoded)
            return decoded
        }
        await residency.evict(.vae)
        samples.append(sample(
            "vae-decode", plan: plan, seconds: Date().timeIntervalSince(decodeStart),
            units: workUnits(for: "vae-decode", plan: plan), machine: machine
        ))

        let warmup = SeedVR2Geometry.causalWarmupFrames(chunkLength: 1)
        let frame = try SeedVR2Frames.image(
            from: SeedVR2Frames.frameSlice(pixels, at: warmup)
        )
        let output = SeedVR2Frames.crop(frame, toWidth: plan.outputWidth, height: plan.outputHeight)
        // The peak of the job is the worst phase, not the last one — peaks are reset per
        // phase so each sample measures its own high-water mark.
        let peak = samples.map(\.peakBytes).max() ?? 0
        await residency.evictAll()
        progress(StageProgress(fraction: 1.0, phase: "done", unitsDone: 1, unitsTotal: 1))

        return ImageResult(
            image: output,
            measurements: Measurements(
                samples: samples, peakBytes: peak,
                totalSeconds: Date().timeIntervalSince(started)
            )
        )
    }

    // MARK: - Video

    /// Upscale a clip. `scratch` holds per-chunk latents so a job that dies at chunk 38 of 61
    /// resumes at 38; `resume` reuses whatever is already there.
    public static func upscale(
        video: VideoMedia,
        plan: SeedVR2Plan,
        components: SeedVR2Components,
        residency: SeedVR2Residency,
        machine: MachineKey,
        outputURL: URL,
        scratch: URL,
        resume: Bool = true,
        progress: @Sendable @escaping (StageProgress) -> Void = { _ in }
    ) async throws -> VideoResult {
        try components.verify()
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)

        var samples: [CalibrationSample] = []
        SeedVR2Residency.resetPeak()
        let started = Date()
        let chunks = plan.chunks

        // Progress weights reflect where the time actually goes: the owner's reference run
        // spent 97% of six and a half hours in the two VAE phases and 3% in the transformer.
        let encodeSpan = 0.35, ditSpan = 0.10, decodeSpan = 0.55

        // ── Phase 1: encode every chunk ────────────────────────────────────────────────
        beginPhase()
        let encodeStart = Date()
        var latentURLs: [URL] = []
        var encodedHere = 0
        try await withVAEAsync(components: components) { vae in
            for (index, chunk) in chunks.enumerated() {
                let url = scratch.appendingPathComponent("latent-\(index).safetensors")
                latentURLs.append(url)
                if resume, FileManager.default.fileExists(atPath: url.path) {
                    progress(StageProgress(
                        fraction: encodeSpan * Double(index + 1) / Double(chunks.count),
                        phase: "vae-encode", unitsDone: index + 1, unitsTotal: chunks.count,
                        checkpoint: url
                    ))
                    continue
                }
                try Task.checkCancellation()
                let frames = try await VideoIO.readFrames(
                    from: video.url, startIndex: chunk.start, count: chunk.realLength, fps: video.fps
                )
                guard !frames.isEmpty else {
                    throw StageError.engineFailure(
                        stage: "SeedVR2", detail: "no frames decoded at index \(chunk.start)"
                    )
                }
                let padded = try padFrameCount(frames, to: chunk.length)
                let prepared = try prepareFrames(
                    padded, plan: plan, sourceWidth: video.width, sourceHeight: video.height
                )
                let latent = try encode(prepared, vae: vae, tiling: plan.encodeTiling, plan: plan)
                eval(latent)
                try MLX.save(arrays: ["latent": latent], url: url)
                encodedHere += 1
                progress(StageProgress(
                    fraction: encodeSpan * Double(index + 1) / Double(chunks.count),
                    phase: "vae-encode", unitsDone: index + 1, unitsTotal: chunks.count,
                    checkpoint: url
                ))
                await Task.yield()
            }
        }
        await residency.evict(.vae)
        // A phase is recorded only if it ran in full. A resumed phase skips the chunks it
        // already has, but its work units count them all — recording it would make this Mac
        // look several times faster than it is.
        var jobPeak = SeedVR2Residency.peakMemoryBytes()
        if encodedHere == chunks.count {
            samples.append(sample(
                "vae-encode", plan: plan, seconds: Date().timeIntervalSince(encodeStart),
                units: workUnits(for: "vae-encode", plan: plan), machine: machine
            ))
        }

        // ── Phase 2: denoise every chunk ───────────────────────────────────────────────
        beginPhase()
        let ditStart = Date()
        var denoisedURLs: [URL] = []
        var denoisedHere = 0
        try await withTransformerAsync(components: components) { transformer, textEmbedding in
            for (index, _) in chunks.enumerated() {
                let url = scratch.appendingPathComponent("denoised-\(index).safetensors")
                denoisedURLs.append(url)
                if resume, FileManager.default.fileExists(atPath: url.path) {
                    progress(StageProgress(
                        fraction: encodeSpan + ditSpan * Double(index + 1) / Double(chunks.count),
                        phase: "dit", unitsDone: index + 1, unitsTotal: chunks.count, checkpoint: url
                    ))
                    continue
                }
                try Task.checkCancellation()
                let stored = try MLX.loadArrays(url: latentURLs[index])
                guard let latent = stored["latent"] else {
                    throw StageError.engineFailure(
                        stage: "SeedVR2", detail: "checkpoint \(url.lastPathComponent) has no latent"
                    )
                }
                // Per-chunk seed offset: identical noise on every chunk would correlate its
                // artefacts across cuts, and a fully random one breaks reproducibility.
                let denoised = try denoise(
                    latent, transformer: transformer, textEmbedding: textEmbedding,
                    seed: plan.seed &+ UInt64(index)
                )
                eval(denoised)
                try MLX.save(arrays: ["latent": denoised], url: url)
                denoisedHere += 1
                progress(StageProgress(
                    fraction: encodeSpan + ditSpan * Double(index + 1) / Double(chunks.count),
                    phase: "dit", unitsDone: index + 1, unitsTotal: chunks.count, checkpoint: url
                ))
                await Task.yield()
            }
        }
        await residency.evict(.transformer)
        jobPeak = max(jobPeak, SeedVR2Residency.peakMemoryBytes())
        if denoisedHere == chunks.count {
            samples.append(sample(
                "dit", plan: plan, seconds: Date().timeIntervalSince(ditStart),
                units: workUnits(for: "dit", plan: plan), machine: machine
            ))
        }

        // ── Phase 3: decode, blend seams, write ────────────────────────────────────────
        //
        // Each chunk is written to its own segment, and the frames it holds back for the
        // next chunk's cross-fade are saved beside it. Decode is the longest phase (35 of 55
        // minutes on the reference clip), so a job stopped part way through resumes at the
        // chunk it was on rather than decoding everything again. The segments are joined by
        // passthrough at the end — no re-encode.
        beginPhase()
        let decodeStart = Date()
        var segments: [URL] = []
        // Frames held back to cross-fade with the next chunk's leading frames.
        var pendingTail: [MLXArray] = []
        var decodedHere = 0

        try await withVAEAsync(components: components) { vae in
            for (index, chunk) in chunks.enumerated() {
                let segment = scratch.appendingPathComponent("segment-\(index).mp4")
                let done = scratch.appendingPathComponent("segment-\(index).done")
                if resume, FileManager.default.fileExists(atPath: done.path) {
                    segments.append(segment)
                    pendingTail = []
                    progress(StageProgress(
                        fraction: encodeSpan + ditSpan + decodeSpan * Double(index + 1) / Double(chunks.count),
                        phase: "vae-decode", unitsDone: index + 1, unitsTotal: chunks.count,
                        checkpoint: segment
                    ))
                    continue
                }
                try Task.checkCancellation()

                // Resuming after a finished chunk: its held-back frames are on disk.
                if pendingTail.isEmpty, index > 0, chunk.blendIn > 0 {
                    let tailURL = scratch.appendingPathComponent("tail-\(index - 1).safetensors")
                    if let stored = try? MLX.loadArrays(url: tailURL) {
                        pendingTail = (0 ..< stored.count).compactMap { stored["frame-\($0)"] }
                    }
                }

                let stored = try MLX.loadArrays(url: denoisedURLs[index])
                guard let latent = stored["latent"] else {
                    throw StageError.engineFailure(
                        stage: "SeedVR2", detail: "checkpoint for chunk \(index) has no latent"
                    )
                }
                let pixels = try decode(latent, vae: vae, tiling: plan.decodeTiling, plan: plan)
                eval(pixels)

                // Drop the causal warm-up, then take this chunk's real frames — the tail
                // padding of a short final chunk is discarded with it.
                let warmup = SeedVR2Geometry.causalWarmupFrames(chunkLength: chunk.length)
                let decodedCount = pixels.dim(2)
                guard decodedCount >= warmup + chunk.realLength else {
                    throw StageError.engineFailure(
                        stage: "SeedVR2",
                        detail: "decoder returned \(decodedCount) frames, expected at least "
                            + "\(warmup + chunk.realLength) for a \(chunk.length)-frame chunk"
                    )
                }
                var frames: [MLXArray] = (0 ..< chunk.realLength).map {
                    SeedVR2Frames.frameSlice(pixels, at: warmup + $0)
                }

                // Cross-fade the frames this chunk shares with its predecessor.
                if !pendingTail.isEmpty {
                    let shared = min(pendingTail.count, frames.count)
                    for offset in 0 ..< shared {
                        let alpha = Float(offset + 1) / Float(shared + 1)
                        frames[offset] = pendingTail[offset] * (1 - alpha) + frames[offset] * alpha
                    }
                    pendingTail = []
                }

                // Hold back what the next chunk will blend with.
                let holdBack = index + 1 < chunks.count ? chunks[index + 1].blendIn : 0
                let writeCount = max(frames.count - holdBack, 0)
                if holdBack > 0, writeCount < frames.count {
                    pendingTail = Array(frames[writeCount...])
                }

                let writer = try VideoIO.Writer(
                    url: segment, width: plan.outputWidth, height: plan.outputHeight, fps: video.fps
                )
                for frame in frames.prefix(writeCount) {
                    let image = try SeedVR2Frames.image(from: frame)
                    try writer.append(SeedVR2Frames.crop(
                        image, toWidth: plan.outputWidth, height: plan.outputHeight
                    ))
                }
                try await writer.finish()

                // Tail first, marker last: the marker means everything for this chunk is on
                // disk.
                if !pendingTail.isEmpty {
                    var arrays: [String: MLXArray] = [:]
                    for (offset, frame) in pendingTail.enumerated() { arrays["frame-\(offset)"] = frame }
                    try MLX.save(
                        arrays: arrays,
                        url: scratch.appendingPathComponent("tail-\(index).safetensors")
                    )
                }
                FileManager.default.createFile(atPath: done.path, contents: nil)
                segments.append(segment)
                decodedHere += 1

                progress(StageProgress(
                    fraction: encodeSpan + ditSpan + decodeSpan * Double(index + 1) / Double(chunks.count),
                    phase: "vae-decode", unitsDone: index + 1, unitsTotal: chunks.count,
                    checkpoint: segment
                ))
                await Task.yield()
            }
        }
        await residency.evict(.vae)
        jobPeak = max(jobPeak, SeedVR2Residency.peakMemoryBytes())
        if decodedHere == chunks.count {
            samples.append(sample(
                "vae-decode", plan: plan, seconds: Date().timeIntervalSince(decodeStart),
                units: workUnits(for: "vae-decode", plan: plan), machine: machine
            ))
        }

        let videoOnly = scratch.appendingPathComponent("video-only.mp4")
        try await VideoIO.concatenate(segments, to: videoOnly)

        // The audio track is never re-encoded — it is remuxed untouched.
        if video.hasAudio {
            try await VideoIO.attachAudio(video: videoOnly, audioFrom: video.url, to: outputURL)
        } else if outputURL != videoOnly {
            if FileManager.default.fileExists(atPath: outputURL.path) {
                try? FileManager.default.removeItem(at: outputURL)
            }
            try FileManager.default.moveItem(at: videoOnly, to: outputURL)
        }

        let peak = jobPeak
        await residency.evictAll()
        progress(StageProgress(fraction: 1.0, phase: "done",
                               unitsDone: chunks.count, unitsTotal: chunks.count))

        let result = try await VideoIO.probe(outputURL)
        return VideoResult(
            video: result,
            measurements: Measurements(
                samples: samples, peakBytes: peak,
                totalSeconds: Date().timeIntervalSince(started)
            )
        )
    }

    // MARK: - Frame preparation

    /// Resize to the output size, then pad to the working size. SeedVR2 conditions on an
    /// already-upsampled image, so the bicubic resize is part of the algorithm, not a
    /// convenience.
    static func prepareFrames(
        _ frames: [CGImage], plan: SeedVR2Plan, sourceWidth: Int, sourceHeight: Int
    ) throws -> MLXArray {
        var prepared: [CGImage] = []
        prepared.reserveCapacity(frames.count)
        for frame in frames {
            let resized = try SeedVR2Frames.resize(
                frame, width: plan.outputWidth, height: plan.outputHeight
            )
            prepared.append(try SeedVR2Frames.pad(
                resized, toWidth: plan.paddedWidth, height: plan.paddedHeight
            ))
        }
        return try SeedVR2Frames.tensor(from: prepared)
    }

    /// Repeat the last real frame up to an aligned 4n+1 count.
    static func padFrameCount(_ frames: [CGImage], to count: Int) throws -> [CGImage] {
        guard frames.count < count else { return Array(frames.prefix(count)) }
        guard let last = frames.last else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "no frames to pad")
        }
        return frames + Array(repeating: last, count: count - frames.count)
    }

    // MARK: - VAE

    static func encode(
        _ input: MLXArray, vae: SeedVR2VAE, tiling: VAETiling?, plan: SeedVR2Plan
    ) throws -> MLXArray {
        guard let tiling else { return vae.encode(input) }
        let stride = SeedVR2Geometry.spatialStride
        return try tiled(
            input: input,
            inputHeight: plan.paddedHeight, inputWidth: plan.paddedWidth,
            outputHeight: plan.latentHeight, outputWidth: plan.latentWidth,
            tileSize: tiling.tileSize, overlap: tiling.overlap, ratio: 1.0 / Double(stride),
            temporalMap: { SeedVR2Geometry.latentFrames($0) },
            transform: { vae.encode($0) }
        )
    }

    static func decode(
        _ latent: MLXArray, vae: SeedVR2VAE, tiling: VAETiling?, plan: SeedVR2Plan
    ) throws -> MLXArray {
        guard let tiling else { return vae.decode(latent) }
        let stride = SeedVR2Geometry.spatialStride
        // Tile in latent space so tiles land on exact latent boundaries.
        let latentTile = max(tiling.tileSize / stride, 8)
        let latentOverlap = max(tiling.overlap / stride, 2)
        return try tiled(
            input: latent,
            inputHeight: plan.latentHeight, inputWidth: plan.latentWidth,
            outputHeight: plan.paddedHeight, outputWidth: plan.paddedWidth,
            tileSize: latentTile, overlap: latentOverlap, ratio: Double(stride),
            temporalMap: { SeedVR2Geometry.decodedFrames(latentFrames: $0) },
            transform: { vae.decode($0) }
        )
    }

    /// Spatial tiling with feathered blending, shared by encode and decode.
    ///
    /// Weighted accumulation rather than hard seams: each tile is multiplied by a linear
    /// ramp over its overlap region and summed into a canvas along with its weights, then
    /// divided. A hard seam in a video is visible as a moving edge.
    static func tiled(
        input: MLXArray,
        inputHeight: Int, inputWidth: Int,
        outputHeight: Int, outputWidth: Int,
        tileSize: Int, overlap: Int, ratio: Double,
        temporalMap: (Int) -> Int,
        transform: (MLXArray) -> MLXArray
    ) throws -> MLXArray {
        let step = max(tileSize - overlap, 1)
        let temporalIn = input.dim(2)
        let temporalOut = temporalMap(temporalIn)

        var accumulator: MLXArray?
        var weights: MLXArray?
        var channels = 0

        var y = 0
        while y < inputHeight {
            let tileHeight = min(tileSize, inputHeight - y)
            var x = 0
            while x < inputWidth {
                let tileWidth = min(tileSize, inputWidth - x)
                let slice = input[0..., 0..., 0..., y ..< (y + tileHeight), x ..< (x + tileWidth)]
                let produced = transform(slice)
                eval(produced)
                channels = produced.dim(1)

                let outY = Int((Double(y) * ratio).rounded())
                let outX = Int((Double(x) * ratio).rounded())
                let outH = produced.dim(3), outW = produced.dim(4)
                let ramp = featherWeights(
                    height: outH, width: outW,
                    fadeTop: outY > 0 ? Int((Double(overlap) * ratio).rounded()) : 0,
                    fadeLeft: outX > 0 ? Int((Double(overlap) * ratio).rounded()) : 0,
                    fadeBottom: outY + outH < outputHeight ? Int((Double(overlap) * ratio).rounded()) : 0,
                    fadeRight: outX + outW < outputWidth ? Int((Double(overlap) * ratio).rounded()) : 0
                )
                let padWidths: [IntOrPair] = [
                    IntOrPair((0, 0)),
                    IntOrPair((0, 0)),
                    IntOrPair((0, 0)),
                    IntOrPair((outY, max(outputHeight - outY - outH, 0))),
                    IntOrPair((outX, max(outputWidth - outX - outW, 0))),
                ]
                let weighted = MLX.padded(produced * ramp, widths: padWidths)
                let weightPlane = MLX.padded(
                    MLX.broadcast(ramp, to: [1, 1, temporalOut, outH, outW]), widths: padWidths
                )
                accumulator = accumulator.map { $0 + weighted } ?? weighted
                weights = weights.map { $0 + weightPlane } ?? weightPlane
                eval(accumulator ?? weighted, weights ?? weightPlane)

                if x + tileWidth >= inputWidth { break }
                x += step
            }
            if y + tileHeight >= inputHeight { break }
            y += step
        }

        guard let accumulator, let weights, channels > 0 else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "tiling produced nothing")
        }
        // Guard against a zero weight anywhere the ramp fully faded.
        let safeWeights = MLX.maximum(weights, MLXArray(Float(1e-4)))
        return (accumulator / safeWeights).asType(.bfloat16)
    }

    /// A `[1, 1, 1, H, W]` linear ramp that fades in over each overlapping edge.
    static func featherWeights(
        height: Int, width: Int,
        fadeTop: Int, fadeLeft: Int, fadeBottom: Int, fadeRight: Int
    ) -> MLXArray {
        var rows = [Float](repeating: 1, count: height)
        var columns = [Float](repeating: 1, count: width)
        for index in 0 ..< min(fadeTop, height) {
            rows[index] = Float(index + 1) / Float(fadeTop + 1)
        }
        for index in 0 ..< min(fadeBottom, height) {
            let value = Float(index + 1) / Float(fadeBottom + 1)
            rows[height - 1 - index] = min(rows[height - 1 - index], value)
        }
        for index in 0 ..< min(fadeLeft, width) {
            columns[index] = Float(index + 1) / Float(fadeLeft + 1)
        }
        for index in 0 ..< min(fadeRight, width) {
            let value = Float(index + 1) / Float(fadeRight + 1)
            columns[width - 1 - index] = min(columns[width - 1 - index], value)
        }
        let rowArray = MLXArray(rows, [1, 1, 1, height, 1])
        let columnArray = MLXArray(columns, [1, 1, 1, 1, width])
        return (rowArray * columnArray).asType(.bfloat16)
    }

    // MARK: - Transformer

    static func denoise(
        _ encoded: MLXArray, transformer: SeedVR2Transformer, textEmbedding: MLXArray, seed: UInt64
    ) throws -> MLXArray {
        // Condition: the encoded frames plus an all-ones mask channel.
        let ones = MLXArray.ones(
            [1, 1, encoded.dim(2), encoded.dim(3), encoded.dim(4)]
        ).asType(encoded.dtype)
        let condition = MLX.concatenated([encoded, ones], axis: 1)

        let key = MLXRandom.key(seed)
        var latents = MLXRandom.normal(
            [1, 16, encoded.dim(2), encoded.dim(3), encoded.dim(4)], key: key
        ).asType(.bfloat16)

        // SeedVR2 is a one-step model: at t = T the Euler update is latents − noisePred.
        let timestep = MLXArray(Float(1000)).asType(.bfloat16)
        let modelInput = MLX.concatenated([latents, condition], axis: 1)
        let prediction = transformer(modelInput, textEmb: textEmbedding, timestep: timestep)
        latents = latents - prediction
        eval(latents)
        return latents
    }

    // MARK: - Weight loading

    static func loadVAE(components: SeedVR2Components) throws -> SeedVR2VAE {
        let weights = try MLX.loadArrays(url: components.vaeURL)
        let vae = SeedVR2VAE()
        try vae.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(vae)
        return vae
    }

    static func loadTransformer(components: SeedVR2Components) throws -> (SeedVR2Transformer, MLXArray) {
        let weights = try MLX.loadArrays(url: components.transformerURL)
        var config = SeedVR2Config()
        if let configURL = components.configURL,
           let data = try? Data(contentsOf: configURL),
           let decoded = try? JSONDecoder().decode(SeedVR2Config.self, from: data) {
            config = decoded
        }
        let transformer = SeedVR2Transformer(c: config)
        // Quantized checkpoints carry `.scales` beside the weight; unquantized ones don't.
        quantize(model: transformer) { path, _ in
            weights["\(path).scales"] != nil ? (64, 8, QuantizationMode.affine) : nil
        }
        try transformer.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(transformer)

        let embeddings = try MLX.loadArrays(url: components.positionEmbeddingURL)
        guard let textEmbedding = embeddings["pos_emb.weight"]
            ?? embeddings["weight"] ?? embeddings["freqs"] ?? embeddings.values.first else {
            throw StageError.componentIncomplete(
                model: components.directory.lastPathComponent,
                detail: "pos_emb.safetensors has no embedding tensor"
            )
        }
        eval(textEmbedding)
        return (transformer, textEmbedding)
    }

    static func withVAE<T>(
        components: SeedVR2Components, _ body: (SeedVR2VAE) throws -> T
    ) throws -> T {
        let vae = try loadVAE(components: components)
        return try body(vae)
    }

    static func withVAEAsync(
        components: SeedVR2Components, _ body: (SeedVR2VAE) async throws -> Void
    ) async throws {
        let vae = try loadVAE(components: components)
        try await body(vae)
    }

    static func withTransformer<T>(
        components: SeedVR2Components, _ body: (SeedVR2Transformer, MLXArray) throws -> T
    ) throws -> T {
        let (transformer, textEmbedding) = try loadTransformer(components: components)
        return try body(transformer, textEmbedding)
    }

    static func withTransformerAsync(
        components: SeedVR2Components,
        _ body: (SeedVR2Transformer, MLXArray) async throws -> Void
    ) async throws {
        let (transformer, textEmbedding) = try loadTransformer(components: components)
        try await body(transformer, textEmbedding)
    }

    // MARK: - Measurement

    static func workUnits(for phase: String, plan: SeedVR2Plan) -> Double {
        plan.phases.first { $0.phase == phase }?.workUnits ?? 0
    }

    /// Record what a phase actually cost. `resetPeak()` runs at the *start* of each phase
    /// (see `beginPhase`), so this reads that phase's own high-water mark rather than the
    /// running maximum for the whole job.
    static func sample(
        _ phase: String, plan: SeedVR2Plan, seconds: Double, units: Double, machine: MachineKey
    ) -> CalibrationSample {
        let estimate = plan.phases.first { $0.phase == phase }
        return CalibrationSample(
            engineID: engineID, phase: phase,
            workUnits: units,
            peakUnits: estimate?.peakUnits ?? 0,
            seconds: seconds,
            peakBytes: SeedVR2Residency.peakMemoryBytes(),
            weightBytes: estimate?.weightBytes ?? 0,
            machine: machine,
            note: "\(plan.variant.rawValue), \(plan.chunks.count) chunk(s) of up to \(plan.chunks.map(\.length).max() ?? 0) frames"
        )
    }

    /// Start a phase: clear the allocator's high-water mark so the phase is measured alone.
    static func beginPhase() {
        SeedVR2Residency.resetPeak()
    }
}
