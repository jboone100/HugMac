import Foundation

// MARK: - Model variants

/// The SeedVR2 checkpoints HugMac can install, cheapest first.
///
/// MLX conversions: `mlx-community/SeedVR2-3B-mlx(-int8)` and `benc0/SeedVR2-7B-mlx(-int8)`.
public enum SeedVR2Variant: String, Sendable, Codable, CaseIterable {
    case threeBInt8 = "SeedVR2-3B-int8"
    case threeBFP16 = "SeedVR2-3B-fp16"
    case sevenBInt8 = "SeedVR2-7B-int8"
    case sevenBFP16 = "SeedVR2-7B-fp16"

    public var parameterCountB: Double {
        switch self {
        case .threeBInt8, .threeBFP16: 3.0
        case .sevenBInt8, .sevenBFP16: 7.0
        }
    }

    public var bytesPerWeight: Double {
        switch self {
        case .threeBInt8, .sevenBInt8: 1.0
        case .threeBFP16, .sevenBFP16: 2.0
        }
    }

    /// Transformer weights resident during the denoise phase.
    ///
    /// **Measured from the published repos, not derived from the parameter count.** A
    /// `params × bytes-per-weight` estimate is wrong by a wide margin here: the 3B int8
    /// transformer is 4.22 GB where the arithmetic says 3.0 GB, and its fp16 sibling is
    /// 7.94 GB where the arithmetic says 6.0 GB — quantization is applied per-layer, and
    /// some tensors stay at higher precision. Guessing would have under-predicted the
    /// denoise phase's peak by more than a gigabyte on every plan.
    public var transformerBytes: Int64 {
        switch self {
        case .threeBInt8: 4_220_000_000
        case .threeBFP16: 7_940_000_000
        case .sevenBInt8: 8_760_000_000
        case .sevenBFP16: 16_480_000_000
        }
    }

    /// Bytes to download for this variant: the transformer plus the shared VAE and the
    /// position embedding.
    public var downloadBytes: Int64 {
        transformerBytes + SeedVR2Geometry.vaeBytes
    }

    /// Quality order, best last — the ladder the resolver walks down.
    public static let byQualityDescending: [SeedVR2Variant] =
        [.sevenBFP16, .sevenBInt8, .threeBFP16, .threeBInt8]

    /// Exactly the four files the engine loads, and the tensors that prove each weight file
    /// is the right one. Names read from the published checkpoint's safetensors header.
    public static let manifest = ComponentManifest(
        include: ["config.json", "pos_emb.safetensors", "transformer.safetensors", "vae.safetensors"],
        requiredTensors: [
            "vae.safetensors": ["encoder.conv_in.weight", "decoder.conv_out.weight"],
            "transformer.safetensors": ["vid_in.proj.weight", "vid_out.proj.weight"],
        ]
    )

    public var hfRepo: String {
        switch self {
        case .threeBFP16: "mlx-community/SeedVR2-3B-mlx"
        case .threeBInt8: "mlx-community/SeedVR2-3B-mlx-int8"
        case .sevenBFP16: "benc0/SeedVR2-7B-mlx"
        case .sevenBInt8: "benc0/SeedVR2-7B-mlx-int8"
        }
    }
}

// MARK: - Geometry constants

/// Fixed properties of the SeedVR2 architecture, as ported in `HugMacMLX`.
public enum SeedVR2Geometry {
    /// VAE spatial stride: 3 spatial downsamples of 2.
    public static let spatialStride = 8
    /// Transformer spatial patch size (`patch_size` = [1, 2, 2]).
    public static let spatialPatch = 2
    /// VAE + patch combined stride. Frame dimensions are padded to a multiple of this,
    /// because an odd latent dimension crashes `patchIn`.
    public static let alignment = spatialStride * spatialPatch  // 16
    /// Temporal compression: the encoder has 2 temporal downsample levels, so 4×, causal —
    /// latent frames = 1 + (frames − 1) / 4. This is why frame counts must be 4n+1.
    public static let temporalStride = 4
    /// VAE weights, fp16 (measured: 250 parameters, 478 MB).
    public static let vaeBytes: Int64 = 501_000_000

    /// The nearest 4n+1 frame count at or above `frames`.
    public static func alignFramesUp(_ frames: Int) -> Int {
        guard frames > 1 else { return 1 }
        let n = Int(ceil(Double(frames - 1) / Double(temporalStride)))
        return n * temporalStride + 1
    }

    /// The nearest 4n+1 frame count at or below `frames`.
    public static func alignFramesDown(_ frames: Int) -> Int {
        guard frames > 1 else { return 1 }
        return ((frames - 1) / temporalStride) * temporalStride + 1
    }

    public static func isAligned(_ frames: Int) -> Bool {
        frames >= 1 && (frames - 1) % temporalStride == 0
    }

    public static func latentFrames(_ frames: Int) -> Int {
        1 + (max(frames, 1) - 1) / temporalStride
    }

    /// Frames the decoder emits for a given latent frame count.
    ///
    /// The decoder's two temporal upsample levels multiply frames by 4 — it does **not**
    /// return `4n+1`. A single latent frame is the exception: each upsample trims a
    /// one-frame input back to one frame, so `T_latent == 1` decodes to exactly one frame
    /// (which is why the single-image path never noticed).
    public static func decodedFrames(latentFrames: Int) -> Int {
        latentFrames <= 1 ? 1 : latentFrames * temporalStride
    }

    /// Leading frames the decoder emits as causal warm-up, which must be discarded.
    ///
    /// A causal VAE's first latent frame carries one real frame while the upsamplers still
    /// produce four, so the first three decoded frames are padding. Concretely: 5 frames →
    /// 2 latent frames → 8 decoded → drop 3 → 5. Getting this wrong silently shifts the
    /// whole clip three frames and mixes padding into the output.
    public static func causalWarmupFrames(chunkLength: Int) -> Int {
        max(decodedFrames(latentFrames: latentFrames(chunkLength)) - chunkLength, 0)
    }

    public static func alignUp(_ value: Int, to multiple: Int) -> Int {
        (value + multiple - 1) / multiple * multiple
    }
}

// MARK: - Chunking

/// One group of frames handed to the model together.
///
/// `length` is always 4n+1. Where the tail of a clip doesn't divide evenly, the last chunk
/// repeats its final real frame (`padTail`) rather than shortening below an aligned count;
/// the padding is discarded on the way out.
public struct FrameChunk: Sendable, Equatable {
    /// Index of this chunk's first frame in the source.
    public let start: Int
    /// Frames fed to the model, always 4n+1.
    public let length: Int
    /// Trailing frames that repeat the last real frame.
    public let padTail: Int
    /// Leading frames shared with the previous chunk, cross-faded on output.
    public let blendIn: Int

    public init(start: Int, length: Int, padTail: Int, blendIn: Int) {
        self.start = start
        self.length = length
        self.padTail = padTail
        self.blendIn = blendIn
    }

    /// Real source frames in this chunk (excluding padding).
    public var realLength: Int { length - padTail }
    /// One past this chunk's last real source frame.
    public var end: Int { start + realLength }
    public var latentFrames: Int { SeedVR2Geometry.latentFrames(length) }
}

// MARK: - VAE tiling

/// Spatial tiling for the VAE. `nil` means whole frames in one pass, which is what the
/// resolver prefers whenever it fits.
///
/// The owner's ComfyUI run tiled at 512 px with 128 px overlap, which cut a 1344×768 frame
/// into 8 heavily overlapping tiles — 2× the pixel work of one pass — on the two phases that
/// accounted for 97% of the run time, on a machine with 17+ GB free.
public struct VAETiling: Sendable, Equatable {
    public let tileSize: Int
    public let overlap: Int

    public init(tileSize: Int, overlap: Int) {
        self.tileSize = tileSize
        self.overlap = overlap
    }

    /// Tiles needed to cover `width × height`, and the pixel area actually processed.
    public func tileCount(width: Int, height: Int) -> Int {
        let step = Swift.max(tileSize - overlap, 1)
        let across = Swift.max(1, Int(ceil(Double(Swift.max(width - overlap, 1)) / Double(step))))
        let down = Swift.max(1, Int(ceil(Double(Swift.max(height - overlap, 1)) / Double(step))))
        return across * down
    }

    public func processedPixels(width: Int, height: Int) -> Int {
        tileCount(width: width, height: height) * tileSize * tileSize
    }
}

// MARK: - Phase estimate

public struct PhaseEstimate: Sendable, Equatable {
    public let phase: String
    public let peakBytes: Int64
    public let workUnits: Double
    /// The largest single piece of work — what peak memory actually scales with.
    public let peakUnits: Double
    /// Weights resident during this phase, so a measurement can separate them out.
    public let weightBytes: Int64
    public let time: TimeEstimate
    /// Whether `peakBytes` came from a measurement on this machine or from the fitted
    /// fallback coefficients.
    public let peakIsMeasured: Bool

    public init(
        phase: String,
        peakBytes: Int64,
        workUnits: Double,
        peakUnits: Double = 0,
        weightBytes: Int64 = 0,
        time: TimeEstimate,
        peakIsMeasured: Bool = false
    ) {
        self.phase = phase
        self.peakBytes = peakBytes
        self.workUnits = workUnits
        self.peakUnits = peakUnits
        self.weightBytes = weightBytes
        self.time = time
        self.peakIsMeasured = peakIsMeasured
    }

    public var peakGB: Double { Double(peakBytes) / 1_073_741_824 }
}

// MARK: - The plan

/// Everything the engine needs to run, and everything the preflight card needs to show —
/// derived from the machine, the input and the user's two choices.
public struct SeedVR2Plan: Sendable, Equatable {
    public let variant: SeedVR2Variant
    /// Output size the user asked for.
    public let outputWidth: Int
    public let outputHeight: Int
    /// Working size: output padded up to a multiple of 16, cropped back on the way out.
    public let paddedWidth: Int
    public let paddedHeight: Int
    public let encodeTiling: VAETiling?
    public let decodeTiling: VAETiling?
    public let chunks: [FrameChunk]
    public let temporalOverlap: Int
    public let seed: UInt64
    public let phases: [PhaseEstimate]
    public let reasons: [SettingReason]

    public var latentWidth: Int { paddedWidth / SeedVR2Geometry.spatialStride }
    public var latentHeight: Int { paddedHeight / SeedVR2Geometry.spatialStride }
    public var isSingleImage: Bool { chunks.count == 1 && chunks[0].length == 1 }
    public var frameCount: Int { chunks.last?.end ?? 0 }

    /// The largest phase peak — what has to fit.
    public var peakBytes: Int64 { phases.map(\.peakBytes).max() ?? 0 }
    public var peakGB: Double { Double(peakBytes) / 1_073_741_824 }

    /// Total time, or `.unknown` if any phase is unmeasured on this machine.
    public var totalTime: TimeEstimate {
        var total = 0.0
        for phase in phases {
            guard let seconds = phase.time.seconds else { return .unknown }
            total += seconds
        }
        return .measured(seconds: total)
    }
}

// MARK: - The resolver

/// Turns intent plus hardware into a plan: picks the variant, the temporal chunk length, the
/// tiling, and the overlap — the settings a ComfyUI user fills in by hand.
///
/// The ladder is walked in **quality order**, and the first candidate whose largest phase
/// fits the plannable memory budget wins. Candidates are ordered so the expensive-but-better
/// choices are tried first: bigger model, then longer temporal chunks (better coherence,
/// fewer seams), then less tiling (much less redundant convolution work).
public struct SeedVR2Resolver: Sendable {
    public let hardware: HardwareProfile
    public let calibration: CalibrationStore
    /// The engine whose measurements apply. Defaults to HugMac's MLX engine.
    public let engineID: String

    public init(
        hardware: HardwareProfile,
        calibration: CalibrationStore = CalibrationStore(),
        engineID: String = SeedVR2Resolver.mlxEngineID
    ) {
        self.hardware = hardware
        self.calibration = calibration
        self.engineID = engineID
    }

    public static let mlxEngineID = "hugmac-seedvr2-mlx"

    /// Temporal chunk lengths to try, longest first. All 4n+1. 1 is the single-image case.
    public static let chunkLadder = [241, 121, 61, 33, 21, 13, 9, 5, 1]
    /// Tiling options, least tiling first. `nil` is one whole-frame pass.
    public static let tilingLadder: [VAETiling?] = [
        nil,
        VAETiling(tileSize: 1024, overlap: 96),
        VAETiling(tileSize: 768, overlap: 96),
        VAETiling(tileSize: 512, overlap: 128),
        VAETiling(tileSize: 384, overlap: 64),
    ]

    public enum Source: Sendable, Equatable {
        case image(width: Int, height: Int)
        case video(VideoMedia)

        var width: Int {
            switch self {
            case .image(let w, _): w
            case .video(let v): v.width
            }
        }
        var height: Int {
            switch self {
            case .image(_, let h): h
            case .video(let v): v.height
            }
        }
        var frameCount: Int {
            switch self {
            case .image: 1
            case .video(let v): v.frameCount
            }
        }
    }

    /// Resolve a plan, or explain why nothing fits.
    public func plan(
        source: Source,
        target: UpscaleTarget,
        quality: QualityPreset = .balanced,
        seed: UInt64 = 42,
        installedVariants: Set<SeedVR2Variant> = Set(SeedVR2Variant.allCases),
        preferredVariant: SeedVR2Variant? = nil
    ) throws -> SeedVR2Plan {
        let (outW, outH) = Self.outputSize(source: source, target: target)
        let padW = SeedVR2Geometry.alignUp(outW, to: SeedVR2Geometry.alignment)
        let padH = SeedVR2Geometry.alignUp(outH, to: SeedVR2Geometry.alignment)
        let frames = source.frameCount
        let budget = hardware.plannableMemoryBytes()

        var reasons: [SettingReason] = []
        reasons.append(SettingReason(
            "output", "\(outW)×\(outH)", .intent,
            because: Self.describe(target) + " from \(source.width)×\(source.height)"
        ))
        if padW != outW || padH != outH {
            reasons.append(SettingReason(
                "working size", "\(padW)×\(padH)", .autoMachine,
                because: "padded to a multiple of \(SeedVR2Geometry.alignment) for the VAE stride and patch size; cropped back on output"
            ))
        }
        reasons.append(SettingReason(
            "memory budget", String(format: "%.1f GB", Double(budget) / 1_073_741_824), .autoMachine,
            because: String(
                format: "%.1f GB available now, less a 15%% margin, capped by the %.1f GB single-model ceiling",
                hardware.availableMemoryGB, hardware.usableMemoryGB
            )
        ))

        let variants = Self.variantLadder(installed: installedVariants, preferred: preferredVariant)
        guard !variants.isEmpty else {
            throw StageError.modelNotInstalled(id: "SeedVR2")
        }

        // A single image is exactly one chunk of one frame; never chunk further. A clip
        // shorter than the smallest ladder entry still gets one padded 5-frame chunk.
        let ladder = frames == 1 ? [1] : Self.chunkLadder.filter { $0 > 1 && $0 <= max(frames, 5) }

        struct Candidate {
            let plan: SeedVR2Plan
            let variantRank: Int
            let work: Double
            let longestChunk: Int
        }

        var candidates: [Candidate] = []
        var smallestConsidered: Int64 = .max

        for (variantRank, variant) in variants.enumerated() {
            for chunkLength in ladder {
                let overlap = frames == 1 ? 0 : min(quality.temporalOverlap, max(chunkLength - 1, 0))
                let chunks = Self.chunks(frameCount: frames, chunkLength: chunkLength, overlap: overlap)
                guard let longest = chunks.map(\.length).max() else { continue }

                for encodeTiling in Self.tilingLadder {
                    for decodeTiling in Self.tilingLadder {
                        let phases = Self.estimatePhases(
                            variant: variant,
                            paddedWidth: padW, paddedHeight: padH,
                            chunks: chunks,
                            encodeTiling: encodeTiling, decodeTiling: decodeTiling,
                            calibration: calibration, engineID: engineID,
                            chipName: hardware.chipName
                        )
                        let peak = phases.map(\.peakBytes).max() ?? 0
                        smallestConsidered = min(smallestConsidered, peak)
                        guard peak <= budget else { continue }

                        var planReasons = reasons
                        planReasons.append(SettingReason(
                            "model", variant.rawValue, preferredVariant == nil ? .autoMachine : .intent,
                            because: preferredVariant == nil
                                ? "best installed checkpoint whose phases fit the budget"
                                : "chosen explicitly"
                        ))
                        if frames > 1 {
                            planReasons.append(SettingReason(
                                "temporal chunk", "\(longest) frames", .autoMachine,
                                because: "longest 4n+1 chunk that fits at this tiling — longer chunks mean fewer seams and better temporal coherence (\(chunks.count) chunk\(chunks.count == 1 ? "" : "s"))"
                            ))
                            planReasons.append(SettingReason(
                                "temporal overlap", "\(overlap) frame\(overlap == 1 ? "" : "s")", .autoMachine,
                                because: "from the \(quality.rawValue) quality preset"
                            ))
                        }
                        planReasons.append(Self.tilingReason(
                            "VAE encode", encodeTiling, width: padW, height: padH,
                            phases: phases, phase: "vae-encode"
                        ))
                        planReasons.append(Self.tilingReason(
                            "VAE decode", decodeTiling, width: padW, height: padH,
                            phases: phases, phase: "vae-decode"
                        ))

                        let plan = SeedVR2Plan(
                            variant: variant,
                            outputWidth: outW, outputHeight: outH,
                            paddedWidth: padW, paddedHeight: padH,
                            encodeTiling: encodeTiling, decodeTiling: decodeTiling,
                            chunks: chunks,
                            temporalOverlap: overlap,
                            seed: seed,
                            phases: phases,
                            reasons: planReasons
                        )
                        // Rank on predicted *seconds* once this machine has measurements,
                        // which weighs the phases against each other properly (decode costs
                        // far more per pixel-frame than encode). Before any measurement,
                        // fall back to raw VAE work units.
                        let predictedSeconds = phases.compactMap { $0.time.seconds }
                        let work: Double = predictedSeconds.count == phases.count
                            ? predictedSeconds.reduce(0, +)
                            : phases.filter { $0.phase != "dit" }.map(\.workUnits).reduce(0, +)
                        candidates.append(Candidate(
                            plan: plan, variantRank: variantRank, work: work, longestChunk: longest
                        ))
                    }
                }
            }
        }

        guard !candidates.isEmpty else {
            throw StageError.insufficientMemory(
                requiredGB: Double(smallestConsidered) / 1_073_741_824,
                availableGB: Double(budget) / 1_073_741_824
            )
        }

        // Rank feasible plans rather than taking the first that fits.
        //
        // The order matters, and the owner's ComfyUI run shows why: tiling multiplies the
        // pixel work of the two VAE phases, which were **97% of that run's time**, while
        // temporal chunk length barely changes total work (it changes seam count and
        // coherence). So the least-tiled plan wins first, and only then the longest chunk.
        // Preferring a long chunk over an untiled pass is how you end up tiling a 1344×768
        // frame into 8 overlapping 512 px tiles on a Mac with 17 GB free.
        let best = candidates.min { a, b in
            if a.variantRank != b.variantRank { return a.variantRank < b.variantRank }
            if a.work != b.work { return a.work < b.work }
            if a.longestChunk != b.longestChunk { return a.longestChunk > b.longestChunk }
            return a.plan.peakBytes < b.plan.peakBytes
        }
        guard let best else {
            throw StageError.insufficientMemory(
                requiredGB: Double(smallestConsidered) / 1_073_741_824,
                availableGB: Double(budget) / 1_073_741_824
            )
        }
        return best.plan
    }

    /// The next plan down when a run hits memory pressure: same intent, one notch cheaper.
    /// Returns nil when already at the bottom of the ladder.
    public func stepDown(from plan: SeedVR2Plan, source: Source, target: UpscaleTarget,
                         quality: QualityPreset) -> SeedVR2Plan? {
        let reduced = HardwareProfile(
            chipName: hardware.chipName, generation: hardware.generation, tier: hardware.tier,
            gpuCoreCount: hardware.gpuCoreCount, memoryBandwidthGBps: hardware.memoryBandwidthGBps,
            totalMemoryBytes: hardware.totalMemoryBytes,
            // Plan against two thirds of what the failing plan actually needed.
            availableMemoryBytes: Int64(Double(plan.peakBytes) * 0.66 / 0.85),
            gpuWiredLimitBytes: hardware.gpuWiredLimitBytes,
            macOSVersion: hardware.macOSVersion
        )
        let resolver = SeedVR2Resolver(hardware: reduced, calibration: calibration, engineID: engineID)
        guard let next = try? resolver.plan(
            source: source, target: target, quality: quality, seed: plan.seed,
            installedVariants: Set(SeedVR2Variant.allCases), preferredVariant: nil
        ) else { return nil }
        return next.peakBytes < plan.peakBytes ? next : nil
    }

    // MARK: - Output size

    static func outputSize(source: Source, target: UpscaleTarget) -> (Int, Int) {
        let w = source.width, h = source.height
        switch target {
        case .scale(let factor):
            return (w * max(factor, 1), h * max(factor, 1))
        case .shortSide(let short):
            guard w > 0, h > 0 else { return (short, short) }
            if w <= h {
                return (short, Int((Double(short) * Double(h) / Double(w)).rounded()))
            }
            return (Int((Double(short) * Double(w) / Double(h)).rounded()), short)
        case .exact(let ew, let eh):
            return (max(ew, 1), max(eh, 1))
        }
    }

    static func describe(_ target: UpscaleTarget) -> String {
        switch target {
        case .scale(let f): "\(f)× upscale"
        case .shortSide(let s): "short side \(s) px"
        case .exact(let w, let h): "exact \(w)×\(h)"
        }
    }

    static func variantLadder(installed: Set<SeedVR2Variant>, preferred: SeedVR2Variant?) -> [SeedVR2Variant] {
        if let preferred { return installed.contains(preferred) ? [preferred] : [] }
        return SeedVR2Variant.byQualityDescending.filter(installed.contains)
    }

    static func tilingReason(
        _ label: String, _ tiling: VAETiling?, width: Int, height: Int,
        phases: [PhaseEstimate], phase: String
    ) -> SettingReason {
        let peak = phases.first { $0.phase == phase }?.peakGB ?? 0
        guard let tiling else {
            return SettingReason(
                label, "untiled", .autoMachine,
                because: String(format: "whole frames in one pass — predicted %.1f GB, which fits", peak)
            )
        }
        let count = tiling.tileCount(width: width, height: height)
        return SettingReason(
            label, "\(tiling.tileSize) px tiles, \(tiling.overlap) px overlap", .autoMachine,
            because: String(
                format: "%d tiles per frame — untiled would not fit; predicted %.1f GB", count, peak
            )
        )
    }

    // MARK: - Estimates

    /// Per-phase peak memory and time.
    ///
    /// **Peak scales with the largest live piece of work, not with the size of the job.**
    /// One chunk of 5 frames peaks the same whether the clip is 5 frames or 243, so the
    /// model multiplies a per-unit byte cost by *peak units* (a chunk's tile area × its
    /// frames for the VAE, a chunk's tokens for the transformer) and adds the phase's
    /// weights. Time, by contrast, scales with the whole job, so it uses *work units*.
    ///
    /// The per-unit costs come from `CalibrationStore` when this engine has run on this
    /// chip, and otherwise from `fallbackBytesPerPeakUnit` — measured on an M2 Max rather
    /// than derived from tensor shapes, because the derivation was badly wrong: the decoder
    /// holds several full-resolution tensors at once, casts GroupNorm to fp32, and expands
    /// channels inside `Upsample3D` before reshaping. A shape-based estimate of 8.5 GB
    /// measured 37.9 GB.
    static func estimatePhases(
        variant: SeedVR2Variant,
        paddedWidth: Int, paddedHeight: Int,
        chunks: [FrameChunk],
        encodeTiling: VAETiling?, decodeTiling: VAETiling?,
        calibration: CalibrationStore, engineID: String, chipName: String
    ) -> [PhaseEstimate] {
        let chunkLength = chunks.map(\.length).max() ?? 1
        let chunkCount = chunks.count
        // ── Peak units ──────────────────────────────────────────────────────────────────
        func tileArea(_ tiling: VAETiling?) -> Double {
            if let tiling { return Double(tiling.tileSize * tiling.tileSize) }
            return Double(paddedWidth * paddedHeight)
        }
        let encodePeakUnits = tileArea(encodeTiling) * Double(chunkLength)
        let decodePeakUnits = tileArea(decodeTiling) * Double(chunkLength)

        let latentW = paddedWidth / SeedVR2Geometry.spatialStride
        let latentH = paddedHeight / SeedVR2Geometry.spatialStride
        let tokensPerChunk = Double(SeedVR2Geometry.latentFrames(chunkLength))
            * Double(latentH / SeedVR2Geometry.spatialPatch)
            * Double(latentW / SeedVR2Geometry.spatialPatch)

        // ── Work units (time) ───────────────────────────────────────────────────────────
        //
        // Counted over the real chunk list, not over the clip, because three effects make
        // those differ and all three matter to the ranking:
        //
        //  1. **Overlap** — frames shared between chunks are encoded and decoded twice.
        //  2. **Tail padding** — a final short chunk is padded up to 4n+1 and the padding
        //     is processed.
        //  3. **Causal warm-up** — the decoder emits `4 × latent` frames and the first
        //     three are discarded. A 5-frame chunk decodes 8 to keep 5 (37% waste); a
        //     21-frame chunk decodes 24 to keep 21 (12%). Ignoring this made the resolver
        //     prefer short chunks and measurably lose time doing it.
        func pixelsPerFrame(_ tiling: VAETiling?) -> Double {
            guard let tiling else { return Double(paddedWidth * paddedHeight) }
            return Double(tiling.processedPixels(width: paddedWidth, height: paddedHeight))
        }
        let encodedFrames = chunks.reduce(0.0) { $0 + Double($1.length) }
        let decodedFrames = chunks.reduce(0.0) {
            $0 + Double(SeedVR2Geometry.decodedFrames(
                latentFrames: SeedVR2Geometry.latentFrames($1.length)
            ))
        }
        let totalTokens = chunks.reduce(0.0) {
            $0 + Double(SeedVR2Geometry.latentFrames($1.length))
                * Double(latentH / SeedVR2Geometry.spatialPatch)
                * Double(latentW / SeedVR2Geometry.spatialPatch)
        }

        func phase(
            _ name: String, peakUnits: Double, workUnits: Double, weights: Int64
        ) -> PhaseEstimate {
            let measured = calibration.bytesPerPeakUnit(
                engineID: engineID, phase: name, chipName: chipName
            )
            let perUnit = measured ?? fallbackBytesPerPeakUnit(name)
            let peak = weights + Int64(perUnit * peakUnits)
            return PhaseEstimate(
                phase: name,
                peakBytes: peak,
                workUnits: workUnits,
                peakUnits: peakUnits,
                weightBytes: weights,
                time: calibration.estimate(
                    engineID: engineID, phase: name, chipName: chipName, workUnits: workUnits
                ),
                peakIsMeasured: measured != nil
            )
        }

        return [
            phase("vae-encode", peakUnits: encodePeakUnits,
                  workUnits: encodedFrames * pixelsPerFrame(encodeTiling),
                  weights: SeedVR2Geometry.vaeBytes),
            phase("dit", peakUnits: tokensPerChunk, workUnits: totalTokens,
                  weights: variant.transformerBytes),
            phase("vae-decode", peakUnits: decodePeakUnits,
                  workUnits: decodedFrames * pixelsPerFrame(decodeTiling),
                  weights: SeedVR2Geometry.vaeBytes),
        ]
    }

    /// Activation bytes per peak unit when this machine has no measurement yet.
    ///
    /// Measured on an M2 Max (32 GB) with the int8 3B checkpoint — see
    /// `HugMac/Design/HugMac-plan.md` §6.4. Replaced per-machine by `CalibrationStore` as
    /// soon as a real run completes.
    static func fallbackBytesPerPeakUnit(_ phase: String) -> Double {
        switch phase {
        // Measured, int8 3B, 1344×768, 9-frame chunks, M2 Max:
        //   encode untiled  3,704 B per pixel-frame  (peaked 32.5 GB — it swapped)
        //   decode 384 px   8,609 B per pixel-frame  (peaked 11.1 GB)
        //   transformer   862,977 B per latent token (peaked 13.7 GB, of which 4.2 weights)
        // A 10% margin on top, because under-predicting costs a swapping job while
        // over-predicting costs a little tiling.
        case "vae-encode":   4_100      // bytes per pixel-frame
        case "vae-decode":   9_500      // bytes per pixel-frame
        case "dit":        950_000      // bytes per latent token
        default:             1_000
        }
    }

    // MARK: - Chunking

    /// Split `frameCount` frames into 4n+1 chunks sharing `overlap` frames with their
    /// predecessor.
    ///
    /// Chunk ranges deliberately *overlap*: the first `blendIn` frames of a chunk were also
    /// produced by the previous chunk, and the assembler cross-fades them so a seam doesn't
    /// show. Every source frame therefore appears in exactly one output frame, but is
    /// computed twice inside the overlap.
    public static func chunks(frameCount: Int, chunkLength: Int, overlap: Int) -> [FrameChunk] {
        let frames = max(frameCount, 1)
        guard frames > 1 else {
            return [FrameChunk(start: 0, length: 1, padTail: 0, blendIn: 0)]
        }
        let length = max(SeedVR2Geometry.alignFramesUp(min(chunkLength, frames)), 5)
        let clampedOverlap = max(0, min(overlap, length - 1))
        let step = max(length - clampedOverlap, 1)

        var result: [FrameChunk] = []
        var start = 0
        while start < frames {
            let available = min(length, frames - start)
            let aligned = SeedVR2Geometry.alignFramesUp(available)
            let chunk = FrameChunk(
                start: start,
                length: aligned,
                padTail: aligned - available,
                blendIn: result.isEmpty ? 0 : clampedOverlap
            )
            result.append(chunk)
            if start + available >= frames { break }
            start += step
        }
        return result
    }
}
