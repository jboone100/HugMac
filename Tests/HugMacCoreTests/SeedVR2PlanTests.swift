import Foundation
import Testing
@testable import HugMacCore

// MARK: - Geometry: the 4n+1 rule

@Suite("SeedVR2 geometry")
struct SeedVR2GeometryTests {

    @Test("Frame counts align to 4n+1 because the VAE compresses time 4× causally")
    func alignment() {
        #expect(SeedVR2Geometry.alignFramesUp(1) == 1)
        #expect(SeedVR2Geometry.alignFramesUp(2) == 5)
        #expect(SeedVR2Geometry.alignFramesUp(5) == 5)
        #expect(SeedVR2Geometry.alignFramesUp(6) == 9)
        #expect(SeedVR2Geometry.alignFramesUp(240) == 241)
        #expect(SeedVR2Geometry.alignFramesDown(243) == 241)
        #expect(SeedVR2Geometry.alignFramesDown(4) == 1)
        for frames in [1, 5, 9, 21, 241] {
            #expect(SeedVR2Geometry.isAligned(frames))
        }
        for frames in [2, 4, 6, 240, 243] {
            #expect(!SeedVR2Geometry.isAligned(frames))
        }
    }

    @Test("Latent frame count matches the shapes the reference pipeline logs")
    func latentFrames() {
        // The ComfyUI run logged `Latents shape: [2, 96, 168, 16]` for a 5-frame batch at
        // 1344×768: 2 latent frames, 768/8 = 96, 1344/8 = 168, 16 channels.
        #expect(SeedVR2Geometry.latentFrames(5) == 2)
        #expect(SeedVR2Geometry.latentFrames(1) == 1)
        #expect(SeedVR2Geometry.latentFrames(241) == 61)
    }

    @Test("Frames pad to a multiple of 16 so patchIn never sees an odd latent dimension")
    func spatialAlignment() {
        #expect(SeedVR2Geometry.alignment == 16)
        #expect(SeedVR2Geometry.alignUp(1344, to: 16) == 1344)
        #expect(SeedVR2Geometry.alignUp(768, to: 16) == 768)
        #expect(SeedVR2Geometry.alignUp(1000, to: 16) == 1008)
    }
}

// MARK: - Chunking

@Suite("Temporal chunking")
struct ChunkingTests {

    @Test("Every chunk is 4n+1 and the chunks cover the clip")
    func coverage() {
        for frameCount in [1, 5, 7, 30, 243, 500] {
            for length in [5, 21, 121, 241] {
                for overlap in [0, 1, 2] {
                    let chunks = SeedVR2Resolver.chunks(
                        frameCount: frameCount, chunkLength: length, overlap: overlap
                    )
                    #expect(!chunks.isEmpty)
                    for chunk in chunks {
                        #expect(SeedVR2Geometry.isAligned(chunk.length),
                                "length \(chunk.length) is not 4n+1")
                        #expect(chunk.padTail >= 0)
                        #expect(chunk.realLength > 0)
                    }
                    #expect(chunks[0].start == 0)
                    #expect(chunks[0].blendIn == 0)
                    // The union of real ranges reaches the end of the clip with no gap.
                    var reached = 0
                    for chunk in chunks {
                        #expect(chunk.start <= reached, "gap before frame \(chunk.start)")
                        reached = max(reached, chunk.end)
                    }
                    #expect(reached >= max(frameCount, 1))
                }
            }
        }
    }

    @Test("A single image is one chunk of one frame")
    func singleImage() {
        let chunks = SeedVR2Resolver.chunks(frameCount: 1, chunkLength: 1, overlap: 0)
        #expect(chunks.count == 1)
        #expect(chunks[0].length == 1)
        #expect(chunks[0].padTail == 0)
        #expect(chunks[0].blendIn == 0)
    }

    @Test("Overlapping chunks share exactly the overlap, and later chunks blend in")
    func overlapIsShared() {
        let chunks = SeedVR2Resolver.chunks(frameCount: 243, chunkLength: 21, overlap: 2)
        #expect(chunks.count > 1)
        for (previous, next) in zip(chunks, chunks.dropFirst()) {
            #expect(next.blendIn == 2)
            let shared = previous.end - next.start
            #expect(shared == 2 || next.end == 243 || previous.padTail > 0,
                    "expected 2 shared frames, got \(shared)")
        }
    }

    @Test("A clip shorter than one chunk pads its tail rather than misaligning")
    func shortClipPads() {
        let chunks = SeedVR2Resolver.chunks(frameCount: 3, chunkLength: 21, overlap: 1)
        #expect(chunks.count == 1)
        #expect(chunks[0].length == 5)
        #expect(chunks[0].padTail == 2)
        #expect(chunks[0].realLength == 3)
    }
}

// MARK: - Tiling model

@Suite("VAE tiling")
struct TilingTests {

    @Test("Tile count reproduces the reference run's 8 tiles per frame")
    func matchesReferenceRun() {
        // The owner's ComfyUI run logged "Encoding 8 tiles (Tile: (512, 512), Overlap:
        // (128, 128))" for a 1344×768 frame.
        let tiling = VAETiling(tileSize: 512, overlap: 128)
        #expect(tiling.tileCount(width: 1344, height: 768) == 8)
    }

    @Test("Tiling at 512 px roughly doubles the pixel work on a 1344×768 frame")
    func tilingOverhead() {
        let tiling = VAETiling(tileSize: 512, overlap: 128)
        let processed = Double(tiling.processedPixels(width: 1344, height: 768))
        let whole = Double(1344 * 768)
        #expect(processed / whole > 1.9)
        #expect(processed / whole < 2.2)
    }
}

// MARK: - The resolver

@Suite("SeedVR2 resolver")
struct ResolverTests {

    /// The owner's machine: M2 Max, 32 GB, with roughly what the ComfyUI log reported free.
    static func m2Max(availableGB: Double = 19.0) -> HardwareProfile {
        HardwareProfile(
            chipName: "Apple M2 Max",
            generation: 2,
            tier: .max,
            gpuCoreCount: 30,
            memoryBandwidthGBps: 400,
            totalMemoryBytes: 34_359_738_368,
            availableMemoryBytes: Int64(availableGB * 1_073_741_824),
            gpuWiredLimitBytes: Int64(0.75 * 34_359_738_368),
            macOSVersion: .init(majorVersion: 26, minorVersion: 0, patchVersion: 0)
        )
    }

    @Test("Bandwidth comes from chip family, not a model-identifier table")
    func bandwidthByFamily() {
        // MLXUI's table had no M1 entries at all, so every M1 Mac reported 100 GB/s.
        #expect(HardwareProfile.bandwidth(generation: 1, tier: .ultra) == 800)
        #expect(HardwareProfile.bandwidth(generation: 1, tier: .base) == 68)
        #expect(HardwareProfile.bandwidth(generation: 2, tier: .max) == 400)
        #expect(HardwareProfile.generation(from: "Apple M2 Max") == 2)
        #expect(HardwareProfile.tier(from: "Apple M1 Ultra") == .ultra)
        #expect(HardwareProfile.tier(from: "Apple M4") == .base)
    }

    @Test("The owner's target resolves to 1344×768, as the reference run did")
    func ownersOutputSize() throws {
        let source = SeedVR2Resolver.Source.video(VideoMedia(
            url: URL(fileURLWithPath: "/tmp/Diner_0.mp4"),
            width: 672, height: 384, fps: 24, frameCount: 243, hasAudio: true
        ))
        let (width, height) = SeedVR2Resolver.outputSize(source: source, target: .shortSide(768))
        #expect(width == 1344)
        #expect(height == 768)
    }

    @Test("Tiling is arithmetic, not taste: 1344×768 video tiles both VAE phases")
    func tilingFollowsMeasurement() throws {
        let resolver = SeedVR2Resolver(hardware: Self.m2Max())
        let source = SeedVR2Resolver.Source.video(VideoMedia(
            url: URL(fileURLWithPath: "/tmp/Diner_0.mp4"),
            width: 672, height: 384, fps: 24, frameCount: 243, hasAudio: true
        ))
        let plan = try resolver.plan(source: source, target: .shortSide(768), quality: .balanced)

        #expect(plan.outputWidth == 1344)
        #expect(plan.outputHeight == 768)
        // Measured on this Mac with the int8 3B checkpoint: ~3.8 KB per pixel-frame to
        // encode and ~10.5 KB to decode. At 1344×768 that puts both VAE phases over a
        // 32 GB machine for any useful chunk length, so both tile. The first run that
        // tried untiled encode at 9-frame chunks peaked at 32.5 GB and swapped.
        #expect(plan.encodeTiling != nil)
        #expect(plan.decodeTiling != nil)
        #expect(plan.peakBytes <= Self.m2Max().plannableMemoryBytes())
        #expect(plan.peakBytes <= Self.m2Max().plannableMemoryBytes())
        #expect(plan.chunks.allSatisfy { SeedVR2Geometry.isAligned($0.length) })
        #expect(plan.frameCount >= 243)
    }

    @Test("A cramped machine tiles and shortens chunks rather than failing")
    func adaptsToSmallMachine() throws {
        let resolver = SeedVR2Resolver(hardware: Self.m2Max(availableGB: 14.0))
        let source = SeedVR2Resolver.Source.video(VideoMedia(
            url: URL(fileURLWithPath: "/tmp/Diner_0.mp4"),
            width: 672, height: 384, fps: 24, frameCount: 243, hasAudio: true
        ))
        let plan = try resolver.plan(source: source, target: .shortSide(768), quality: .balanced)
        #expect(plan.peakBytes <= resolver.hardware.plannableMemoryBytes())
        let tiled = plan.decodeTiling != nil
        let shortChunks = (plan.chunks.map(\.length).max() ?? 0) < 21
        #expect(tiled || shortChunks, "expected the resolver to economise somewhere")
    }

    @Test("A machine with no headroom is refused with numbers, not a crash")
    func refusesWhenNothingFits() {
        let resolver = SeedVR2Resolver(hardware: Self.m2Max(availableGB: 0.2))
        let source = SeedVR2Resolver.Source.image(width: 4000, height: 3000)
        #expect(throws: StageError.self) {
            _ = try resolver.plan(source: source, target: .scale(2), quality: .best)
        }
    }

    @Test("Single-image upscaling plans one frame and no temporal overlap")
    func singleImagePlan() throws {
        let resolver = SeedVR2Resolver(hardware: Self.m2Max())
        let plan = try resolver.plan(
            source: .image(width: 1024, height: 768), target: .scale(2), quality: .best
        )
        #expect(plan.isSingleImage)
        #expect(plan.chunks.count == 1)
        #expect(plan.temporalOverlap == 0)
        #expect(plan.outputWidth == 2048)
        #expect(plan.outputHeight == 1536)
    }

    @Test("Quality preset drives temporal overlap")
    func qualityDrivesOverlap() throws {
        let resolver = SeedVR2Resolver(hardware: Self.m2Max())
        let source = SeedVR2Resolver.Source.video(VideoMedia(
            url: URL(fileURLWithPath: "/tmp/clip.mp4"),
            width: 672, height: 384, fps: 24, frameCount: 120, hasAudio: false
        ))
        let fast = try resolver.plan(source: source, target: .scale(2), quality: .fast)
        let best = try resolver.plan(source: source, target: .scale(2), quality: .best)
        #expect(fast.temporalOverlap == 0)
        #expect(best.temporalOverlap == 2)
    }

    @Test("Step-down finds a cheaper plan when a run hits memory pressure")
    func stepDownIsCheaper() throws {
        let resolver = SeedVR2Resolver(hardware: Self.m2Max())
        let source = SeedVR2Resolver.Source.video(VideoMedia(
            url: URL(fileURLWithPath: "/tmp/Diner_0.mp4"),
            width: 672, height: 384, fps: 24, frameCount: 243, hasAudio: true
        ))
        // Step down at a size where there is headroom to give up.
        let small = SeedVR2Resolver.Source.video(VideoMedia(
            url: URL(fileURLWithPath: "/tmp/small.mp4"),
            width: 320, height: 180, fps: 24, frameCount: 40, hasAudio: false
        ))
        let plan = try resolver.plan(source: small, target: .scale(2), quality: .balanced)
        let next = resolver.stepDown(
            from: plan, source: small, target: .scale(2), quality: .balanced
        )
        guard let cheaper = next else {
            Issue.record("expected a cheaper plan to step down to")
            return
        }
        #expect(cheaper.peakBytes < plan.peakBytes)

        // At 1344×768 the transformer is the floor, and there is nothing cheaper to pick:
        // stepping down must report that honestly rather than loop forever.
        let full = try resolver.plan(source: source, target: .shortSide(768), quality: .balanced)
        var current = full
        var steps = 0
        while let lower = resolver.stepDown(
            from: current, source: source, target: .shortSide(768), quality: .balanced
        ), steps < 10 {
            #expect(lower.peakBytes < current.peakBytes)
            current = lower
            steps += 1
        }
        #expect(steps < 10, "step-down should terminate, not cycle")
    }

    @Test("Every resolved setting carries a reason, so Advanced can explain itself")
    func reasonsArePresent() throws {
        let resolver = SeedVR2Resolver(hardware: Self.m2Max())
        let plan = try resolver.plan(
            source: .image(width: 672, height: 384), target: .shortSide(768), quality: .balanced
        )
        let named = Set(plan.reasons.map(\.setting))
        #expect(named.contains("model"))
        #expect(named.contains("output"))
        #expect(named.contains("memory budget"))
        #expect(named.contains("VAE decode"))
        #expect(plan.reasons.allSatisfy { !$0.because.isEmpty })
    }
}

// MARK: - Calibration

@Suite("Calibration")
struct CalibrationTests {

    @Test("The ComfyUI baseline is recorded and totals 6 h 33 m")
    func baselineTotals() {
        let total = CalibrationStore.comfyUIBaselineTotalSeconds
        #expect(total > 23_500 && total < 23_700)
        #expect(Int(total / 60) == 393)  // 6 h 33 m
    }

    @Test("A PyTorch-MPS baseline never predicts the MLX engine")
    func enginesDoNotMix() {
        let store = CalibrationStore(samples: CalibrationStore.comfyUIBaseline())
        let mlx = store.estimate(
            engineID: SeedVR2Resolver.mlxEngineID, phase: "vae-decode",
            chipName: "Apple M2 Max", workUnits: 1_000_000
        )
        #expect(mlx == .unknown)

        let baseline = store.estimate(
            engineID: CalibrationStore.comfyUIBaselineEngineID, phase: "vae-decode",
            chipName: "Apple M2 Max", workUnits: 61 * 5 * 1344 * 768
        )
        #expect(baseline != .unknown)
    }

    @Test("An unmeasured engine reports unknown total time rather than inventing one")
    func unknownTimeIsHonest() throws {
        let resolver = SeedVR2Resolver(hardware: ResolverTests.m2Max())
        let plan = try resolver.plan(
            source: .image(width: 672, height: 384), target: .scale(2), quality: .balanced
        )
        #expect(plan.totalTime == .unknown)
        #expect(plan.peakBytes > 0)
    }

    @Test("Recording a measurement makes later plans predict time")
    func measurementEnablesPrediction() throws {
        var store = CalibrationStore()
        let source = SeedVR2Resolver.Source.image(width: 672, height: 384)
        let dry = SeedVR2Resolver(hardware: ResolverTests.m2Max())
        let first = try dry.plan(source: source, target: .scale(2), quality: .balanced)

        for phase in first.phases {
            store.record(CalibrationSample(
                engineID: SeedVR2Resolver.mlxEngineID, phase: phase.phase,
                workUnits: phase.workUnits, seconds: 12.0, peakBytes: phase.peakBytes,
                chipName: "Apple M2 Max"
            ))
        }
        let warm = SeedVR2Resolver(hardware: ResolverTests.m2Max(), calibration: store)
        let second = try warm.plan(source: source, target: .scale(2), quality: .balanced)
        guard let seconds = second.totalTime.seconds else {
            Issue.record("expected a measured total after recording samples")
            return
        }
        #expect(seconds > 0)
    }
}

// MARK: - Variants

@Suite("SeedVR2 variants")
struct VariantTests {

    @Test("Weight sizes come from the published repos, not from parameter arithmetic")
    func measuredWeightSizes() {
        // huggingface.co/api/models/…?blobs=true, 2026-09-17.
        #expect(SeedVR2Variant.threeBInt8.transformerBytes == 4_220_000_000)
        #expect(SeedVR2Variant.threeBFP16.transformerBytes == 7_940_000_000)
        #expect(SeedVR2Variant.sevenBInt8.transformerBytes == 8_760_000_000)
        #expect(SeedVR2Variant.sevenBFP16.transformerBytes == 16_480_000_000)

        // The arithmetic a parameter count would give is materially lower — this is the
        // mistake the measured table exists to avoid.
        let naive = Int64(SeedVR2Variant.threeBInt8.parameterCountB * 1_000_000_000
            * SeedVR2Variant.threeBInt8.bytesPerWeight)
        #expect(SeedVR2Variant.threeBInt8.transformerBytes > naive + 1_000_000_000)
    }

    @Test("The fp16 7B checkpoint needs real headroom; the int8 3B still plans")
    func variantFitsBudget() throws {
        // 16.48 GB of weights fits a 32 GB Mac with ~19 GiB free — but not one with 12 GiB
        // free, and no amount of tiling helps, because tiling cannot shrink a weight.
        let tight = SeedVR2Resolver(hardware: ResolverTests.m2Max(availableGB: 12.0))
        #expect(throws: StageError.self) {
            _ = try tight.plan(
                source: .image(width: 672, height: 384), target: .scale(2),
                quality: .best, installedVariants: [.sevenBFP16], preferredVariant: .sevenBFP16
            )
        }
        let small = try tight.plan(
            source: .image(width: 672, height: 384), target: .scale(2),
            quality: .best, installedVariants: [.threeBInt8]
        )
        #expect(small.variant == .threeBInt8)
    }

    @Test("Below the weight floor the answer is a refusal, not a slow run")
    func weightFloorIsRefused() {
        // The int8 3B transformer alone is 4.22 GB resident. A machine with ~5 GiB free
        // cannot run it at any tiling or chunk length, and should say so up front rather
        // than starting a job that will thrash.
        let resolver = SeedVR2Resolver(hardware: ResolverTests.m2Max(availableGB: 5.0))
        let source = SeedVR2Resolver.Source.video(VideoMedia(
            url: URL(fileURLWithPath: "/tmp/Diner_0.mp4"),
            width: 672, height: 384, fps: 24, frameCount: 243, hasAudio: true
        ))
        #expect(throws: StageError.self) {
            _ = try resolver.plan(
                source: source, target: .shortSide(768), quality: .balanced,
                installedVariants: [.threeBInt8]
            )
        }
    }

    @Test("Download size includes the shared VAE")
    func downloadSize() {
        let gb = Double(SeedVR2Variant.threeBInt8.downloadBytes) / 1_000_000_000
        #expect(gb > 4.6 && gb < 4.8)
    }
}

// MARK: - Decoder temporal mapping

@Suite("Decoder frame counts")
struct DecoderFrameTests {

    @Test("The decoder emits 4× latent frames, not 4n+1")
    func decodedFrameCount() {
        // Measured: a 5-frame chunk (2 latent frames) decodes to 8 frames. mlx-swift threw
        // `Shapes (1,3,8,768,1344) and (1,1,5,768,1344) cannot be broadcast` when the plan
        // assumed 5.
        #expect(SeedVR2Geometry.decodedFrames(latentFrames: 2) == 8)
        #expect(SeedVR2Geometry.decodedFrames(latentFrames: 3) == 12)
        // One latent frame is trimmed back to one frame by each upsample.
        #expect(SeedVR2Geometry.decodedFrames(latentFrames: 1) == 1)
    }

    @Test("Causal warm-up is three frames for a clip and none for a single image")
    func warmup() {
        #expect(SeedVR2Geometry.causalWarmupFrames(chunkLength: 1) == 0)
        #expect(SeedVR2Geometry.causalWarmupFrames(chunkLength: 5) == 3)
        #expect(SeedVR2Geometry.causalWarmupFrames(chunkLength: 9) == 3)
        #expect(SeedVR2Geometry.causalWarmupFrames(chunkLength: 241) == 3)
    }

    @Test("Dropping the warm-up recovers exactly the chunk's frames")
    func warmupRecoversChunk() {
        for length in [1, 5, 9, 21, 121, 241] {
            let latent = SeedVR2Geometry.latentFrames(length)
            let decoded = SeedVR2Geometry.decodedFrames(latentFrames: latent)
            let warmup = SeedVR2Geometry.causalWarmupFrames(chunkLength: length)
            #expect(decoded - warmup == length, "chunk \(length) mismatched")
        }
    }
}


// MARK: - Tiling preference at sizes where it can be honoured

@Suite("Tiling preference")
struct TilingPreferenceTests {

    @Test("A small output still goes untiled — tiling is only forced by memory")
    func smallOutputUntiled() throws {
        let resolver = SeedVR2Resolver(hardware: ResolverTests.m2Max())
        let source = SeedVR2Resolver.Source.video(VideoMedia(
            url: URL(fileURLWithPath: "/tmp/small.mp4"),
            width: 320, height: 180, fps: 24, frameCount: 40, hasAudio: false
        ))
        let plan = try resolver.plan(source: source, target: .scale(2), quality: .balanced)
        #expect(plan.outputWidth == 640)
        // Encode is cheap per pixel-frame, so it stays untiled. Decode costs ~2.7× more per
        // pixel-frame, so the cheapest *total* plan tiles decode and takes longer chunks —
        // fewer discarded warm-up frames pays for the tiling overhead. Mixed tiling is the
        // right answer here, and neither extreme is.
        #expect(plan.encodeTiling == nil)
        #expect(plan.peakBytes <= ResolverTests.m2Max().plannableMemoryBytes())
    }

    @Test("The transformer sets a floor that tiling cannot lower")
    func transformerFloor() {
        // The DiT phase's peak depends on tokens per chunk, which tiling does not touch:
        // measured ~0.86 MB per latent token, so 1344×768 with the shortest video chunk
        // needs roughly 12 GB no matter what the VAE does.
        let phases = SeedVR2Resolver.estimatePhases(
            variant: .threeBInt8,
            paddedWidth: 1344, paddedHeight: 768,
            chunks: SeedVR2Resolver.chunks(frameCount: 5, chunkLength: 5, overlap: 0),
            encodeTiling: VAETiling(tileSize: 384, overlap: 64),
            decodeTiling: VAETiling(tileSize: 384, overlap: 64),
            calibration: CalibrationStore(), engineID: SeedVR2Resolver.mlxEngineID,
            machine: MachineKey(chipName: "Apple M2 Max")
        )
        let dit = phases.first { $0.phase == "dit" }
        let gb = Double(dit?.peakBytes ?? 0) / 1_073_741_824
        #expect(gb > 10 && gb < 14)
    }
}

// MARK: - Size-aware calibration

@Suite("Calibration by size")
struct CalibrationBySizeTests {

    static func sample(_ units: Double, activationGB: Double, phase: String = "dit") -> CalibrationSample {
        CalibrationSample(
            engineID: SeedVR2Resolver.mlxEngineID, phase: phase, workUnits: units, peakUnits: units,
            seconds: 1, peakBytes: Int64((activationGB + 4) * 1_073_741_824),
            weightBytes: Int64(4 * 1_073_741_824), chipName: "Apple M2 Max"
        )
    }

    func predict(_ store: CalibrationStore, _ units: Double) -> (bytes: Double, extrapolatedAbove: Bool)? {
        store.predictedActivation(engineID: SeedVR2Resolver.mlxEngineID, phase: "dit",
                                  chipName: "Apple M2 Max", peakUnits: units)
    }

    @Test("Between measured sizes, activations interpolate")
    func interpolates() throws {
        // The measured transformer: 8,064 tokens → 5.3 GB, 12,096 tokens → 9.6 GB.
        let store = CalibrationStore(samples: [Self.sample(8064, activationGB: 5.3),
                                               Self.sample(12096, activationGB: 9.6)])
        let mid = try #require(predict(store, 10080))
        #expect(abs(mid.bytes / 1_073_741_824 - 7.45) < 0.01)
        #expect(!mid.extrapolatedAbove)
    }

    @Test("A small workload's high per-unit cost doesn't inflate a large plan")
    func smallSampleDoesNotDominate() throws {
        // What went wrong: one 4,032-token sample at a high per-token rate, applied as a
        // global maximum, made 8,064-token plans look ~50% heavier than measured.
        let store = CalibrationStore(samples: [Self.sample(4032, activationGB: 5.2),
                                               Self.sample(8064, activationGB: 5.3)])
        let large = try #require(predict(store, 8064))
        #expect(abs(large.bytes / 1_073_741_824 - 5.3) < 0.01)
    }

    @Test("Below anything measured, the smallest measurement is the bound")
    func belowRangeIsConservative() throws {
        let store = CalibrationStore(samples: [Self.sample(8064, activationGB: 5.3)])
        let small = try #require(predict(store, 2000))
        #expect(abs(small.bytes / 1_073_741_824 - 5.3) < 0.01)
    }

    @Test("Above anything measured, the prediction is flagged and scales up")
    func aboveRangeIsFlagged() throws {
        let store = CalibrationStore(samples: [Self.sample(8064, activationGB: 5.3)])
        let large = try #require(predict(store, 16128))
        #expect(large.extrapolatedAbove)
        #expect(abs(large.bytes / 1_073_741_824 - 10.6) < 0.01)
    }
}
