import Foundation
import Testing
@testable import LocalLabCore

private let gib = 1_073_741_824.0

private func mac(_ chip: String, cores: Int, memoryGB: Double, availableGB: Double) -> HardwareProfile {
    let total = Int64(memoryGB * gib)
    return HardwareProfile(
        chipName: chip,
        generation: HardwareProfile.generation(from: chip),
        tier: HardwareProfile.tier(from: chip),
        gpuCoreCount: cores,
        memoryBandwidthGBps: HardwareProfile.bandwidth(
            generation: HardwareProfile.generation(from: chip), tier: HardwareProfile.tier(from: chip)
        ),
        totalMemoryBytes: total,
        availableMemoryBytes: Int64(availableGB * gib),
        gpuWiredLimitBytes: HardwareProfile.wiredLimit(total: total),
        macOSVersion: .init(majorVersion: 26, minorVersion: 6, patchVersion: 2)
    )
}

private let referenceOnly = CalibrationStore(reference: ReferenceMachine.all)
private let flux = ImageModelCatalog.fluxSchnell4bit

@Suite("Create Image")
struct ImageGenerationTests {
    /// The repo's file list as the Hub listed it on 2026-09-18.
    static let repoFiles = [
        ".gitattributes", "README.md", "metadata.json",
        "text_encoder/flux1_mlx_model.safetensors",
        "text_encoder_2/flux1_mlx_model-00001-of-00002.safetensors",
        "text_encoder_2/flux1_mlx_model-00002-of-00002.safetensors",
        "tokenizer/merges.txt", "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer_2/special_tokens_map.json", "tokenizer_2/spiece.model", "tokenizer_2/tokenizer.json",
        "tokenizer_2/tokenizer_config.json",
        "transformer/flux1_mlx_model-00001-of-00004.safetensors",
        "transformer/flux1_mlx_model-00002-of-00004.safetensors",
        "transformer/flux1_mlx_model-00003-of-00004.safetensors",
        "transformer/flux1_mlx_model-00004-of-00004.safetensors",
        "vae/flux1_mlx_model.safetensors",
    ]

    @Test("the manifest downloads the weights and the two tokenizer files the engine reads, nothing else")
    func manifestSelection() {
        let chosen = Set(flux.manifest.select(from: Self.repoFiles))
        #expect(chosen.count == 11)
        #expect(chosen.contains("tokenizer_2/spiece.model"))
        #expect(chosen.contains("tokenizer/vocab.json") && chosen.contains("tokenizer/merges.txt"))
        #expect(!chosen.contains("tokenizer_2/tokenizer.json"))
        #expect(!chosen.contains("README.md"))
        // Every file a required tensor is checked in is one the install downloads.
        for path in flux.manifest.requiredTensors.keys {
            #expect(chosen.contains(path), "\(path) is checked but not downloaded")
        }
    }

    @Test("an install of a known repo uses its engine's manifest")
    func knownManifests() {
        #expect(ComponentManifest.known(for: flux.repo) == flux.manifest)
        #expect(ComponentManifest.known(for: SeedVR2Variant.threeBInt8.hfRepo) == SeedVR2Variant.manifest)
        #expect(ComponentManifest.known(for: "mlx-community/Qwen3.5-9B-MLX-4bit") == nil)
    }

    @Test("Browse runs the curated FLUX build and explains the rest")
    func runner() {
        #expect(RunnerSupport.runner(for: CatalogEntry(repo: flux.repo, task: "text-to-image")) == .createImage(flux))
        let other = RunnerSupport.runner(for: CatalogEntry(repo: "someone/flux-dev-mlx", task: "text-to-image"))
        #expect(!other.isRunnable)
    }

    @Test("every offered size is a multiple of 16 and at most about a megapixel")
    func sizes() {
        for size in ImageSize.allCases {
            #expect(size.width % 16 == 0 && size.height % 16 == 0)
            #expect(size.width * size.height <= 1_050_000)
        }
    }

    @Test("on the reference Mac: 1024² takes about 75 s and peaks at the transformer")
    func referenceMac() throws {
        let m2Max = mac("Apple M2 Max", cores: 30, memoryGB: 32, availableGB: 20)
        let fit = ImageModelFitter(hardware: m2Max, calibration: referenceOnly).fit(flux, size: .square1024)
        #expect(fit.grade == .green)
        let seconds = try #require(fit.time.seconds)
        #expect((65 ... 85).contains(seconds))
        // The measured peak was 7.69–7.81 GB (7.2–7.3 GiB).
        #expect((7.1 ... 7.4).contains(Double(fit.peakBytes) / gib))
        // Smaller images are faster, roughly by token count.
        let small = try #require(ImageModelFitter(hardware: m2Max, calibration: referenceOnly)
            .fit(flux, size: .square512).time.seconds)
        #expect(small < seconds / 3)
    }

    @Test("an 8 GB Mac can't hold the transformer; a busy 16 GB Mac is told to close apps")
    func smallerMacs() {
        let eight = mac("Apple M1", cores: 8, memoryGB: 8, availableGB: 5)
        if case .red = ImageModelFitter(hardware: eight, calibration: referenceOnly).fit(flux).grade {} else {
            Issue.record("an 8 GB Mac should be red")
        }
        let busy = mac("Apple M2", cores: 10, memoryGB: 16, availableGB: 5)
        let fit = ImageModelFitter(hardware: busy, calibration: referenceOnly).fit(flux, size: .square512)
        #expect(!fit.fitsNow)
        #expect(fit.grade != .green)
        if case .red = fit.grade { Issue.record("16 GB should fit once apps are closed") }
    }

    @Test("this Mac's own measurements replace the built-in figures")
    func ownMeasurements() throws {
        let m2Max = mac("Apple M2 Max", cores: 30, memoryGB: 32, availableGB: 20)
        var calibration = referenceOnly
        let machine = m2Max.machineKey
        for (phase, work, peak, bytes, weights) in [
            ("text-encode", 1.0, 1.0, Int64(3.3e9), flux.textEncoderBytes),
            ("transformer", 16384.0, 4096.0, Int64(8.0e9), flux.transformerBytes),
            ("vae-decode", 1048576.0, 1048576.0, Int64(7.0e9), flux.vaeBytes),
        ] {
            calibration.record(CalibrationSample(
                engineID: ImageModelFitter.engineID, phase: phase, workUnits: work, peakUnits: peak,
                seconds: phase == "transformer" ? 40 : 1, peakBytes: bytes, weightBytes: weights,
                machine: machine, chunkFrames: 1, note: "test"
            ))
        }
        let fit = ImageModelFitter(hardware: m2Max, calibration: calibration).fit(flux, size: .square1024)
        #expect(fit.time == .measured(seconds: 42))
        #expect(fit.peakBytes == Int64(8.0e9))
    }

    @Test("a create-image job survives a save and follows a library move")
    func jobPersistence() throws {
        let spec = ImageJobSpec(model: flux.repo, prompt: "a fox", width: 1024, height: 1024, steps: 4,
                                seed: 7, outputURL: URL(fileURLWithPath: "/old/outputs/a-fox.png"))
        let job = Job(title: "a fox", kind: .createImage(spec))
        let decoded = try JSONDecoder().decode(Job.self, from: JSONEncoder().encode(job))
        #expect(decoded == job)
        #expect(job.kind.engineID == "create-image")
        #expect(job.inputURLs.isEmpty)
        let moved = job.remappingURLs { URL(fileURLWithPath: $0.path.replacingOccurrences(of: "/old", with: "/new")) }
        #expect(moved.kind.outputURL.path == "/new/outputs/a-fox.png")
    }

    @Test("durations read naturally")
    func durations() {
        #expect(ImageModelFitter.duration(23) == "25 s")
        #expect(ImageModelFitter.duration(74) == "75 s")
        #expect(ImageModelFitter.duration(150) == "2½ min")
        #expect(ImageModelFitter.duration(245) == "4 min")
    }
}
