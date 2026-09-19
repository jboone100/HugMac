import Foundation
import Testing
@testable import LocalLabCore

private let gib = 1_073_741_824.0

private func mac(memoryGB: Double, availableGB: Double) -> HardwareProfile {
    let total = Int64(memoryGB * gib)
    return HardwareProfile(
        chipName: "Apple M2 Max", generation: 2, tier: .max, gpuCoreCount: 30, memoryBandwidthGBps: 400,
        totalMemoryBytes: total, availableMemoryBytes: Int64(availableGB * gib),
        gpuWiredLimitBytes: HardwareProfile.wiredLimit(total: total),
        macOSVersion: .init(majorVersion: 26, minorVersion: 6, patchVersion: 2)
    )
}

/// Two entries exactly as `GET /api/models?author=mlx-community&search=Qwen3.5-9B-4bit&expand[]=…`
/// returned them on 2026-09-18, minus the chat template.
let catalogFixture = #"""
[
 {
  "id": "mlx-community/Qwen3.5-9B-MLX-4bit",
  "cardData": {
   "base_model": "Qwen/Qwen3.5-9B",
   "library_name": "mlx",
   "pipeline_tag": "image-text-to-text",
   "tags": [
    "mlx",
    "qwen3.5",
    "vision-language-model",
    "quantized",
    "4bit"
   ],
   "license": "apache-2.0"
  },
  "gated": false,
  "lastModified": "2026-03-23T17:58:27.000Z",
  "likes": 173,
  "config": {
   "architectures": [
    "Qwen3_5ForConditionalGeneration"
   ],
   "model_type": "qwen3_5",
   "quantization_config": {
    "bits": 4
   }
  },
  "downloads": 27120,
  "safetensors": {
   "parameters": {
    "U32": 8952741888,
    "BF16": 457071088,
    "F32": 768
   },
   "total": 9409813744
  },
  "tags": [
   "mlx",
   "safetensors",
   "qwen3_5",
   "qwen3.5",
   "vision-language-model",
   "quantized",
   "4bit",
   "image-text-to-text",
   "conversational",
   "base_model:Qwen/Qwen3.5-9B",
   "base_model:quantized:Qwen/Qwen3.5-9B",
   "license:apache-2.0",
   "4-bit",
   "region:us"
  ],
  "pipeline_tag": "image-text-to-text",
  "library_name": "mlx"
 },
 {
  "id": "mlx-community/Qwen3.5-9B-4bit",
  "cardData": {
   "library_name": "transformers",
   "license": "apache-2.0",
   "license_link": "https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE",
   "pipeline_tag": "image-text-to-text",
   "base_model": [
    "Qwen/Qwen3.5-9B-Base"
   ],
   "tags": [
    "mlx"
   ]
  },
  "gated": false,
  "lastModified": "2026-03-02T16:42:26.000Z",
  "likes": 15,
  "config": {
   "architectures": [
    "Qwen3_5ForConditionalGeneration"
   ],
   "model_type": "qwen3_5",
   "quantization_config": {
    "bits": 4
   }
  },
  "downloads": 13845,
  "safetensors": {
   "parameters": {
    "U32": 8952741888,
    "BF16": 457071088,
    "F32": 768
   },
   "total": 9409813744
  },
  "tags": [
   "transformers",
   "safetensors",
   "qwen3_5",
   "image-text-to-text",
   "mlx",
   "conversational",
   "base_model:Qwen/Qwen3.5-9B-Base",
   "base_model:quantized:Qwen/Qwen3.5-9B-Base",
   "license:apache-2.0",
   "endpoints_compatible",
   "4-bit",
   "region:us"
  ],
  "pipeline_tag": "image-text-to-text",
  "library_name": "transformers"
 }
]
"""#

@Suite("Catalog")
struct CatalogTests {
    @Test func parsesTheRealListingShape() throws {
        let entries = try HuggingFaceCatalog.parseList(Data(catalogFixture.utf8))
        #expect(entries.map(\.repo) == ["mlx-community/Qwen3.5-9B-MLX-4bit", "mlx-community/Qwen3.5-9B-4bit"])
        let first = entries[0]
        #expect(first.task == "image-text-to-text")
        #expect(first.modelType == "qwen3_5")
        #expect(first.bits == 4)
        #expect(first.license == "apache-2.0")
        #expect(first.baseModel == "Qwen/Qwen3.5-9B")
        #expect(first.downloads == 27_120)
        #expect(!first.gated)
        #expect(entries[1].baseModel == "Qwen/Qwen3.5-9B-Base", "base_model may be a list")
    }

    @Test func sizeEstimateIncludesQuantizationScales() throws {
        let entries = try HuggingFaceCatalog.parseList(Data(catalogFixture.utf8))
        let nine = try #require(entries.first { $0.repo == "mlx-community/Qwen3.5-9B-4bit" })
        // The repo's .safetensors files total 5.950 GB.
        let estimate = try #require(nine.estimatedWeightBytes)
        #expect(abs(Double(estimate) - 5.950e9) / 5.950e9 < 0.01)
    }

    @Test func searchURLCarriesTheFilters() throws {
        let catalog = HuggingFaceCatalog(token: { nil })
        let url = try #require(catalog.url(for: CatalogQuery(text: "qwen", task: .chat, sort: .likes), pageSize: 50))
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        func value(_ name: String) -> String? { items.first { $0.name == name }?.value }
        #expect(value("filter") == "mlx")
        #expect(value("author") == "mlx-community")
        #expect(value("search") == "qwen")
        #expect(value("pipeline_tag") == "text-generation")
        #expect(value("sort") == "likes")
        #expect(items.filter { $0.name == "expand[]" }.count == HuggingFaceCatalog.expansions.count)
        let everyone = try #require(catalog.url(for: CatalogQuery(publisher: .everyone), pageSize: 50))
        #expect(!everyone.absoluteString.contains("author="))
    }

    @Test func pagesFollowTheLinkHeader() {
        let header = #"<https://huggingface.co/api/models?filter=mlx&cursor=abc123>; rel="next""#
        #expect(HuggingFaceCatalog.nextURL(fromLinkHeader: header)?.absoluteString == "https://huggingface.co/api/models?filter=mlx&cursor=abc123")
        #expect(HuggingFaceCatalog.nextURL(fromLinkHeader: #"<https://x.y/z>; rel="prev""#) == nil)
    }

    @Test func cacheServesTheLastResultsOffline() {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("locallab-catalog-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let cache = CatalogCache(directory: directory)
        let query = CatalogQuery(text: "Qwen 3.5/9B")
        let when = Date(timeIntervalSince1970: 1_800_000_000)
        cache.save([CatalogEntry(repo: "a/b")], for: query, at: when)
        #expect(cache.load(query)?.entries.map(\.repo) == ["a/b"])
        #expect(cache.load(query)?.fetched == when)
        #expect(cache.load(CatalogQuery(text: "other")) == nil)
    }

    @Test func configSummaryReadsVLMAndHybridModels() {
        let config = #"{"model_type":"qwen3_5","quantization":{"bits":4,"group_size":64},"text_config":{"num_hidden_layers":32,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"]}}"#
        let summary = ModelConfigSummary.parse(Data(config.utf8))
        #expect(summary?.modelType == "qwen3_5")
        #expect(summary?.layers == 32)
        #expect(summary?.cacheLayers == 2, "only full-attention layers grow a cache")
        #expect(summary?.headDim == 256)
        #expect(summary?.bits == 4)
    }
}

@Suite("Which engine runs a model")
struct RunnerSupportTests {
    @Test func classifiesByArchitectureAndTask() {
        #expect(RunnerSupport.runner(for: CatalogEntry(repo: "x/a", task: "text-generation", modelType: "llama")) == .chat)
        #expect(RunnerSupport.runner(for: CatalogEntry(repo: "x/b", task: "image-text-to-text", modelType: "qwen3_5")) == .chat,
                "a VLM whose text half the chat engine loads")
        #expect(RunnerSupport.runner(for: CatalogEntry(repo: SeedVR2Variant.threeBInt8.hfRepo, task: "image-to-image", modelType: "seedvr2"))
                == .upscale(.threeBInt8))
        guard case .notYet(let reason) = RunnerSupport.runner(for: CatalogEntry(repo: "x/c", task: "text-generation", modelType: "gpt_oss")) else {
            Issue.record("gpt-oss should be refused"); return
        }
        #expect(reason.contains("harmony"))
        guard case .notYet(let whisper) = RunnerSupport.runner(for: CatalogEntry(repo: "x/d", task: "automatic-speech-recognition", modelType: "whisper")) else {
            Issue.record("whisper isn't runnable yet"); return
        }
        #expect(whisper.contains("Speech to text"))
    }

    @Test func visionModelsChatAndSayTheySeeImages() {
        let qwenVL = CatalogEntry(repo: "x/Qwen2.5-VL-7B", task: "image-text-to-text", modelType: "qwen2_5_vl")
        #expect(RunnerSupport.runner(for: qwenVL) == .chat, "vision-only architectures chat through the vision load")
        #expect(RunnerSupport.seesImages(qwenVL))
        let qwen35Text = CatalogEntry(repo: "x/Qwen3.5-9B-text", task: "text-generation", modelType: "qwen3_5")
        #expect(!RunnerSupport.seesImages(qwen35Text), "its task says text only")
        let config = #"{"model_type":"qwen2_5_vl","num_hidden_layers":28,"num_attention_heads":28,"num_key_value_heads":4,"hidden_size":3584,"vision_config":{"depth":32}}"#
        let summary = ModelConfigSummary.parse(Data(config.utf8))
        #expect(summary?.hasVision == true)
        let spec = ChatModelSpec.estimated(repo: "x/Qwen2.5-VL-7B", weightBytes: 5_000_000_000, config: summary)
        #expect(spec.seesImages)
        #expect(spec.needsVisionLoad, "no text-only load exists for Qwen2.5-VL")
        #expect(ChatCatalog.models.allSatisfy { $0.seesImages }, "Qwen3.5 is natively multimodal")
    }

    @Test func licencesThatRestrictUseMustBeRead() {
        #expect(!LicenseInfo.needsAcknowledgement("apache-2.0"))
        #expect(!LicenseInfo.needsAcknowledgement("MIT"))
        #expect(LicenseInfo.needsAcknowledgement("llama3.3"))
        #expect(LicenseInfo.needsAcknowledgement("other"))
        #expect(LicenseInfo.needsAcknowledgement(nil), "no licence stated is not permission")
    }
}

@Suite("Smart Fit for catalog models")
struct ModelGraderTests {
    let m2Max = mac(memoryGB: 32, availableGB: 19)

    @Test func aCuratedChatModelGetsItsReviewedFigures() throws {
        let entries = try HuggingFaceCatalog.parseList(Data(catalogFixture.utf8))
        let nine = try #require(entries.first { $0.repo == "mlx-community/Qwen3.5-9B-4bit" })
        let verdict = ModelGrader.verdict(for: nine, hardware: m2Max, calibration: CalibrationStore())
        #expect(verdict.grade == .green)
        #expect(verdict.runner == .chat)
        #expect(!verdict.isEstimate, "curated: exact without opening it")
        #expect(verdict.arithmetic.contains("(32k)"))
    }

    @Test func anUncuratedChatModelIsGradedAndLabelledAnEstimate() throws {
        let entries = try HuggingFaceCatalog.parseList(Data(catalogFixture.utf8))
        let other = try #require(entries.first { $0.repo == "mlx-community/Qwen3.5-9B-MLX-4bit" })
        let verdict = ModelGrader.verdict(for: other, hardware: m2Max, calibration: CalibrationStore())
        #expect(verdict.runner == .chat)
        #expect(verdict.isEstimate)
        #expect(verdict.grade == .green)
    }

    @Test func tooLargeIsRedWithTheNumbers() {
        let huge = CatalogEntry(repo: "x/huge", task: "text-generation", modelType: "llama", bits: 4,
                                parameters: ["U32": 120_000_000_000, "BF16": 1_000_000_000])
        let verdict = ModelGrader.verdict(for: huge, hardware: m2Max, calibration: CalibrationStore())
        #expect(verdict.grade == .red)
        #expect(verdict.caveat?.contains("this Mac's GPU can use") == true)
    }

    @Test func notRunnableStillSaysWhetherItWouldFit() {
        let whisper = CatalogEntry(repo: "x/whisper", task: "automatic-speech-recognition", modelType: "whisper",
                                   parameters: ["F16": 800_000_000])
        let verdict = ModelGrader.verdict(for: whisper, hardware: m2Max, calibration: CalibrationStore())
        #expect(verdict.grade == .notRunnable(fitsIfSupported: true))
        #expect(verdict.headline == "LocalLab can't run this yet")
    }

    @Test func openingAModelReplacesTheEstimate() {
        let entry = CatalogEntry(repo: "x/dense", task: "text-generation", modelType: "llama", bits: 4,
                                 parameters: ["U32": 8_000_000_000])
        let details = CatalogDetails(
            repo: "x/dense", revision: "abc", weightBytes: 4_500_000_000, downloadBytes: 4_600_000_000,
            config: ModelConfigSummary(modelType: "llama", layers: 32, cacheLayers: 32, kvHeads: 8, headDim: 128,
                                       maxContext: 131_072, experts: nil, expertsPerToken: nil, bits: 4, groupSize: 64)
        )
        let verdict = ModelGrader.verdict(for: entry, details: details, hardware: m2Max, calibration: CalibrationStore())
        #expect(!verdict.isEstimate)
        #expect(verdict.weightBytes == 4_500_000_000)
    }

    @Test func bestFitPutsWhatRunsWellFirst() {
        let good = (CatalogEntry(repo: "a/good", downloads: 1), ModelVerdict(runner: .chat, grade: .green, headline: "", arithmetic: "", caveat: nil, isEstimate: false, weightBytes: nil))
        let popularButNot = (CatalogEntry(repo: "a/tts", downloads: 1_000_000), ModelVerdict(runner: .notYet("x"), grade: .notRunnable(fitsIfSupported: true), headline: "", arithmetic: "", caveat: nil, isEstimate: false, weightBytes: nil))
        let tooBig = (CatalogEntry(repo: "a/big", downloads: 500), ModelVerdict(runner: .chat, grade: .red, headline: "", arithmetic: "", caveat: nil, isEstimate: false, weightBytes: nil))
        let sorted = [popularButNot, tooBig, good].sorted(by: ModelGrader.bestFitOrder).map(\.0.repo)
        #expect(sorted == ["a/good", "a/big", "a/tts"])
    }

    @Test func uncuratedSpecsEstimateExpertsAndQuality() {
        let config = ModelConfigSummary(modelType: "qwen3_moe", layers: 48, cacheLayers: 48, kvHeads: 4, headDim: 128,
                                        maxContext: 40_960, experts: 128, expertsPerToken: 8, bits: 4, groupSize: 64)
        let spec = ChatModelSpec.estimated(repo: "x/moe", weightBytes: 17_000_000_000, config: config)
        #expect(!spec.isCurated)
        #expect(spec.activeParamsB < spec.totalParamsB / 5, "a mixture of experts reads a fraction per token")
        #expect(spec.thinks)
        let small = ChatModelSpec.estimated(repo: "x/small", weightBytes: 1_000_000_000, config: nil)
        #expect(small.qualityScore < spec.qualityScore)
    }
}
