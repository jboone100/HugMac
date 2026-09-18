import Foundation
import Testing
@testable import HugMacCore

private let gib = 1_073_741_824.0

private func mac(_ chip: String, cores: Int, memoryGB: Double, availableGB: Double) -> HardwareProfile {
    let total = Int64(memoryGB * gib)
    let generation = HardwareProfile.generation(from: chip)
    let tier = HardwareProfile.tier(from: chip)
    return HardwareProfile(
        chipName: chip, generation: generation, tier: tier, gpuCoreCount: cores,
        memoryBandwidthGBps: HardwareProfile.bandwidth(generation: generation, tier: tier),
        totalMemoryBytes: total, availableMemoryBytes: Int64(availableGB * gib),
        gpuWiredLimitBytes: HardwareProfile.wiredLimit(total: total),
        macOSVersion: .init(majorVersion: 26, minorVersion: 6, patchVersion: 2)
    )
}

private func spec(_ size: String) -> ChatModelSpec {
    ChatCatalog.models.first { $0.repo == "mlx-community/Qwen3.5-\(size)-4bit" }!
}

@Suite("Chat model picker")
struct ChatPickerTests {
    let m2Max = mac("Apple M2 Max", cores: 30, memoryGB: 32, availableGB: 19)

    @Test func cacheGrowsOnlyOnFullAttentionLayers() {
        // Qwen3.5 9B: 32 layers, one in four keeps a cache; 4 KV heads × 256 dims, fp16.
        #expect(spec("9B").kvLayers == 8)
        #expect(spec("9B").kvCacheBytes(context: 32_768) == 1_073_741_824)
    }

    @Test func mixtureOfExpertsIsPricedByWhatItReadsPerToken() {
        let moe = spec("35B-A3B")
        #expect(abs(moe.activeWeightBytes - Double(moe.weightBytes) * 3 / 35) < 1)
        let picker = ChatModelPicker(hardware: m2Max, calibration: CalibrationStore())
        #expect(picker.speed(of: moe).tokensPerSecond > picker.speed(of: spec("9B")).tokensPerSecond)
    }

    @Test func gradesShowTheirArithmetic() {
        let picker = ChatModelPicker(hardware: m2Max, calibration: CalibrationStore())
        let nine = picker.fit(spec("9B"))
        #expect(nine.grade == .green)
        #expect(nine.context == 32_768)
        #expect(nine.arithmetic.contains("GB weights +"))
        #expect(nine.arithmetic.contains("(32k)"))
        #expect(picker.fit(spec("122B-A10B")).grade.isRed)
        guard case .yellow = picker.fit(spec("27B")).grade else {
            Issue.record("27B on a 32 GB Mac should run, with a caveat")
            return
        }
    }

    @Test func automaticPrefersAModelThatRunsWellOverOneThatBarelyFits() {
        let picker = ChatModelPicker(hardware: m2Max, calibration: CalibrationStore())
        let choice = picker.automatic(installed: [spec("9B").repo, spec("27B").repo])
        #expect(choice?.spec == spec("9B"))
    }

    @Test func automaticTakesTheBestOfSeveralGoodOnes() {
        let big = mac("Apple M4 Max", cores: 40, memoryGB: 128, availableGB: 100)
        let picker = ChatModelPicker(hardware: big, calibration: CalibrationStore())
        let choice = picker.automatic(installed: [spec("4B").repo, spec("9B").repo, spec("27B").repo])
        #expect(choice?.spec == spec("27B"))
    }

    @Test func nothingInstalledMeansNothingAutomatic() {
        let picker = ChatModelPicker(hardware: m2Max, calibration: CalibrationStore())
        #expect(picker.automatic(installed: []) == nil)
    }

    @Test func recommendationsFitTheMac() {
        let small = mac("Apple M1", cores: 8, memoryGB: 8, availableGB: 5)
        let smallPicks = ChatModelPicker(hardware: small, calibration: CalibrationStore()).recommendations()
        #expect(smallPicks.map(\.spec) == [spec("2B"), spec("0.8B")])

        let big = mac("Apple M4 Max", cores: 40, memoryGB: 128, availableGB: 100)
        let bigPicks = ChatModelPicker(hardware: big, calibration: CalibrationStore()).recommendations()
        #expect(bigPicks.first?.spec == spec("122B-A10B"))
    }

    @Test func aBetterModelIsNamedOnlyWhenItIsBetter() {
        let picker = ChatModelPicker(hardware: m2Max, calibration: CalibrationStore())
        let small = picker.fit(spec("0.8B"))
        #expect(picker.betterThan(small, installed: [small.spec.repo]) != nil)
        let best = picker.recommendations(limit: 1).first
        #expect(picker.betterThan(best, installed: [best?.spec.repo ?? ""]) == nil)
    }

    @Test func aMeasuredReplyReplacesTheEstimate() {
        let machine = m2Max.machineKey
        let nine = spec("9B")
        // 200 tokens in 4 s: 50 tok/s for the 9B, measured.
        let sample = CalibrationSample(
            engineID: ChatModelPicker.engineID, phase: "decode",
            workUnits: 200 * nine.activeWeightBytes / 1e9, seconds: 4, peakBytes: 0, machine: machine
        )
        let picker = ChatModelPicker(hardware: m2Max, calibration: CalibrationStore(samples: [sample]))
        let speed = picker.speed(of: nine)
        #expect(abs(speed.tokensPerSecond - 50) < 0.01)
        guard case .measured = speed.source else {
            Issue.record("expected measured, got \(speed.source)")
            return
        }
        // …and prices another model by bytes read.
        #expect(abs(picker.speed(of: spec("4B")).tokensPerSecond - 50 * nine.activeWeightBytes / spec("4B").activeWeightBytes) < 0.01)
    }

    @Test func probesBeatTheSpecSheet() {
        var probes = ProbeStore()
        probes.record(ProbeReport(
            machine: m2Max.machineKey, macOSVersion: "26.6.2", durationSeconds: 5,
            results: [ProbeResult(kind: .memoryBandwidth, value: 320, detail: "test")]
        ))
        let picker = ChatModelPicker(hardware: m2Max, calibration: CalibrationStore(probes: probes))
        guard case .extrapolated(_, _, .probes) = picker.speed(of: spec("9B")).source else {
            Issue.record("expected a probe-based estimate")
            return
        }
    }
}

@Suite("Markdown blocks")
struct MarkdownTests {
    @Test func headingsParagraphsAndRules() {
        let blocks = MarkdownParser.parse("# Title\n\nSome *text*\nmore\n\n---\n## Next")
        #expect(blocks == [
            .heading(level: 1, text: "Title"),
            .paragraph("Some *text*\nmore"),
            .rule,
            .heading(level: 2, text: "Next"),
        ])
    }

    @Test func aHashWithoutASpaceIsNotAHeading() {
        #expect(MarkdownParser.parse("#hashtag") == [.paragraph("#hashtag")])
    }

    @Test func fencedCodeKeepsItsLanguageAndIndentation() {
        let blocks = MarkdownParser.parse("Look:\n```swift\nlet x = 1\n    y()\n```\nDone")
        #expect(blocks == [
            .paragraph("Look:"),
            .code(language: "swift", text: "let x = 1\n    y()", closed: true),
            .paragraph("Done"),
        ])
    }

    @Test func anUnfinishedFenceIsCodeInProgressNotPlainText() {
        #expect(MarkdownParser.parse("```python\nprint(1)") == [.code(language: "python", text: "print(1)", closed: false)])
    }

    @Test func lists() {
        #expect(MarkdownParser.parse("- one\n- two\n  continued\n\n1. a\n2) b") == [
            .list(ordered: false, items: ["one", "two\ncontinued"]),
            .list(ordered: true, items: ["a", "b"]),
        ])
    }

    @Test func tables() {
        let blocks = MarkdownParser.parse("| Model | GB |\n|---|--:|\n| 9B | 6 |\n| 27B | 16 |")
        #expect(blocks == [.table(header: ["Model", "GB"], rows: [["9B", "6"], ["27B", "16"]])])
    }

    @Test func quotes() {
        #expect(MarkdownParser.parse("> said\n> twice") == [.quote("said\ntwice")])
    }

    @Test func thinkingIsSeparatedFromTheAnswer() {
        #expect(MarkdownParser.parse("<think>\nhmm\n</think>\n\nAnswer") == [
            .thinking("hmm", closed: true), .paragraph("Answer"),
        ])
        #expect(MarkdownParser.parse("<think>still going") == [.thinking("still going", closed: false)])
        // Qwen3.5 opens the block in the prompt; the reply only closes it.
        #expect(MarkdownParser.parse("Let me work it out.\n</think>\n\n391") == [
            .thinking("Let me work it out.", closed: true), .paragraph("391"),
        ])
    }

    @Test func autoDetectsJSONAndLeavesProseAlone() {
        guard case .json(let pretty) = OutputClassifier.classify(" {\"b\":1,\"a\":[1,2]} ") else {
            Issue.record("expected JSON")
            return
        }
        #expect(pretty.contains("\"a\" : ["))
        #expect(OutputClassifier.classify("{ not json") == .markdown)
        #expect(OutputClassifier.classify("Plain words.") == .markdown)
    }
}

@Suite("Conversations")
struct ConversationTests {
    @Test func titlesComeFromTheFirstLine() {
        #expect(Conversation.title(forFirstPrompt: "Hello\nsecond line") == "Hello")
        let long = String(repeating: "word ", count: 20)
        #expect(Conversation.title(forFirstPrompt: long).count == 48)
        #expect(Conversation.title(forFirstPrompt: "   ") == Conversation.untitled)
    }

    @Test func storeRoundTripsNewestFirst() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("hugmac-conv-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ConversationStore(directory: directory)
        var older = Conversation(title: "older", createdAt: Date(timeIntervalSince1970: 1_000))
        older.messages = [ChatMessage(role: .user, text: "hi"),
                          ChatMessage(role: .assistant, text: "**hello**",
                                      stats: ChatStats(promptTokens: 5, generatedTokens: 40, promptSeconds: 0.1, generateSeconds: 1))]
        let newer = Conversation(title: "newer", createdAt: Date(timeIntervalSince1970: 2_000))
        try store.save(older)
        try store.save(newer)
        let loaded = store.loadAll()
        #expect(loaded.map(\.title) == ["newer", "older"])
        #expect(loaded[1].messages == older.messages)
        store.delete(newer.id)
        #expect(store.loadAll().map(\.title) == ["older"])
    }
}
