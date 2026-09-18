import Foundation
import HugMacCore
import HugMacMLX

/// `hugmac-bench --chat <repo> <prompt> [--think]` — run one reply through the real MLX chat
/// engine from an installed model, stream it, and report speed and memory, then eject and
/// report what came back.
enum ChatCommand {
    static func run(arguments: [String]) async throws {
        let positional = arguments.filter { !$0.hasPrefix("--") }
        guard positional.count >= 2 else {
            throw Failure("usage: hugmac-bench --chat <repo> <prompt> [--think]")
        }
        let repo = positional[0], prompt = positional[1]
        let store = ModelStore()
        guard store.isInstalled(repo: repo) else {
            throw Failure("\(repo) isn't installed — hugmac-bench --install \(repo)")
        }
        let hardware = HardwareProfile.detect()
        let picker = ChatModelPicker(hardware: hardware, calibration: CalibrationStore.load())
        if let spec = ChatCatalog.spec(forRepo: repo) {
            print("Plan: " + picker.fit(spec).arithmetic)
        }

        let engine = MLXChatEngine()
        let before = MemoryRelease.heldBytes
        let loadStart = Date()
        try await engine.load(directory: store.directory(forRepo: repo))
        print(String(format: "Loaded in %.1f s, %@ held\n", Date().timeIntervalSince(loadStart),
                     gb(MemoryRelease.heldBytes - before)))

        var stats: ChatStats?
        let think = arguments.contains("--think")
        let options = ChatOptions(maxTokens: think ? 4_096 : 512, thinking: think)
        var reply = ""
        for try await event in engine.stream([ChatTurn(.user, prompt)], options: options) {
            switch event {
            case .text(let text):
                reply += text
                if !think { print(text, terminator: "") }
            case .trimmed(let dropped): print("[trimmed \(dropped)]")
            case .finished(let finished): stats = finished
            }
        }
        print("\n")
        // How the transcript will render it.
        let blocks = MarkdownParser.parse(reply).map { block -> String in
            switch block {
            case .thinking(let text, let closed): "thinking(\(closed ? "closed" : "open"), \(text.split(whereSeparator: \.isWhitespace).count) words)"
            case .paragraph(let text): "paragraph: \(text.prefix(80))"
            case .table(let header, let rows): "table \(header.count)×\(rows.count)"
            case .code(let language, _, let closed): "code(\(language ?? "-"), \(closed ? "closed" : "open"))"
            case .list(let ordered, let items): "\(ordered ? "ordered" : "bulleted") list of \(items.count)"
            case .heading(let level, let text): "h\(level) \(text)"
            case .quote: "quote"
            case .rule: "rule"
            }
        }
        print("Rendered as: " + blocks.joined(separator: " · "))
        if let stats {
            print(String(format: "Prompt %d tokens in %.2f s · reply %d tokens at %.1f tok/s",
                         stats.promptTokens, stats.promptSeconds, stats.generatedTokens, stats.tokensPerSecond))
        }
        let loaded = MemoryRelease.heldBytes
        await engine.eject()
        print("Held by MLX: \(gb(loaded)) loaded → \(gb(MemoryRelease.heldBytes)) after eject")
    }
}
