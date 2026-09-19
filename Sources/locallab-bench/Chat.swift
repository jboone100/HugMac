import Foundation
import LocalLabCore
import LocalLabMLX

/// `locallab-bench --chat <repo> <prompt> [--think] [--followup <prompt>]` — run one reply through the real MLX chat
/// engine from an installed model, stream it, and report speed and memory, then eject and
/// report what came back.
enum ChatCommand {
    static func run(arguments: [String]) async throws {
        var arguments = arguments
        var followup: String?
        if let index = arguments.firstIndex(of: "--followup"), index + 1 < arguments.count {
            followup = arguments[index + 1]
            arguments.removeSubrange(index ... index + 1)
        }
        let positional = arguments.filter { !$0.hasPrefix("--") }
        guard positional.count >= 2 else {
            throw Failure("usage: locallab-bench --chat <repo> <prompt> [--think]")
        }
        let repo = positional[0], prompt = positional[1]
        let store = LibraryLocation.resolve().store
        guard store.isInstalled(repo: repo) else {
            throw Failure("\(repo) isn't installed — locallab-bench --install \(repo)")
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
        if let followup {
            // The same conversation, one message on: only the new message should be read.
            let turns = [ChatTurn(.user, prompt), ChatTurn(.assistant, reply), ChatTurn(.user, followup)]
            var second: ChatStats?
            print("— follow-up: \(followup)\n")
            for try await event in engine.stream(turns, options: options) {
                switch event {
                case .text(let text): if !think { print(text, terminator: "") }
                case .trimmed(let dropped): print("[trimmed \(dropped)]")
                case .finished(let finished): second = finished
                }
            }
            print("\n")
            if let second {
                print(String(format: "Follow-up read %d new tokens in %.2f s, %d already cached · reply %d tokens at %.1f tok/s",
                             second.promptTokens, second.promptSeconds, second.cachedTokens ?? 0,
                             second.generatedTokens, second.tokensPerSecond))
            }
        }
        let loaded = MemoryRelease.heldBytes
        await engine.eject()
        print("Held by MLX: \(gb(loaded)) loaded → \(gb(MemoryRelease.heldBytes)) after eject")
    }
}
