import Foundation
import LocalLabCore
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Runs chat models with `mlx-swift-lm`, loading only files the installer already put in
/// the library — it never downloads (plan §10).
///
/// Holds at most one model. `eject()` actually returns the memory, which MLXUI never did:
/// dropping the container alone leaves MLX's buffer cache holding the weights.
public actor MLXChatEngine: ChatBackend {
    private var container: ModelContainer?
    private var loadedDirectory: URL?

    public init() {}

    public var isLoaded: Bool { container != nil }

    public func load(directory: URL) async throws {
        if loadedDirectory == directory, container != nil { return }
        eject()
        container = try await LLMModelFactory.shared.loadContainer(from: directory, using: HFTokenizerLoader())
        loadedDirectory = directory
    }

    public func eject() {
        container = nil
        loadedDirectory = nil
        MemoryRelease.returnCachedBuffers()
    }

    public nonisolated func stream(
        _ turns: [ChatTurn], options: ChatOptions
    ) -> AsyncThrowingStream<ChatEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { await self.run(turns, options: options, continuation: continuation) }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(
        _ turns: [ChatTurn], options: ChatOptions,
        continuation: AsyncThrowingStream<ChatEvent, Error>.Continuation
    ) async {
        guard let container else {
            continuation.finish(throwing: ChatEngineError.notLoaded)
            return
        }
        do {
            // Fit the prompt and the reply into the context: drop the oldest exchange until it
            // does, always keeping the system prompt and the newest message.
            var kept = turns
            var dropped = 0
            var input: LMInput
            while true {
                input = try await container.prepare(input: Self.userInput(kept, thinking: options.thinking))
                let promptTokens = input.text.tokens.size
                guard promptTokens + options.maxTokens > options.contextTokens,
                      let oldest = kept.firstIndex(where: { $0.role != .system }),
                      oldest < kept.count - 1 else { break }
                kept.remove(at: oldest)
                dropped += 1
            }
            if dropped > 0 { continuation.yield(.trimmed(droppedTurns: dropped)) }

            // Qwen3.5's template opens the thinking block inside the prompt, so the reply
            // starts mid-thought with no `<think>`. Put the tag back so the transcript can
            // fold the thinking away.
            if options.thinking {
                let ids = input.text.tokens.asArray(Int.self).suffix(6)
                let tail = await container.decode(tokenIds: Array(ids))
                if let open = tail.range(of: "<think>", options: .backwards),
                   !tail[open.upperBound...].contains("</think>") {
                    continuation.yield(.text("<think>\n"))
                }
            }

            let parameters = GenerateParameters(maxTokens: options.maxTokens, temperature: options.temperature)
            let generation = try await container.generate(input: input, parameters: parameters)
            var stats: ChatStats?
            for await item in generation {
                if Task.isCancelled { break }
                switch item {
                case .chunk(let text):
                    continuation.yield(.text(text))
                case .info(let info):
                    stats = ChatStats(
                        promptTokens: info.promptTokenCount, generatedTokens: info.generationTokenCount,
                        promptSeconds: info.promptTime, generateSeconds: info.generateTime
                    )
                case .toolCall:
                    break
                }
            }
            if Task.isCancelled {
                continuation.finish(throwing: CancellationError())
                return
            }
            if let stats { continuation.yield(.finished(stats)) }
            continuation.finish()
        } catch {
            continuation.finish(throwing: error)
        }
    }

    static func userInput(_ turns: [ChatTurn], thinking: Bool) -> UserInput {
        let messages: [Chat.Message] = turns.map { turn in
            switch turn.role {
            case .system: .system(turn.content)
            case .user: .user(turn.content)
            case .assistant: .assistant(turn.content)
            }
        }
        // Qwen3.5's template reads `enable_thinking`; templates that don't use it ignore it.
        return UserInput(chat: messages, additionalContext: ["enable_thinking": thinking])
    }
}

public enum ChatEngineError: Error, LocalizedError {
    case notLoaded

    public var errorDescription: String? {
        switch self {
        case .notLoaded: "No chat model is loaded."
        }
    }
}

/// Returning memory to the system: clamp MLX's buffer cache to nothing, clear it, restore
/// the limit. Clearing under a large limit leaves the buffers parked — the reason MLXUI's
/// memory never came back.
public enum MemoryRelease {
    public static func returnCachedBuffers(restoringLimitTo limit: Int? = nil) {
        let restore = limit ?? Memory.cacheLimit
        Memory.cacheLimit = 0
        Memory.clearCache()
        Memory.cacheLimit = restore
    }

    /// Bytes MLX holds right now, in use or cached.
    public static var heldBytes: Int64 {
        Int64(Memory.activeMemory + Memory.cacheMemory)
    }
}

// MARK: - Tokenizer

/// The model's real tokenizer and Jinja chat template, via swift-transformers — ported from
/// MLXUI's `HFTokenizerLoader`, which replaced a hand-rolled tokenizer that produced garbage.
struct HFTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        HFTokenizerBridge(try await AutoTokenizer.from(modelFolder: directory))
    }
}

private struct HFTokenizerBridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    // swift-transformers names it `decode(tokens:)`.
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext
            )
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}
