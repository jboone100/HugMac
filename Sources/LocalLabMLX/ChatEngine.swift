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
///
/// **Keeps the conversation's key/value cache between turns.** A follow-up in the same
/// conversation feeds the model only the new message, so a long conversation doesn't re-read
/// itself before every reply. The cache is rebuilt — the conversation read once, oldest
/// messages dropped if it no longer fits — when the conversation, *Think first* or the
/// context length changes, or when it would overflow. It lives with the loaded model and goes
/// when the model is ejected; the conversation itself is kept by the app regardless.
public actor MLXChatEngine: ChatBackend {
    private var container: ModelContainer?
    private var loadedDirectory: URL?
    /// The template opens `<think>` in the prompt (Qwen3.5), so a thinking reply starts
    /// mid-thought and the tag must be put back for the transcript to fold it.
    private var templateOpensThinking = false
    private var session: Session?

    /// A cached conversation: the turns it has seen, as the caller will send them back, and
    /// how many tokens its cache holds.
    struct Session {
        let chat: ChatSession
        var turns: [ChatTurn]
        var tokens: Int
        let thinking: Bool
        let context: Int
    }

    public init() {}

    public var isLoaded: Bool { container != nil }

    public func residentBytes() -> Int64 {
        container == nil ? 0 : MemoryRelease.heldBytes
    }

    public func load(directory: URL) async throws {
        if loadedDirectory == directory, container != nil { return }
        eject()
        container = try await LLMModelFactory.shared.loadContainer(from: directory, using: HFTokenizerLoader())
        loadedDirectory = directory
        templateOpensThinking = Self.templateOpensThinking(in: directory)
    }

    public func eject() {
        session = nil
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
        guard let message = turns.last, message.role == .user else {
            continuation.finish(throwing: ChatEngineError.noMessage)
            return
        }
        let prior = Array(turns.dropLast())
        let parameters = GenerateParameters(maxTokens: options.maxTokens, temperature: options.temperature)
        do {
            // Continue the cached conversation when this is its next message and it still
            // fits; otherwise start one from the history.
            var cachedTokens = 0
            if let current = session, current.turns == prior, current.thinking == options.thinking,
               current.context == options.contextTokens,
               current.tokens + (await container.encode(message.content)).count + 32 + options.maxTokens
                   <= options.contextTokens {
                cachedTokens = current.tokens
            } else {
                let kept = try await fit(turns, options: options, container: container, continuation: continuation)
                let history: [Chat.Message] = kept.dropLast().map(Self.message)
                session = Session(
                    chat: ChatSession(container, history: history, generateParameters: parameters,
                                      additionalContext: ["enable_thinking": options.thinking]),
                    turns: prior, tokens: 0, thinking: options.thinking, context: options.contextTokens
                )
            }
            guard let chat = session?.chat else { throw ChatEngineError.notLoaded }

            if options.thinking && templateOpensThinking { continuation.yield(.text("<think>\n")) }
            var reply = options.thinking && templateOpensThinking ? "<think>\n" : ""
            var stats: ChatStats?
            for try await item in chat.streamDetails(to: message.content, images: [], videos: []) {
                if Task.isCancelled { break }
                switch item {
                case .chunk(let text):
                    reply += text
                    continuation.yield(.text(text))
                case .info(let info):
                    stats = ChatStats(
                        promptTokens: info.promptTokenCount, generatedTokens: info.generationTokenCount,
                        promptSeconds: info.promptTime, generateSeconds: info.generateTime,
                        cachedTokens: cachedTokens
                    )
                case .toolCall:
                    break
                }
            }
            // The turns this cache now holds, exactly as the app will send them back next time.
            session?.turns = turns + [ChatTurn(.assistant, reply)]
            session?.tokens += (stats?.promptTokens ?? 0) + (stats?.generatedTokens ?? 0)
            if Task.isCancelled {
                continuation.finish(throwing: CancellationError())
                return
            }
            if let stats { continuation.yield(.finished(stats)) }
            continuation.finish()
        } catch {
            // A failed turn leaves the cache in an unknown state: start over next time.
            session = nil
            continuation.finish(throwing: error)
        }
    }

    /// The turns that fit the context with room for the reply: the oldest are dropped until
    /// they do, always keeping the newest message.
    private func fit(
        _ turns: [ChatTurn], options: ChatOptions, container: ModelContainer,
        continuation: AsyncThrowingStream<ChatEvent, Error>.Continuation
    ) async throws -> [ChatTurn] {
        var kept = turns
        var dropped = 0
        while true {
            let input = try await container.prepare(input: Self.userInput(kept, thinking: options.thinking))
            guard input.text.tokens.size + options.maxTokens > options.contextTokens,
                  let oldest = kept.firstIndex(where: { $0.role != .system }),
                  oldest < kept.count - 1 else { break }
            kept.remove(at: oldest)
            dropped += 1
        }
        if dropped > 0 { continuation.yield(.trimmed(droppedTurns: dropped)) }
        return kept
    }

    static func message(_ turn: ChatTurn) -> Chat.Message {
        switch turn.role {
        case .system: .system(turn.content)
        case .user: .user(turn.content)
        case .assistant: .assistant(turn.content)
        }
    }

    static func userInput(_ turns: [ChatTurn], thinking: Bool) -> UserInput {
        // Qwen3.5's template reads `enable_thinking`; templates that don't use it ignore it.
        UserInput(chat: turns.map(message), additionalContext: ["enable_thinking": thinking])
    }

    /// Whether the chat template itself opens a `<think>` block when thinking is on.
    static func templateOpensThinking(in directory: URL) -> Bool {
        let jinja = try? String(contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        let config = (try? Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        let template = jinja ?? (config?["chat_template"] as? String) ?? ""
        return template.contains("enable_thinking") && template.contains("'<think>\\n'")
    }
}

public enum ChatEngineError: Error, LocalizedError {
    case notLoaded
    case noMessage

    public var errorDescription: String? {
        switch self {
        case .notLoaded: "No chat model is loaded."
        case .noMessage: "There's no new message to answer."
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
