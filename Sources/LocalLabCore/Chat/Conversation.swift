import Foundation

/// How a message is shown. `auto` classifies the text (plan §5.5); the others override it.
public enum RenderMode: String, Sendable, Codable, CaseIterable {
    case auto, markdown, raw
}

public struct ChatMessage: Sendable, Codable, Equatable, Identifiable {
    public enum Role: String, Sendable, Codable { case user, assistant }

    public let id: UUID
    public let role: Role
    public var text: String
    /// Generation figures for an assistant reply, once it finishes.
    public var stats: ChatStats?
    /// A per-message override of the renderer; nil follows the conversation's default.
    public var renderMode: RenderMode?
    /// The reply was stopped before it finished.
    public var stopped: Bool

    public init(id: UUID = UUID(), role: Role, text: String, stats: ChatStats? = nil,
                renderMode: RenderMode? = nil, stopped: Bool = false) {
        self.id = id
        self.role = role
        self.text = text
        self.stats = stats
        self.renderMode = renderMode
        self.stopped = stopped
    }
}

public struct ChatStats: Sendable, Codable, Equatable {
    public let promptTokens: Int
    public let generatedTokens: Int
    public let promptSeconds: Double
    public let generateSeconds: Double

    public init(promptTokens: Int, generatedTokens: Int, promptSeconds: Double, generateSeconds: Double) {
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.promptSeconds = promptSeconds
        self.generateSeconds = generateSeconds
    }

    public var tokensPerSecond: Double {
        generateSeconds > 0 ? Double(generatedTokens) / generateSeconds : 0
    }
}

/// One conversation, stored as a file in the per-user Application Support folder — local,
/// never sent anywhere (plan §5.12).
public struct Conversation: Sendable, Codable, Equatable, Identifiable {
    public let id: UUID
    public var title: String
    /// The model that answered last.
    public var modelRepo: String?
    public let createdAt: Date
    public var updatedAt: Date
    public var messages: [ChatMessage]

    public static let untitled = "New chat"

    public init(id: UUID = UUID(), title: String = Conversation.untitled, modelRepo: String? = nil,
                createdAt: Date = Date(), messages: [ChatMessage] = []) {
        self.id = id
        self.title = title
        self.modelRepo = modelRepo
        self.createdAt = createdAt
        self.updatedAt = createdAt
        self.messages = messages
    }

    /// The first line of the first prompt, shortened — a title the user would recognise.
    public static func title(forFirstPrompt prompt: String) -> String {
        let line = prompt.split(whereSeparator: \.isNewline).first.map(String.init) ?? prompt
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.count > 48 else { return trimmed.isEmpty ? untitled : trimmed }
        return String(trimmed.prefix(47)) + "…"
    }
}

public struct ConversationStore: Sendable {
    public let directory: URL

    public init(directory: URL = ConversationStore.defaultDirectory()) {
        self.directory = directory
    }

    public static func defaultDirectory() -> URL {
        CalibrationStore.defaultURL().deletingLastPathComponent()
            .appendingPathComponent("conversations", isDirectory: true)
    }

    /// Every saved conversation, most recently used first. Unreadable files are skipped.
    public func loadAll() -> [Conversation] {
        let decoder = Self.decoder
        let files = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )) ?? []
        return files.filter { $0.pathExtension == "json" }
            .compactMap { try? decoder.decode(Conversation.self, from: Data(contentsOf: $0)) }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    public func save(_ conversation: Conversation) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Self.encoder.encode(conversation).write(to: url(for: conversation.id), options: .atomic)
    }

    public func delete(_ id: UUID) {
        try? FileManager.default.removeItem(at: url(for: id))
    }

    func url(for id: UUID) -> URL {
        directory.appendingPathComponent("\(id.uuidString).json")
    }

    static var encoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    static var decoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}

// MARK: - The engine contract

/// One turn sent to a chat engine.
public struct ChatTurn: Sendable, Equatable {
    public enum Role: String, Sendable { case system, user, assistant }
    public let role: Role
    public let content: String

    public init(_ role: Role, _ content: String) {
        self.role = role
        self.content = content
    }
}

public struct ChatOptions: Sendable, Equatable {
    public var maxTokens: Int
    public var temperature: Float
    /// Prompt plus reply must fit in this many tokens; the oldest turns go first.
    public var contextTokens: Int
    /// Let a reasoning model think before answering. Off by default: faster replies, and
    /// the thinking is shown folded when it's on.
    public var thinking: Bool

    public init(maxTokens: Int = 2_048, temperature: Float = 0.7, contextTokens: Int = 8_192, thinking: Bool = false) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.contextTokens = contextTokens
        self.thinking = thinking
    }
}

public enum ChatEvent: Sendable, Equatable {
    case text(String)
    /// The oldest turns were left out to fit the context.
    case trimmed(droppedTurns: Int)
    case finished(ChatStats)
}

/// What the chat screen needs from an engine. The MLX implementation lives in LocalLabMLX;
/// tests supply a fake.
public protocol ChatBackend: Sendable {
    /// Load a model from a directory the installer filled. Loading a different model
    /// replaces the resident one.
    func load(directory: URL) async throws
    func stream(_ turns: [ChatTurn], options: ChatOptions) -> AsyncThrowingStream<ChatEvent, Error>
    /// Drop the model and give its memory back to the system.
    func eject() async
    /// Bytes the loaded model holds now — 0 when nothing is loaded.
    func residentBytes() async -> Int64
}

public extension ChatBackend {
    func residentBytes() async -> Int64 { 0 }
}
