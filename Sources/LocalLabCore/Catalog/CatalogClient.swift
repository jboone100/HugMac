import Foundation

public enum CatalogSort: String, Sendable, CaseIterable, Codable {
    case bestFit, downloads, likes, recent

    /// The API's sort key. "Best fit" is applied here, over the most-downloaded results.
    var apiKey: String {
        switch self {
        case .bestFit, .downloads: "downloads"
        case .likes: "likes"
        case .recent: "lastModified"
        }
    }
}

public enum CatalogPublisher: String, Sendable, CaseIterable, Codable {
    /// The organisation that does most MLX conversions — the default.
    case mlxCommunity
    /// Everyone publishing MLX weights, including republished and modified models.
    case everyone
}

/// A task filter, in a user's words, mapped to the API's pipeline tags.
public enum CatalogTask: String, Sendable, CaseIterable, Codable {
    case chat, vision, speechToText, textToSpeech, image, video, upscale, embeddings

    public var title: String {
        switch self {
        case .chat: "Chat"
        case .vision: "Image Q&A"
        case .speechToText: "Speech to text"
        case .textToSpeech: "Text to speech"
        case .image: "Image generation"
        case .video: "Video"
        case .upscale: "Upscale"
        case .embeddings: "Embeddings"
        }
    }

    var pipelineTag: String {
        switch self {
        case .chat: "text-generation"
        case .vision: "image-text-to-text"
        case .speechToText: "automatic-speech-recognition"
        case .textToSpeech: "text-to-speech"
        case .image: "text-to-image"
        case .video: "text-to-video"
        case .upscale: "image-to-image"
        case .embeddings: "feature-extraction"
        }
    }
}

public struct CatalogQuery: Sendable, Equatable, Codable, Hashable {
    public var text: String
    public var task: CatalogTask?
    public var sort: CatalogSort
    public var publisher: CatalogPublisher

    public init(text: String = "", task: CatalogTask? = nil, sort: CatalogSort = .bestFit,
                publisher: CatalogPublisher = .mlxCommunity) {
        self.text = text
        self.task = task
        self.sort = sort
        self.publisher = publisher
    }
}

public struct CatalogPage: Sendable, Equatable {
    public let entries: [CatalogEntry]
    /// The API's cursor for the next page, from its `Link` header.
    public let next: URL?

    public init(entries: [CatalogEntry], next: URL?) {
        self.entries = entries
        self.next = next
    }
}

/// Where Browse gets models. The only thing sent is the search itself, to huggingface.co —
/// the same host LocalLab installs from (plan §10).
public protocol CatalogClient: Sendable {
    func search(_ query: CatalogQuery, pageSize: Int) async throws -> CatalogPage
    func page(at url: URL) async throws -> CatalogPage
    func details(repo: String) async throws -> CatalogDetails
}

public struct HuggingFaceCatalog: CatalogClient {
    public let host: String
    let token: @Sendable () -> String?

    public init(host: String = "huggingface.co", token: @Sendable @escaping () -> String? = { Keychain.huggingFaceToken() }) {
        self.host = host
        self.token = token
    }

    static let expansions = ["downloads", "likes", "lastModified", "pipeline_tag", "library_name",
                             "gated", "tags", "cardData", "config", "safetensors"]

    public func url(for query: CatalogQuery, pageSize: Int) -> URL? {
        var components = URLComponents()
        components.scheme = "https"
        components.host = host
        components.path = "/api/models"
        var items = [
            URLQueryItem(name: "filter", value: "mlx"),
            URLQueryItem(name: "sort", value: query.sort.apiKey),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: String(pageSize)),
        ]
        if query.publisher == .mlxCommunity { items.append(URLQueryItem(name: "author", value: "mlx-community")) }
        let text = query.text.trimmingCharacters(in: .whitespaces)
        if !text.isEmpty { items.append(URLQueryItem(name: "search", value: text)) }
        if let task = query.task { items.append(URLQueryItem(name: "pipeline_tag", value: task.pipelineTag)) }
        items += Self.expansions.map { URLQueryItem(name: "expand[]", value: $0) }
        components.queryItems = items
        return components.url
    }

    public func search(_ query: CatalogQuery, pageSize: Int) async throws -> CatalogPage {
        guard let url = url(for: query, pageSize: pageSize) else { throw HubError.malformedResponse(repo: "search") }
        return try await page(at: url)
    }

    public func page(at url: URL) async throws -> CatalogPage {
        let (data, response) = try await URLSession.shared.data(for: request(url))
        if let http = response as? HTTPURLResponse, let error = HubError.fromStatus(http.statusCode, repo: "search") {
            throw error
        }
        let link = (response as? HTTPURLResponse)?.value(forHTTPHeaderField: "Link")
        return CatalogPage(entries: try Self.parseList(data), next: link.flatMap(Self.nextURL))
    }

    public func details(repo: String) async throws -> CatalogDetails {
        let snapshot = try await HuggingFaceHub(host: host, token: token).snapshot(repo: repo)
        let weights = snapshot.files.filter { $0.path.hasSuffix(".safetensors") }.reduce(Int64(0)) { $0 + $1.size }
        let total = snapshot.files.reduce(Int64(0)) { $0 + $1.size }
        var config: ModelConfigSummary?
        if snapshot.files.contains(where: { $0.path == "config.json" }),
           let url = URL(string: "https://\(host)/\(repo)/resolve/\(snapshot.revision)/config.json") {
            let (data, response) = try await URLSession.shared.data(for: request(url))
            if (response as? HTTPURLResponse)?.statusCode == 200 { config = ModelConfigSummary.parse(data) }
        }
        return CatalogDetails(repo: repo, revision: snapshot.revision, weightBytes: weights,
                              downloadBytes: total, config: config)
    }

    func request(_ url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        request.setValue("LocalLab/0.1", forHTTPHeaderField: "User-Agent")
        if let token = token() { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        request.timeoutInterval = 20
        return request
    }

    public static func parseList(_ data: Data) throws -> [CatalogEntry] {
        guard let array = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw HubError.malformedResponse(repo: "search")
        }
        return array.compactMap(CatalogEntry.parse)
    }

    /// `<https://…&cursor=…>; rel="next"` → the URL.
    public static func nextURL(fromLinkHeader header: String) -> URL? {
        for part in header.split(separator: ",") where part.contains("rel=\"next\"") || part.contains("rel=next") {
            guard let open = part.firstIndex(of: "<"), let close = part.firstIndex(of: ">"), open < close else { continue }
            return URL(string: String(part[part.index(after: open) ..< close]))
        }
        return nil
    }
}

/// The last results for each search, kept per-user so Browse still shows something offline
/// — labelled with when it was fetched.
public struct CatalogCache: Sendable {
    public let directory: URL

    public init(directory: URL = CalibrationStore.defaultURL().deletingLastPathComponent()
        .appendingPathComponent("catalog-cache", isDirectory: true)) {
        self.directory = directory
    }

    struct Stored: Codable {
        let fetched: Date
        let entries: [CatalogEntry]
    }

    public func save(_ entries: [CatalogEntry], for query: CatalogQuery, at date: Date = Date()) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? JSONEncoder().encode(Stored(fetched: date, entries: entries)).write(to: url(for: query), options: .atomic)
    }

    public func load(_ query: CatalogQuery) -> (entries: [CatalogEntry], fetched: Date)? {
        guard let data = try? Data(contentsOf: url(for: query)),
              let stored = try? JSONDecoder().decode(Stored.self, from: data) else { return nil }
        return (stored.entries, stored.fetched)
    }

    func url(for query: CatalogQuery) -> URL {
        let key = [query.text.lowercased(), query.task?.rawValue ?? "all", query.sort.rawValue, query.publisher.rawValue]
            .joined(separator: "|")
        let safe = key.unicodeScalars.map { CharacterSet.alphanumerics.contains($0) ? String($0) : "_" }.joined()
        return directory.appendingPathComponent("\(safe.prefix(120)).json")
    }
}
