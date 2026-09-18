import Foundation

/// One file in a repo snapshot, with what it takes to verify it.
public struct RemoteFile: Sendable, Equatable {
    /// Path within the repo.
    public let path: String
    public let size: Int64
    public let expected: Integrity.Expected

    public init(path: String, size: Int64, expected: Integrity.Expected) {
        self.path = path
        self.size = size
        self.expected = expected
    }
}

/// A repo pinned to one commit. Every file an install fetches comes from `revision`, so a
/// repo that updates mid-install can't hand back a mix of old and new files — MLXUI fetched
/// from `resolve/main` and could.
public struct RepoSnapshot: Sendable, Equatable {
    public let repo: String
    public let revision: String
    public let files: [RemoteFile]
    public let gated: Bool

    public init(repo: String, revision: String, files: [RemoteFile], gated: Bool = false) {
        self.repo = repo
        self.revision = revision
        self.files = files
        self.gated = gated
    }
}

public enum HubError: Error, Equatable, CustomStringConvertible {
    /// 401/403: gated or private, and the token is missing or insufficient.
    case needsAuth(repo: String)
    case notFound(repo: String)
    case http(status: Int, repo: String)
    case malformedResponse(repo: String)

    public var description: String {
        switch self {
        case .needsAuth(let repo):
            return "\(repo) is gated or private — add a Hugging Face token in Settings, then retry."
        case .notFound(let repo):
            return "\(repo) doesn't exist on Hugging Face (or isn't visible with this token)."
        case .http(let status, let repo):
            return "Hugging Face returned HTTP \(status) for \(repo)."
        case .malformedResponse(let repo):
            return "Hugging Face sent an unexpected response for \(repo)."
        }
    }

    /// 401/403 → gated; 404 → missing. Pure so it can be tested directly.
    public static func fromStatus(_ status: Int, repo: String) -> HubError? {
        switch status {
        case 200 ..< 300, 416: return nil
        case 401, 403: return .needsAuth(repo: repo)
        case 404: return .notFound(repo: repo)
        default: return .http(status: status, repo: repo)
        }
    }
}

/// The network boundary, as a protocol so the installer can be tested without it.
public protocol HubClient: Sendable {
    func snapshot(repo: String) async throws -> RepoSnapshot

    /// Download `path` at `revision` into `destination`, **appending** from the file's
    /// current length — so a partially downloaded file resumes where it stopped.
    /// `progress` reports the file's total bytes on disk so far.
    func download(
        repo: String,
        revision: String,
        path: String,
        to destination: URL,
        progress: @Sendable @escaping (Int64) -> Void
    ) async throws
}

// MARK: - Hugging Face

public struct HuggingFaceHub: HubClient {
    public let host: String
    let token: @Sendable () -> String?

    public init(host: String = "huggingface.co", token: @Sendable @escaping () -> String? = { Keychain.huggingFaceToken() }) {
        self.host = host
        self.token = token
    }

    public func snapshot(repo: String) async throws -> RepoSnapshot {
        var components = URLComponents()
        components.scheme = "https"
        components.host = host
        components.path = "/api/models/\(repo)"
        components.queryItems = [URLQueryItem(name: "blobs", value: "true")]
        guard let url = components.url else { throw HubError.malformedResponse(repo: repo) }

        var request = URLRequest(url: url)
        request.setValue("HugMac/0.1", forHTTPHeaderField: "User-Agent")
        if let token = token() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse,
           let error = HubError.fromStatus(http.statusCode, repo: repo) {
            throw error
        }
        return try Self.parseSnapshot(data, repo: repo)
    }

    /// Parse `GET /api/models/{repo}?blobs=true`. LFS files carry `lfs.sha256`; small files
    /// carry `blobId`, the git blob SHA-1 — so every file has something to verify against.
    public static func parseSnapshot(_ data: Data, repo: String) throws -> RepoSnapshot {
        struct LFS: Decodable { let sha256: String; let size: Int64 }
        struct Sibling: Decodable {
            let rfilename: String
            let size: Int64?
            let blobId: String?
            let lfs: LFS?
        }
        struct Info: Decodable {
            let sha: String?
            let siblings: [Sibling]?
            let gated: GatedValue?
        }
        // `gated` is `false` or a string ("auto"/"manual").
        enum GatedValue: Decodable {
            case flag(Bool), mode(String)
            init(from decoder: Decoder) throws {
                let container = try decoder.singleValueContainer()
                if let flag = try? container.decode(Bool.self) { self = .flag(flag) } else {
                    self = .mode((try? container.decode(String.self)) ?? "")
                }
            }
            var isGated: Bool {
                switch self {
                case .flag(let flag): flag
                case .mode(let mode): !mode.isEmpty
                }
            }
        }

        guard let info = try? JSONDecoder().decode(Info.self, from: data),
              let revision = info.sha, let siblings = info.siblings else {
            throw HubError.malformedResponse(repo: repo)
        }
        let files = siblings.map { sibling -> RemoteFile in
            if let lfs = sibling.lfs {
                return RemoteFile(path: sibling.rfilename, size: lfs.size, expected: .sha256(lfs.sha256))
            }
            let expected: Integrity.Expected = sibling.blobId.map { .gitBlobSHA1($0) } ?? .none
            return RemoteFile(path: sibling.rfilename, size: sibling.size ?? 0, expected: expected)
        }
        return RepoSnapshot(
            repo: repo, revision: revision, files: files, gated: info.gated?.isGated ?? false
        )
    }

    public func download(
        repo: String,
        revision: String,
        path: String,
        to destination: URL,
        progress: @Sendable @escaping (Int64) -> Void
    ) async throws {
        var components = URLComponents()
        components.scheme = "https"
        components.host = host
        components.path = "/\(repo)/resolve/\(revision)/\(path)"
        guard let url = components.url else { throw HubError.malformedResponse(repo: repo) }

        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        if !FileManager.default.fileExists(atPath: destination.path) {
            FileManager.default.createFile(atPath: destination.path, contents: nil)
        }
        let existing = Integrity.fileSize(destination)

        var request = URLRequest(url: url)
        request.setValue("HugMac/0.1", forHTTPHeaderField: "User-Agent")
        if let token = token() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if existing > 0 {
            request.setValue("bytes=\(existing)-", forHTTPHeaderField: "Range")
        }

        let stream = try StreamingDownload(
            destination: destination, startingAt: existing, repo: repo,
            originalHost: host, progress: progress
        )
        try await stream.run(request)
    }
}

// MARK: - Streaming download

/// Writes a response body straight to disk as it arrives, appending to a partial file.
///
/// MLXUI used a download task, whose file URLSession deletes the moment the delegate returns,
/// so every file was copied out to a temp directory, then copied into staging, then copied
/// into place — up to three copies of a multi-gigabyte file, and a 2× disk requirement.
/// Streaming into the staging file directly needs 1×, and a cancelled or failed download
/// leaves a partial that the next attempt resumes with a `Range` request.
final class StreamingDownload: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let destination: URL
    private let repo: String
    private let originalHost: String
    private let progress: @Sendable (Int64) -> Void
    private let lock = NSLock()

    private var handle: FileHandle?
    private var offset: Int64
    private var continuation: CheckedContinuation<Void, Error>?
    private var finished = false
    private var lastReported = Date.distantPast

    init(
        destination: URL, startingAt offset: Int64, repo: String, originalHost: String,
        progress: @Sendable @escaping (Int64) -> Void
    ) throws {
        self.destination = destination
        self.offset = offset
        self.repo = repo
        self.originalHost = originalHost
        self.progress = progress
        super.init()
    }

    func run(_ request: URLRequest) async throws {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 60
        configuration.timeoutIntervalForResource = 24 * 3600
        let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let task = session.dataTask(with: request)

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                lock.lock()
                self.continuation = continuation
                lock.unlock()
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }

    private func finish(_ result: Result<Void, Error>) {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        finished = true
        let continuation = self.continuation
        self.continuation = nil
        try? handle?.close()
        handle = nil
        let reached = offset
        lock.unlock()
        progress(reached)
        continuation?.resume(with: result)
    }

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            completionHandler(.cancel)
            finish(.failure(HubError.malformedResponse(repo: repo)))
            return
        }
        // 416: we asked for bytes past the end — the partial is already complete.
        if http.statusCode == 416 {
            completionHandler(.cancel)
            finish(.success(()))
            return
        }
        if let error = HubError.fromStatus(http.statusCode, repo: repo) {
            completionHandler(.cancel)
            finish(.failure(error))
            return
        }
        do {
            lock.lock()
            defer { lock.unlock() }
            // 200 in answer to a Range request means the server ignored it: start over
            // rather than append a whole file onto a partial one.
            if http.statusCode == 200 && offset > 0 {
                try FileManager.default.removeItem(at: destination)
                FileManager.default.createFile(atPath: destination.path, contents: nil)
                offset = 0
            }
            let handle = try FileHandle(forWritingTo: destination)
            try handle.seekToEnd()
            self.handle = handle
        } catch {
            completionHandler(.cancel)
            finish(.failure(error))
            return
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        do {
            try handle?.write(contentsOf: data)
            offset += Int64(data.count)
        } catch {
            lock.unlock()
            dataTask.cancel()
            finish(.failure(error))
            return
        }
        let now = Date()
        let shouldReport = now.timeIntervalSince(lastReported) > 0.25
        if shouldReport { lastReported = now }
        let reached = offset
        lock.unlock()
        if shouldReport { progress(reached) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            let cancelled = (error as? URLError)?.code == .cancelled
            finish(.failure(cancelled ? CancellationError() : error))
        } else {
            finish(.success(()))
        }
    }

    /// Hugging Face redirects file downloads to a CDN. Carry the `Range` header across so a
    /// resume stays a resume, and drop the token once it would leave huggingface.co — the
    /// redirect URL is already signed, and the token shouldn't travel to another host.
    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        var redirected = request
        if let range = task.originalRequest?.value(forHTTPHeaderField: "Range") {
            redirected.setValue(range, forHTTPHeaderField: "Range")
        }
        if let host = request.url?.host, host != originalHost, !host.hasSuffix(".\(originalHost)") {
            redirected.setValue(nil, forHTTPHeaderField: "Authorization")
        }
        completionHandler(redirected)
    }
}
