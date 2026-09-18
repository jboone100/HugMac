import CryptoKit
import Foundation

/// Proving a downloaded file is the file the repo published.
///
/// MLXUI checked size only, within a kilobyte, and skipped the check entirely when the API
/// reported size 0 — so a truncated or corrupt file could receive an `.installed` marker. The
/// HuggingFace API publishes a SHA-256 for every LFS file and a git blob SHA-1 for every
/// small one, so every file can be verified exactly.
public enum Integrity {

    public enum Expected: Sendable, Equatable {
        /// LFS files: SHA-256 of the content.
        case sha256(String)
        /// Small, non-LFS files: git's blob id, SHA-1 of `"blob <size>\0" + content`.
        case gitBlobSHA1(String)
        /// Nothing published — size is all there is to check.
        case none
    }

    public enum Failure: Error, Equatable, CustomStringConvertible {
        case missing(String)
        case sizeMismatch(path: String, expected: Int64, actual: Int64)
        case hashMismatch(path: String)

        public var description: String {
            switch self {
            case .missing(let path):
                return "\(path) is missing"
            case .sizeMismatch(let path, let expected, let actual):
                return "\(path) is \(actual) bytes, expected \(expected)"
            case .hashMismatch(let path):
                return "\(path) does not match the published checksum"
            }
        }
    }

    /// Verify size, then content hash. Reads the file once, in large blocks.
    public static func verify(file: URL, displayPath: String, size: Int64, expected: Expected) throws {
        guard FileManager.default.fileExists(atPath: file.path) else {
            throw Failure.missing(displayPath)
        }
        let actual = fileSize(file)
        guard actual == size else {
            throw Failure.sizeMismatch(path: displayPath, expected: size, actual: actual)
        }
        let digest: String
        switch expected {
        case .none:
            return
        case .sha256(let hex):
            digest = try hash(file: file, into: SHA256())
            guard digest == hex.lowercased() else { throw Failure.hashMismatch(path: displayPath) }
        case .gitBlobSHA1(let hex):
            var hasher = Insecure.SHA1()
            hasher.update(data: Data("blob \(size)\u{0}".utf8))
            digest = try hash(file: file, into: hasher)
            guard digest == hex.lowercased() else { throw Failure.hashMismatch(path: displayPath) }
        }
    }

    public static func fileSize(_ url: URL) -> Int64 {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes?[.size] as? NSNumber)?.int64Value ?? 0
    }

    static func hash<H: HashFunction>(file: URL, into initial: H) throws -> String {
        var hasher = initial
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
