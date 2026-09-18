import Foundation

/// Reading a safetensors file's header without loading its tensors.
///
/// The format is an 8-byte little-endian header length, a JSON header mapping each tensor to
/// its dtype, shape and byte range, then the raw data. The header alone is enough to prove
/// the file is structurally whole — every tensor's bytes fit the file, and each byte range is
/// exactly `dtype size × element count` — and to list tensor names, without mapping gigabytes
/// of weights. (MLXUI's SeedVR2 check called `MLX.loadArrays` on a 4 GB file just to read
/// names.)
public struct SafetensorsHeader: Sendable {
    public struct Tensor: Sendable, Equatable {
        public let dtype: String
        public let shape: [Int]
        public let begin: Int64
        public let end: Int64
    }

    public let tensors: [String: Tensor]
    public let headerLength: Int64
    public let fileSize: Int64

    public var names: Set<String> { Set(tensors.keys) }

    public enum Failure: Error, Equatable, CustomStringConvertible {
        case unreadable(String)
        case badHeaderLength(Int64)
        case malformedJSON
        case truncated(expectedBytes: Int64, actualBytes: Int64)
        case inconsistentTensor(String)
        case missingTensors([String])

        public var description: String {
            switch self {
            case .unreadable(let why): return "the file can't be read (\(why))"
            case .badHeaderLength(let n): return "the header claims \(n) bytes, which isn't plausible"
            case .malformedJSON: return "the header isn't valid JSON"
            case .truncated(let expected, let actual):
                return "the data is truncated — tensors need \(expected) bytes, the file has \(actual)"
            case .inconsistentTensor(let name):
                return "tensor '\(name)' has a byte range that doesn't match its shape"
            case .missingTensors(let names):
                return "missing tensors: \(names.joined(separator: ", "))"
            }
        }
    }

    /// Bytes per element for the dtypes safetensors defines.
    static let dtypeSizes: [String: Int64] = [
        "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
        "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
        "U32": 4, "I32": 4, "F32": 4,
        "U64": 8, "I64": 8, "F64": 8,
    ]

    public static func read(_ url: URL) throws -> SafetensorsHeader {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw Failure.unreadable(error.localizedDescription)
        }
        defer { try? handle.close() }

        let fileSize = Integrity.fileSize(url)
        guard let lengthBytes = try handle.read(upToCount: 8), lengthBytes.count == 8 else {
            throw Failure.unreadable("shorter than 8 bytes")
        }
        let length = lengthBytes.withUnsafeBytes { Int64(littleEndian: $0.loadUnaligned(as: Int64.self)) }
        guard length > 1, length < 100 << 20, length <= fileSize - 8 else {
            throw Failure.badHeaderLength(length)
        }
        guard let json = try handle.read(upToCount: Int(length)), json.count == Int(length),
              let object = try? JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw Failure.malformedJSON
        }

        var tensors: [String: Tensor] = [:]
        var maxEnd: Int64 = 0
        for (name, value) in object where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int],
                  let offsets = entry["data_offsets"] as? [NSNumber], offsets.count == 2 else {
                throw Failure.inconsistentTensor(name)
            }
            let begin = offsets[0].int64Value, end = offsets[1].int64Value
            guard begin >= 0, end >= begin else { throw Failure.inconsistentTensor(name) }
            if let size = dtypeSizes[dtype] {
                let elements = shape.reduce(Int64(1)) { $0 * Int64($1) }
                guard elements * size == end - begin else { throw Failure.inconsistentTensor(name) }
            }
            tensors[name] = Tensor(dtype: dtype, shape: shape, begin: begin, end: end)
            maxEnd = max(maxEnd, end)
        }

        let dataBytes = fileSize - 8 - length
        guard maxEnd <= dataBytes else {
            throw Failure.truncated(expectedBytes: maxEnd, actualBytes: dataBytes)
        }
        return SafetensorsHeader(tensors: tensors, headerLength: length, fileSize: fileSize)
    }

    /// Throw unless every name in `required` is present. A quantized checkpoint stores a
    /// layer's weight alongside `.scales`/`.biases`; the weight name itself is what's checked.
    public func require(_ required: [String]) throws {
        let missing = required.filter { tensors[$0] == nil }
        guard missing.isEmpty else { throw Failure.missingTensors(missing) }
    }
}
