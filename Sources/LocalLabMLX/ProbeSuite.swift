import Darwin
import Foundation
import LocalLabCore
import MLX
import MLXFast
import MLXRandom

/// The first-run probes (plan §5.14): measure this Mac with small MLX kernels shaped like
/// the work LocalLab's engines do, **without downloading a model**.
///
/// Each probe warms up, then times several rounds and keeps the best — the steady-state
/// rate once clocks have settled, which is what a long job runs at. The results replace
/// spec-sheet guesses when a reference Mac's timing is scaled to this one.
public enum ProbeSuite {
    public struct Progress: Sendable {
        public let completed: Int
        public let total: Int
        /// The probe about to run, or nil when finished.
        public let running: ProbeKind?
    }

    /// Run every probe. `library` is where models live: the disk probe measures that
    /// volume, not the boot disk.
    public static func run(
        hardware: HardwareProfile = .detect(),
        library: URL,
        progress: @Sendable (Progress) -> Void = { _ in }
    ) async throws -> ProbeReport {
        let started = Date()
        let kinds = ProbeKind.allCases
        var results: [ProbeResult] = []
        let previousCacheLimit = Memory.cacheLimit
        defer { SeedVR2Residency.releaseBuffers(restoringLimitTo: previousCacheLimit) }

        for (index, kind) in kinds.enumerated() {
            try Task.checkCancellation()
            progress(Progress(completed: index, total: kinds.count, running: kind))
            // Yield so a cancel or a UI update gets in between the GPU-bound probes.
            await Task.yield()
            let result = try run(kind, hardware: hardware, library: library)
            results.append(result)
            SeedVR2Residency.releaseBuffers(restoringLimitTo: previousCacheLimit)
        }
        progress(Progress(completed: kinds.count, total: kinds.count, running: nil))

        return ProbeReport(
            machine: hardware.machineKey,
            macOSVersion: hardware.macOSVersionString,
            durationSeconds: Date().timeIntervalSince(started),
            results: results
        )
    }

    static func run(_ kind: ProbeKind, hardware: HardwareProfile, library: URL) throws -> ProbeResult {
        switch kind {
        case .memoryBandwidth: bandwidth()
        case .matmulF16: matmul()
        case .quantizedMatmul4: quantizedMatmul(bits: 4, kind: .quantizedMatmul4)
        case .quantizedMatmul8: quantizedMatmul(bits: 8, kind: .quantizedMatmul8)
        case .attention: attention()
        case .conv3d: conv3d()
        case .memoryHeadroom: headroom(hardware: hardware)
        case .diskRead: try diskRead(library: library)
        }
    }

    // MARK: - Timing

    /// Best seconds per iteration over `rounds`, after `warmup` untimed iterations.
    static func time(
        warmup: Int = 2, rounds: Int = 3, iterations: Int, _ body: () -> MLXArray
    ) -> Double {
        for _ in 0 ..< warmup { eval(body()) }
        var best = Double.infinity
        for _ in 0 ..< rounds {
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< iterations { eval(body()) }
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            best = min(best, elapsed / Double(iterations))
        }
        return best
    }

    static let gigabyte = 1e9
    static let teraflop = 1e12

    // MARK: - Probes

    /// Stream 1 GiB through a reduction: what LLM token generation and weight loading are
    /// bound by.
    static func bandwidth() -> ProbeResult {
        let count = 256 * 1024 * 1024
        let buffer = MLXArray.ones([count], dtype: .float32)
        eval(buffer)
        let seconds = time(iterations: 10) { buffer.sum() }
        let bytes = Double(count * 4)
        return ProbeResult(
            kind: .memoryBandwidth, value: bytes / seconds / gigabyte,
            detail: "sum over 1 GiB fp32, best of 3 × 10"
        )
    }

    /// Dense fp16 multiply: the diffusion transformer's dominant cost.
    static func matmul() -> ProbeResult {
        let n = 4096
        let a = MLXRandom.normal([n, n]).asType(.float16)
        let b = MLXRandom.normal([n, n]).asType(.float16)
        eval(a, b)
        let seconds = time(iterations: 10) { MLX.matmul(a, b) }
        let flops = 2.0 * Double(n) * Double(n) * Double(n)
        return ProbeResult(
            kind: .matmulF16, value: flops / seconds / teraflop,
            detail: "\(n)×\(n) · \(n)×\(n) fp16, best of 3 × 10"
        )
    }

    /// Multiply against quantized weights, at a batch large enough to be compute-bound:
    /// prompt processing for a quantized LLM, the denoise phase of an int8 SeedVR2.
    static func quantizedMatmul(bits: Int, kind: ProbeKind) -> ProbeResult {
        let n = 4096, batch = 2048, group = 64
        let weights = MLXRandom.normal([n, n]).asType(.float16)
        let (packed, scales, biases) = quantized(weights, groupSize: group, bits: bits)
        let x = MLXRandom.normal([batch, n]).asType(.float16)
        eval(packed, scales, x)
        if let biases { eval(biases) }
        let seconds = time(iterations: 10) {
            quantizedMM(x, packed, scales: scales, biases: biases, transpose: true, groupSize: group, bits: bits)
        }
        let flops = 2.0 * Double(batch) * Double(n) * Double(n)
        return ProbeResult(
            kind: kind, value: flops / seconds / teraflop,
            detail: "\(batch)×\(n) · \(bits)-bit \(n)×\(n), group \(group), best of 3 × 10"
        )
    }

    /// Fused attention over a 4k sequence — the shape of a long prompt or a video DiT's
    /// full attention.
    static func attention() -> ProbeResult {
        let heads = 24, length = 4096, dim = 128
        let q = MLXRandom.normal([1, heads, length, dim]).asType(.float16)
        let k = MLXRandom.normal([1, heads, length, dim]).asType(.float16)
        let v = MLXRandom.normal([1, heads, length, dim]).asType(.float16)
        eval(q, k, v)
        let scale = 1 / Float(dim).squareRoot()
        let seconds = time(iterations: 5) {
            MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        }
        let flops = 4.0 * Double(heads) * Double(length) * Double(length) * Double(dim)
        return ProbeResult(
            kind: .attention, value: flops / seconds / teraflop,
            detail: "\(heads) heads × \(length) tokens × \(dim), fp16, best of 3 × 5"
        )
    }

    /// A 3×3×3 convolution over a five-frame video tensor: what the SeedVR2 and Wan VAEs
    /// spend their time on — 97% of the owner's ComfyUI upscale.
    static func conv3d() -> ProbeResult {
        let frames = 5, size = 128, channels = 256
        let input = MLXRandom.normal([1, frames, size, size, channels]).asType(.float16)
        let weight = MLXRandom.normal([channels, 3, 3, 3, channels]).asType(.float16)
        eval(input, weight)
        let seconds = time(iterations: 3) { MLX.conv3d(input, weight, padding: 1) }
        let outputs = Double(frames * size * size * channels)
        let flops = 2.0 * outputs * 27 * Double(channels)
        return ProbeResult(
            kind: .conv3d, value: flops / seconds / teraflop,
            detail: "\(frames)×\(size)×\(size)×\(channels), 3×3×3 kernel, fp16, best of 3 × 3"
        )
    }

    /// Allocate in 512 MB steps up to three quarters of what's free (never past the GPU's
    /// usable ceiling), then release it and check it actually went back — the thing MLXUI
    /// never did. Deliberately cautious: it measures headroom, it doesn't hunt for the limit.
    static func headroom(hardware: HardwareProfile) -> ProbeResult {
        let step = 512 * 1024 * 1024
        let target = min(hardware.usableMemoryBytes, Int64(Double(hardware.availableMemoryBytes) * 0.75))
        let before = Memory.activeMemory + Memory.cacheMemory
        var held: [MLXArray] = []
        var reached = 0
        while Int64(reached + step) <= target {
            let block = MLXArray.ones([step / 4], dtype: .float32)
            eval(block)
            held.append(block)
            reached += step
        }
        held.removeAll()
        SeedVR2Residency.releaseBuffers(restoringLimitTo: Memory.cacheLimit)
        let after = Memory.activeMemory + Memory.cacheMemory
        let returned = after <= before + step
        return ProbeResult(
            kind: .memoryHeadroom, value: Double(reached) / 1_073_741_824,
            detail: String(
                format: "allocated %.1f GB of %.1f GB free; %@ to the system afterwards",
                Double(reached) / 1_073_741_824, hardware.availableMemoryGB,
                returned ? "all returned" : "not all returned"
            )
        )
    }

    /// Write then read back 512 MB on the library volume with the file cache bypassed, so
    /// the figure is the disk's and not memory's.
    static func diskRead(library: URL) throws -> ProbeResult {
        try FileManager.default.createDirectory(at: library, withIntermediateDirectories: true)
        let url = library.appendingPathComponent(".locallab-probe-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: url) }
        let blockSize = 8 * 1024 * 1024, blocks = 64
        var block = [UInt8](repeating: 0, count: blockSize)
        for index in block.indices { block[index] = UInt8(truncatingIfNeeded: index &* 31) }

        let writeFD = open(url.path, O_CREAT | O_WRONLY | O_TRUNC, 0o600)
        guard writeFD >= 0 else { throw ProbeError.disk("couldn't create a test file in \(library.path)") }
        _ = fcntl(writeFD, F_NOCACHE, 1)
        let writeStart = Date()
        for _ in 0 ..< blocks {
            let written = block.withUnsafeBytes { write(writeFD, $0.baseAddress, blockSize) }
            guard written == blockSize else {
                close(writeFD)
                throw ProbeError.disk("couldn't write the test file")
            }
        }
        fsync(writeFD)
        close(writeFD)
        let writeSeconds = Date().timeIntervalSince(writeStart)

        let readFD = open(url.path, O_RDONLY)
        guard readFD >= 0 else { throw ProbeError.disk("couldn't reopen the test file") }
        _ = fcntl(readFD, F_NOCACHE, 1)
        let readStart = Date()
        var total = 0
        while true {
            let count = block.withUnsafeMutableBytes { read(readFD, $0.baseAddress, blockSize) }
            if count <= 0 { break }
            total += count
        }
        close(readFD)
        let readSeconds = Date().timeIntervalSince(readStart)
        let bytes = Double(total)
        return ProbeResult(
            kind: .diskRead, value: bytes / max(readSeconds, 1e-6) / gigabyte,
            detail: String(
                // No path: a report must never carry one (plan §5.15).
                format: "512 MB uncached on the library's volume; write %.1f GB/s",
                Double(blockSize * blocks) / max(writeSeconds, 1e-6) / gigabyte
            )
        )
    }

    public enum ProbeError: Error, LocalizedError {
        case disk(String)
        public var errorDescription: String? {
            switch self {
            case .disk(let detail): "Disk probe: \(detail)"
            }
        }
    }
}
