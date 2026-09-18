import CoreGraphics
import Foundation
import LocalLabCore

/// `--trim <in> <frames> <out>` — the first N frames of a clip, for short real runs.
/// `--diff <a> <b>` — per-frame mean and max absolute pixel difference (0–255).
enum Compare {
    static func trim(arguments: [String]) async throws {
        guard arguments.count >= 3, let count = Int(arguments[1]) else {
            throw Failure("usage: locallab-bench --trim <in.mp4> <frames> <out.mp4>")
        }
        let input = URL(fileURLWithPath: arguments[0]), output = URL(fileURLWithPath: arguments[2])
        let probed = try await VideoIO.probe(input)
        let frames = try await VideoIO.readFrames(from: input, startIndex: 0, count: count, fps: probed.fps)
        let writer = try VideoIO.Writer(url: output, width: probed.width, height: probed.height, fps: probed.fps)
        for frame in frames { try writer.append(frame) }
        try await writer.finish()
        print("wrote \(output.path) — \(frames.count) frames, \(probed.width)×\(probed.height)")
    }

    static func diff(arguments: [String]) async throws {
        guard arguments.count >= 2 else { throw Failure("usage: locallab-bench --diff <a.mp4> <b.mp4>") }
        let a = URL(fileURLWithPath: arguments[0]), b = URL(fileURLWithPath: arguments[1])
        let pa = try await VideoIO.probe(a), pb = try await VideoIO.probe(b)
        let fa = try await VideoIO.readFrames(from: a, startIndex: 0, count: pa.frameCount + 8, fps: pa.fps)
        let fb = try await VideoIO.readFrames(from: b, startIndex: 0, count: pb.frameCount + 8, fps: pb.fps)
        print("frames: \(fa.count) vs \(fb.count) · size \(pa.width)×\(pa.height) vs \(pb.width)×\(pb.height)")
        var worst = 0.0
        for index in 0 ..< min(fa.count, fb.count) {
            guard let x = rgba(fa[index]), let y = rgba(fb[index]), x.count == y.count else {
                print("  frame \(index): size mismatch"); continue
            }
            var sum = 0, peak = 0
            for i in stride(from: 0, to: x.count, by: 4) {
                for c in 0 ..< 3 {
                    let d = abs(Int(x[i + c]) - Int(y[i + c]))
                    sum += d
                    peak = max(peak, d)
                }
            }
            let mean = Double(sum) / Double(x.count / 4 * 3)
            worst = max(worst, mean)
            print(String(format: "  frame %3d  mean %.3f  max %3d", index, mean, peak))
        }
        print(String(format: "worst per-frame mean difference: %.3f", worst))
    }

    static func rgba(_ image: CGImage) -> [UInt8]? {
        let width = image.width, height = image.height
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let ok = bytes.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            ) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        return ok ? bytes : nil
    }
}
