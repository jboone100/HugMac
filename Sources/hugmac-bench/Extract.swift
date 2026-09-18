import CoreGraphics
import Foundation
import HugMacCore
import ImageIO
import UniformTypeIdentifiers

/// `hugmac-bench --extract <video> <frameIndex> <out.png>`
///
/// Verification aid: pull one decoded frame out of a clip so source and output can be
/// compared by eye. The frame-alignment bug (three causal warm-up frames) produced a video
/// that played but was shifted, which no amount of unit testing would have caught.
enum Extract {
    static func run(arguments: [String]) async throws {
        guard arguments.count >= 3,
              let index = Int(arguments[1]) else {
            throw Failure("usage: hugmac-bench --extract <video> <frameIndex> <out.png>")
        }
        let input = URL(fileURLWithPath: arguments[0])
        let output = URL(fileURLWithPath: arguments[2])
        let probed = try await VideoIO.probe(input)
        let frames = try await VideoIO.readFrames(
            from: input, startIndex: index, count: 1, fps: probed.fps
        )
        guard let frame = frames.first else {
            throw Failure("no frame at index \(index) of \(probed.frameCount)")
        }
        guard let destination = CGImageDestinationCreateWithURL(
            output as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw Failure("couldn't create \(output.path)")
        }
        CGImageDestinationAddImage(destination, frame, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw Failure("couldn't write \(output.path)")
        }
        print("wrote \(output.path) — \(frame.width)×\(frame.height) "
            + "(frame \(index) of \(probed.frameCount))")
    }
}
