import CoreGraphics
import Foundation

/// A typed payload passed between stages.
///
/// Ported from MLXUI's `Core/Media.swift` with one addition that the video work requires:
/// `video`. Video is **file-backed**, never a frame array — 15 s of 24 fps 1344×768 RGB is
/// ~1.1 GB raw, and 2560×1440 is ~4 GB. Stages stream frames through `VideoIO` in chunks.
public enum Media: Sendable {
    case text(String)
    case audio(AudioBuffer)
    case image(ImageMedia)
    case video(VideoMedia)
    case embedding([[Float]])

    public var kind: MediaKind {
        switch self {
        case .text:      .text
        case .audio:     .audio
        case .image:     .image
        case .video:     .video
        case .embedding: .embedding
        }
    }
}

/// The shape of a `Media` value, independent of its payload.
public enum MediaKind: String, Sendable, CaseIterable, Codable {
    case text, audio, image, video, embedding
}

/// Mono PCM in [-1, 1]. The sample rate travels with the samples so downstream code
/// never has to guess.
public struct AudioBuffer: Sendable, Equatable {
    public var samples: [Float]
    public var sampleRate: Int

    public init(samples: [Float], sampleRate: Int) {
        self.samples = samples
        self.sampleRate = sampleRate
    }
}

/// `CGImage` is thread-safe in practice but not marked `Sendable`, so we vouch for it
/// here rather than leaking `@unchecked` across the codebase.
public struct ImageMedia: @unchecked Sendable {
    public let cgImage: CGImage

    public init(cgImage: CGImage) {
        self.cgImage = cgImage
    }
}

/// A video on disk, plus the facts a stage needs to plan its work without opening it.
///
/// `hasAudio` matters for upscaling: SeedVR2 transforms frames only, so the audio track is
/// demuxed once and remuxed untouched (the owner's source clips carry AAC audio, and H3
/// emits stereo audio natively).
public struct VideoMedia: Sendable, Equatable, Codable {
    public let url: URL
    public let width: Int
    public let height: Int
    public let fps: Double
    public let frameCount: Int
    public let hasAudio: Bool
    /// True when the source carries a meaningful alpha channel — the app detects this
    /// rather than making the user wire up an alpha input.
    public let hasAlpha: Bool

    public init(
        url: URL,
        width: Int,
        height: Int,
        fps: Double,
        frameCount: Int,
        hasAudio: Bool,
        hasAlpha: Bool = false
    ) {
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.frameCount = frameCount
        self.hasAudio = hasAudio
        self.hasAlpha = hasAlpha
    }

    public var durationSeconds: Double {
        fps > 0 ? Double(frameCount) / fps : 0
    }

    public var pixelsPerFrame: Int { width * height }
}
