import AVFoundation
import CoreGraphics
import CoreImage
import Foundation

/// Reading and writing video, so no stage ever holds a whole clip in memory and no user ever
/// wires up "get video components".
///
/// In the owner's ComfyUI graph this was four nodes (Load Video → Get Video Components →
/// Join Image with Alpha → the encoder, with audio and fps threaded past by hand). Here it is
/// one type: probe, read frames in chunks, write frames, remux the original audio.
public enum VideoIO {

    public enum IOError: Error, CustomStringConvertible {
        case noVideoTrack(URL)
        case readerFailed(String)
        case writerFailed(String)
        case frameConversionFailed(index: Int)
        case muxFailed(String)

        public var description: String {
            switch self {
            case .noVideoTrack(let url):
                return "'\(url.lastPathComponent)' has no video track."
            case .readerFailed(let detail):
                return "Couldn't read the video — \(detail)."
            case .writerFailed(let detail):
                return "Couldn't write the video — \(detail)."
            case .frameConversionFailed(let index):
                return "Couldn't convert frame \(index) to an image."
            case .muxFailed(let detail):
                return "Couldn't attach the audio track — \(detail)."
            }
        }
    }

    // MARK: - Probe

    /// Everything the planner needs, without decoding a frame.
    public static func probe(_ url: URL) async throws -> VideoMedia {
        let asset = AVURLAsset(url: url)
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = videoTracks.first else { throw IOError.noVideoTrack(url) }

        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let nominalRate = try await track.load(.nominalFrameRate)
        let duration = try await asset.load(.duration)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        let formats = try await track.load(.formatDescriptions)

        // Respect rotation: a portrait clip reports a landscape naturalSize plus a transform.
        let oriented = naturalSize.applying(transform)
        let width = Int(abs(oriented.width).rounded())
        let height = Int(abs(oriented.height).rounded())

        let fps = nominalRate > 0 ? Double(nominalRate) : 24
        let seconds = duration.seconds.isFinite ? duration.seconds : 0
        let frameCount = max(Int((seconds * fps).rounded()), 1)

        return VideoMedia(
            url: url,
            width: width,
            height: height,
            fps: fps,
            frameCount: frameCount,
            hasAudio: !audioTracks.isEmpty,
            hasAlpha: formats.contains { hasAlphaChannel($0) }
        )
    }

    /// Whether a track's format carries alpha — so the app can upscale an alpha channel when
    /// one exists instead of asking the user to enable it.
    static func hasAlphaChannel(_ description: CMFormatDescription) -> Bool {
        let extensions = CMFormatDescriptionGetExtensions(description) as NSDictionary?
        if let depth = extensions?["Depth"] as? Int, depth == 32 { return true }
        if let alpha = extensions?[kCMFormatDescriptionExtension_ContainsAlphaChannel as String] as? Bool {
            return alpha
        }
        let subType = CMFormatDescriptionGetMediaSubType(description)
        // ProRes 4444 and 4444 XQ carry alpha.
        return subType == kCMVideoCodecType_AppleProRes4444
            || subType == kCMVideoCodecType_AppleProRes4444XQ
    }

    // MARK: - Reading

    /// Decodes a half-open frame range to CGImages. One reader per call, with a `timeRange`,
    /// so a chunked job never decodes more than the chunk it is about to process.
    public static func readFrames(
        from url: URL,
        startIndex: Int,
        count: Int,
        fps: Double
    ) async throws -> [CGImage] {
        guard count > 0 else { return [] }
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw IOError.noVideoTrack(url)
        }

        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw IOError.readerFailed(error.localizedDescription)
        }

        let timescale: CMTimeScale = 600
        let start = CMTime(seconds: Double(startIndex) / fps, preferredTimescale: timescale)
        // A half-frame of slack so the last requested frame is inside the range.
        let duration = CMTime(seconds: (Double(count) + 0.5) / fps, preferredTimescale: timescale)
        reader.timeRange = CMTimeRange(start: start, duration: duration)

        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        )
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw IOError.readerFailed("the decoder rejected this track")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw IOError.readerFailed(reader.error?.localizedDescription ?? "unknown reason")
        }
        defer { reader.cancelReading() }

        // `copyNextSampleBuffer` blocks until the decoder hands over a frame, and a decoder
        // that never does — seen when several decode at once — blocks it forever. A watchdog
        // cancels the reader after `stallSeconds` without a frame, which returns the call.
        let watchdog = ReaderWatchdog(reader: reader)
        let watch = Task.detached {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(stallSeconds))
                if Task.isCancelled { return }
                if watchdog.stalled() { return }
            }
        }
        defer { watch.cancel() }

        let context = CIContext(options: [.useSoftwareRenderer: false])
        var frames: [CGImage] = []
        frames.reserveCapacity(count)
        while frames.count < count, let sample = output.copyNextSampleBuffer() {
            watchdog.progressed()
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            let image = CIImage(cvPixelBuffer: buffer)
            guard let cgImage = context.createCGImage(image, from: image.extent) else {
                throw IOError.frameConversionFailed(index: startIndex + frames.count)
            }
            frames.append(cgImage)
        }
        if watchdog.didStall {
            throw IOError.readerFailed("the video decoder stopped responding")
        }
        if let error = reader.error {
            throw IOError.readerFailed(error.localizedDescription)
        }
        return frames
    }

    /// Seconds without a decoded frame before reading is abandoned.
    static let stallSeconds: Double = 20

    /// The first frame, for a thumbnail — through `AVAssetImageGenerator`, whose async API
    /// can't block the way a reader's `copyNextSampleBuffer` can.
    public static func thumbnail(of url: URL, maxDimension: CGFloat = 640) async throws -> CGImage {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: maxDimension, height: maxDimension)
        return try await generator.image(at: .zero).image
    }

    /// Notices when a reader stops producing frames, and cancels it.
    final class ReaderWatchdog: @unchecked Sendable {
        private let reader: AVAssetReader
        private let lock = NSLock()
        private var frames = 0
        private var seen = 0
        private(set) var didStall = false

        init(reader: AVAssetReader) { self.reader = reader }

        func progressed() {
            lock.withLock { frames += 1 }
        }

        /// Called every `stallSeconds`: true (and the reader cancelled) if no frame arrived
        /// since the last call.
        func stalled() -> Bool {
            let stuck = lock.withLock { () -> Bool in
                defer { seen = frames }
                return frames == seen
            }
            guard stuck, reader.status == .reading else { return false }
            lock.withLock { didStall = true }
            reader.cancelReading()
            return true
        }
    }

    // MARK: - Writing

    /// Appends frames to a video file. Frames arrive chunk by chunk, so a long job never
    /// materialises the whole output.
    public final class Writer {
        private let writer: AVAssetWriter
        private let input: AVAssetWriterInput
        private let adaptor: AVAssetWriterInputPixelBufferAdaptor
        private let fps: Double
        private var frameIndex = 0

        public init(url: URL, width: Int, height: Int, fps: Double, codec: AVVideoCodecType = .hevc) throws {
            if FileManager.default.fileExists(atPath: url.path) {
                try? FileManager.default.removeItem(at: url)
            }
            do {
                writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
            } catch {
                throw IOError.writerFailed(error.localizedDescription)
            }
            self.fps = fps
            input = AVAssetWriterInput(
                mediaType: .video,
                outputSettings: [
                    AVVideoCodecKey: codec,
                    AVVideoWidthKey: width,
                    AVVideoHeightKey: height,
                ]
            )
            input.expectsMediaDataInRealTime = false
            adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: width,
                    kCVPixelBufferHeightKey as String: height,
                ]
            )
            guard writer.canAdd(input) else {
                throw IOError.writerFailed("the encoder rejected these settings")
            }
            writer.add(input)
            guard writer.startWriting() else {
                throw IOError.writerFailed(writer.error?.localizedDescription ?? "unknown reason")
            }
            writer.startSession(atSourceTime: .zero)
        }

        public func append(_ image: CGImage) throws {
            while !input.isReadyForMoreMediaData {
                // The encoder is behind; give it a moment rather than dropping frames.
                Thread.sleep(forTimeInterval: 0.002)
            }
            guard let pool = adaptor.pixelBufferPool else {
                throw IOError.writerFailed("no pixel buffer pool")
            }
            var maybeBuffer: CVPixelBuffer?
            guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &maybeBuffer) == kCVReturnSuccess,
                  let buffer = maybeBuffer else {
                throw IOError.writerFailed("couldn't allocate a pixel buffer")
            }
            CVPixelBufferLockBaseAddress(buffer, [])
            defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
            guard let base = CVPixelBufferGetBaseAddress(buffer) else {
                throw IOError.writerFailed("pixel buffer had no base address")
            }
            let context = CGContext(
                data: base,
                width: CVPixelBufferGetWidth(buffer),
                height: CVPixelBufferGetHeight(buffer),
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue
            )
            guard let context else { throw IOError.writerFailed("couldn't draw the frame") }
            context.draw(image, in: CGRect(
                x: 0, y: 0,
                width: CVPixelBufferGetWidth(buffer),
                height: CVPixelBufferGetHeight(buffer)
            ))
            let time = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(fps.rounded()))
            guard adaptor.append(buffer, withPresentationTime: time) else {
                throw IOError.writerFailed(writer.error?.localizedDescription ?? "the encoder refused a frame")
            }
            frameIndex += 1
        }

        public func finish() async throws {
            input.markAsFinished()
            await writer.finishWriting()
            if writer.status == .failed {
                throw IOError.writerFailed(writer.error?.localizedDescription ?? "unknown reason")
            }
        }
    }

    // MARK: - Audio passthrough

    /// Copies `audioFrom`'s first audio track onto a video-only file, writing `to`.
    ///
    /// Upscaling transforms frames only, so the audio is never re-encoded — it is remuxed
    /// untouched. H3 emits stereo audio with its video, and the owner's clips carry AAC, so
    /// silently dropping it would be a bug.
    public static func attachAudio(video: URL, audioFrom source: URL, to output: URL) async throws {
        let composition = AVMutableComposition()
        let videoAsset = AVURLAsset(url: video)
        let sourceAsset = AVURLAsset(url: source)

        guard let videoTrack = try await videoAsset.loadTracks(withMediaType: .video).first,
              let compositionVideo = composition.addMutableTrack(
                  withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid
              ) else {
            throw IOError.muxFailed("no video track to carry over")
        }
        let videoDuration = try await videoAsset.load(.duration)
        try compositionVideo.insertTimeRange(
            CMTimeRange(start: .zero, duration: videoDuration), of: videoTrack, at: .zero
        )

        if let audioTrack = try await sourceAsset.loadTracks(withMediaType: .audio).first,
           let compositionAudio = composition.addMutableTrack(
               withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid
           ) {
            let sourceDuration = try await sourceAsset.load(.duration)
            // Never let a slightly longer audio track stretch the clip.
            let span = CMTimeMinimum(videoDuration, sourceDuration)
            try compositionAudio.insertTimeRange(
                CMTimeRange(start: .zero, duration: span), of: audioTrack, at: .zero
            )
        }

        guard let export = AVAssetExportSession(
            asset: composition, presetName: AVAssetExportPresetPassthrough
        ) else {
            throw IOError.muxFailed("couldn't start a passthrough export")
        }
        if FileManager.default.fileExists(atPath: output.path) {
            try? FileManager.default.removeItem(at: output)
        }
        try await export.export(to: output, as: .mp4)
    }
}

// MARK: - Joining segments

public extension VideoIO {
    /// Join video files end to end without re-encoding.
    ///
    /// The upscaler writes each chunk's frames to its own segment so a job interrupted
    /// during decode — the longest phase — resumes at the chunk it was on instead of
    /// decoding everything again. The segments are then joined here, by passthrough.
    static func concatenate(_ segments: [URL], to output: URL) async throws {
        guard !segments.isEmpty else { throw IOError.muxFailed("no segments to join") }
        let composition = AVMutableComposition()
        guard let track = composition.addMutableTrack(
            withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid
        ) else { throw IOError.muxFailed("couldn't create a video track") }

        var cursor = CMTime.zero
        for segment in segments {
            let asset = AVURLAsset(url: segment)
            guard let source = try await asset.loadTracks(withMediaType: .video).first else {
                throw IOError.noVideoTrack(segment)
            }
            let duration = try await asset.load(.duration)
            try track.insertTimeRange(CMTimeRange(start: .zero, duration: duration), of: source, at: cursor)
            cursor = cursor + duration
        }

        guard let export = AVAssetExportSession(
            asset: composition, presetName: AVAssetExportPresetPassthrough
        ) else { throw IOError.muxFailed("couldn't start a passthrough export") }
        if FileManager.default.fileExists(atPath: output.path) {
            try? FileManager.default.removeItem(at: output)
        }
        try await export.export(to: output, as: .mp4)
    }
}
