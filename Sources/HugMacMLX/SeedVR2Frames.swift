import CoreGraphics
import Foundation
import HugMacCore
import MLX

/// Pixels in and out of MLX, for one frame or many.
///
/// MLXUI's engine had the single-frame versions of these inline; the tensor layout it used
/// is already `[B, C, T, H, W]` with `T = 1`, so feeding it real video is a matter of
/// stacking frames on the existing temporal axis.
enum SeedVR2Frames {

    /// Frames → `[1, 3, T, H, W]` bfloat16 in [-1, 1]. Every frame must be the same size.
    static func tensor(from frames: [CGImage]) throws -> MLXArray {
        guard let first = frames.first else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "no frames to encode")
        }
        let width = first.width, height = first.height
        var values = [Float](repeating: 0, count: frames.count * height * width * 3)
        var offset = 0
        for (index, frame) in frames.enumerated() {
            guard frame.width == width, frame.height == height else {
                throw StageError.engineFailure(
                    stage: "SeedVR2",
                    detail: "frame \(index) is \(frame.width)×\(frame.height), expected \(width)×\(height)"
                )
            }
            guard let bytes = bgraBytes(frame) else {
                throw StageError.engineFailure(
                    stage: "SeedVR2", detail: "couldn't read pixels from frame \(index)"
                )
            }
            for pixel in 0 ..< width * height {
                values[offset + pixel * 3 + 0] = Float(bytes[pixel * 4 + 0]) / 127.5 - 1.0
                values[offset + pixel * 3 + 1] = Float(bytes[pixel * 4 + 1]) / 127.5 - 1.0
                values[offset + pixel * 3 + 2] = Float(bytes[pixel * 4 + 2]) / 127.5 - 1.0
            }
            offset += width * height * 3
        }
        // [T, H, W, 3] → [3, T, H, W] → [1, 3, T, H, W]
        return MLXArray(values, [frames.count, height, width, 3])
            .transposed(3, 0, 1, 2)
            .expandedDimensions(axis: 0)
            .asType(.bfloat16)
    }

    /// One frame of a `[1, 3, T, H, W]` tensor as `[3, H, W]`.
    static func frameSlice(_ tensor: MLXArray, at index: Int) -> MLXArray {
        tensor[0, 0..., index, 0..., 0...]
    }

    /// `[3, H, W]` in [-1, 1] → CGImage.
    static func image(from frame: MLXArray) throws -> CGImage {
        let hwc = frame.transposed(1, 2, 0)
        let clipped = MLX.clip(hwc * 127.5 + 127.5, min: MLXArray(0.0), max: MLXArray(255.0))
        let pixels = clipped.asType(.uint8)
        eval(pixels)
        let height = pixels.dim(0), width = pixels.dim(1)
        let bytes: [UInt8] = pixels.asArray(UInt8.self)
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "couldn't wrap output pixels")
        }
        guard let image = CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: width * 3,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ) else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "couldn't build the output image")
        }
        return image
    }

    // MARK: - CoreGraphics helpers

    static func bgraBytes(_ image: CGImage) -> [UInt8]? {
        let width = image.width, height = image.height
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let success: Bool = bytes.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let context = CGContext(
                      data: base, width: width, height: height,
                      bitsPerComponent: 8, bytesPerRow: width * 4,
                      space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                          | CGImageAlphaInfo.noneSkipLast.rawValue
                  ) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        return success ? bytes : nil
    }

    /// Bicubic resize to an explicit size. SeedVR2 conditions on an already-upsampled image,
    /// and the target may not be an integer multiple (768 short side from 384 is 2×, but
    /// 1440 from 768 is not).
    static func resize(_ image: CGImage, width: Int, height: Int) throws -> CGImage {
        if image.width == width && image.height == height { return image }
        guard let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                | CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "couldn't create a resize context")
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let output = context.makeImage() else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "resize produced no image")
        }
        return output
    }

    /// Pad to the working size (a multiple of 16) by stretching the edge row and column,
    /// rather than padding with black — a hard border is something the VAE would try to
    /// reproduce, and it would bleed into the real pixels next to it.
    ///
    /// The source sits at the **top-left**, which is what `crop(_:toWidth:height:)` undoes.
    static func pad(_ image: CGImage, toWidth width: Int, height: Int) throws -> CGImage {
        if image.width == width && image.height == height { return image }
        guard width >= image.width, height >= image.height else {
            throw StageError.engineFailure(
                stage: "SeedVR2", detail: "padding target is smaller than the frame"
            )
        }
        guard let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                | CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "couldn't create a padding context")
        }
        context.interpolationQuality = .high

        // CoreGraphics draws from the bottom-left; `top` is where the source begins.
        let top = height - image.height
        let padRight = width - image.width
        context.draw(image, in: CGRect(x: 0, y: top, width: image.width, height: image.height))

        // `CGImage.cropping` works in image space (origin top-left).
        if padRight > 0,
           let rightEdge = image.cropping(to: CGRect(
               x: image.width - 1, y: 0, width: 1, height: image.height)) {
            context.draw(rightEdge, in: CGRect(
                x: image.width, y: top, width: padRight, height: image.height))
        }
        if top > 0,
           let bottomEdge = image.cropping(to: CGRect(
               x: 0, y: image.height - 1, width: image.width, height: 1)) {
            context.draw(bottomEdge, in: CGRect(x: 0, y: 0, width: image.width, height: top))
        }
        if padRight > 0, top > 0,
           let corner = image.cropping(to: CGRect(
               x: image.width - 1, y: image.height - 1, width: 1, height: 1)) {
            context.draw(corner, in: CGRect(x: image.width, y: 0, width: padRight, height: top))
        }

        guard let output = context.makeImage() else {
            throw StageError.engineFailure(stage: "SeedVR2", detail: "padding produced no image")
        }
        return output
    }

    /// Crop the working size back to what the user asked for.
    static func crop(_ image: CGImage, toWidth width: Int, height: Int) -> CGImage {
        if image.width == width && image.height == height { return image }
        let rect = CGRect(x: 0, y: image.height - height, width: width, height: height)
        return image.cropping(to: rect) ?? image
    }
}
