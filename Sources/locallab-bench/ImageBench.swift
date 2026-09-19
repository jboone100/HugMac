import CoreGraphics
import Foundation
import ImageIO
import LocalLabCore
import LocalLabMLX
import UniformTypeIdentifiers

/// `locallab-bench --image <file> [--width N | --fit] [--target-scale 2] [--plan-only]`
///
/// Upscale one image with the real engine and record what each phase actually used, so a
/// size that was extrapolated gets a measurement near it. `--fit` picks the largest output
/// the planner says fits in the memory free now. Afterwards it re-plans `--target-scale`×
/// (default 2×) with the new measurement and reports what that now needs.
enum ImageBench {
    static func run(arguments: [String]) async throws {
        guard let path = arguments.first else {
            throw Failure("usage: locallab-bench --image <file> [--width N | --fit] [--target-scale 2]")
        }
        let url = URL(fileURLWithPath: path)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw Failure("couldn't read \(path)")
        }
        func value(_ flag: String) -> String? {
            arguments.firstIndex(of: flag).flatMap { $0 + 1 < arguments.count ? arguments[$0 + 1] : nil }
        }
        let targetScale = Int(value("--target-scale") ?? "2") ?? 2
        let hardware = HardwareProfile.detect()
        var calibration = CalibrationStore.load()
        let store = LibraryLocation.resolve().store
        let resolverSource = SeedVR2Resolver.Source.image(width: image.width, height: image.height)
        print("\(url.lastPathComponent): \(image.width)×\(image.height) on \(hardware.machineKey.displayName), "
            + String(format: "%.1f GB free now", hardware.availableMemoryGB))

        func plan(width: Int, on hw: HardwareProfile, with cal: CalibrationStore) throws -> SeedVR2Plan {
            let height = Int((Double(width) * Double(image.height) / Double(image.width)).rounded())
            return try SeedVR2Resolver(hardware: hw, calibration: cal).plan(
                source: resolverSource, target: .exact(width: width, height: height),
                quality: .balanced, installedVariants: [.threeBInt8]
            )
        }

        // The target as it stands, against everything the GPU may use.
        let emptied = hardware.withAvailableMemory(Int64(Double(hardware.usableMemoryBytes) / 0.85))
        let targetWidth = image.width * targetScale
        func describeTarget(_ label: String, _ cal: CalibrationStore) {
            do {
                let p = try plan(width: targetWidth, on: emptied, with: cal)
                let dit = p.phases.first { $0.phase == "dit" }
                print("  \(label): \(targetScale)× (\(p.outputWidth)×\(p.outputHeight)) needs \(gb(p.peakBytes)) "
                    + "— transformer \(gb(dit?.peakBytes ?? 0)) for \(Int(dit?.peakUnits ?? 0)) patches"
                    + (dit?.peakIsMeasured == true ? " (from measurements)" : " (built-in estimate)"))
            } catch StageError.insufficientMemory(let required, _) {
                print("  \(label): \(targetScale)× needs \(String(format: "%.1f", required)) GB — more than this Mac's GPU can use")
            } catch {
                print("  \(label): \(error)")
            }
        }
        describeTarget("before", calibration)
        do {
            let now = try plan(width: targetWidth, on: hardware, with: calibration)
            print("  now: \(targetScale)× fits in the memory free now — peak \(gb(now.peakBytes))")
        } catch StageError.insufficientMemory(let required, let available) {
            print(String(format: "  now: %d× needs %.1f GB at its smallest; %.1f GB can be used now", targetScale, required, available))
        }
        if arguments.contains("--plan-only") { return }

        // Choose the size to run.
        let chosen: SeedVR2Plan
        if let width = value("--width").flatMap(Int.init) {
            chosen = try plan(width: width, on: hardware, with: calibration)
        } else {
            var width = targetWidth
            var found: SeedVR2Plan?
            while width > image.width, found == nil {
                found = try? plan(width: width, on: hardware, with: calibration)
                if found == nil { width -= 32 }
            }
            guard let found else { throw Failure("no enlargement of this image fits in the memory free now") }
            chosen = found
        }
        let dit = chosen.phases.first { $0.phase == "dit" }
        print("\n  running \(chosen.outputWidth)×\(chosen.outputHeight) — \(Int(dit?.peakUnits ?? 0)) transformer patches, "
            + "predicted peak \(gb(chosen.peakBytes))")
        for phase in chosen.phases {
            print(String(format: "    %-11@ predicted %@", phase.phase as NSString, gb(phase.peakBytes)))
        }

        let components = SeedVR2Components(directory: store.directory(forRepo: SeedVR2Variant.threeBInt8.hfRepo))
        try components.verify()
        let result = try await SeedVR2Engine.upscale(
            image: image, plan: chosen, components: components, residency: SeedVR2Residency(),
            machine: hardware.machineKey
        )

        print("\n  measured:")
        for sample in result.measurements.samples {
            let predicted = chosen.phases.first { $0.phase == sample.phase }?.peakBytes ?? 0
            print(String(format: "    %-11@ peak %@ (predicted %@) · %@ bytes per patch · %.1f s",
                         sample.phase as NSString, gb(sample.peakBytes), gb(predicted),
                         String(format: "%.0f", sample.bytesPerPeakUnit ?? 0), sample.seconds))
        }

        let output = ModelStore.uniqueOutputURL(
            in: store.outputsDirectory,
            stem: url.deletingPathExtension().lastPathComponent + "-\(chosen.outputWidth)x\(chosen.outputHeight)",
            extension: "png"
        )
        if let destination = CGImageDestinationCreateWithURL(output as CFURL, UTType.png.identifier as CFString, 1, nil) {
            CGImageDestinationAddImage(destination, result.image, nil)
            CGImageDestinationFinalize(destination)
            print("  output \(output.path)")
        }

        calibration.merge(result.measurements.samples)
        try calibration.save()
        print("  recorded \(result.measurements.samples.count) measurements\n")
        describeTarget("after", calibration)
    }
}
