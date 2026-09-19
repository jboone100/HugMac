import CoreGraphics
import Foundation
import ImageIO
import LocalLabCore
import LocalLabMLX
import UniformTypeIdentifiers

/// `locallab-bench --generate "<prompt>" [--size 1024x1024] [--steps 4] [--seed 42] [--out file.png]
/// [--record]`
///
/// Generate one image with FLUX.1 schnell from the default library and print what each phase
/// took. `--record` saves the measurements, as a job in the app would.
enum GenerateCommand {
    static func run(arguments: [String]) async throws {
        guard let prompt = arguments.first, !prompt.hasPrefix("--") else {
            throw Failure(#"usage: locallab-bench --generate "<prompt>" [--size 1024x1024] [--steps 4] [--seed 42] [--out file.png] [--record]"#)
        }
        func value(_ flag: String) -> String? {
            arguments.firstIndex(of: flag).flatMap { $0 + 1 < arguments.count ? arguments[$0 + 1] : nil }
        }
        let size = (value("--size") ?? "1024x1024").split(separator: "x").compactMap { Int($0) }
        guard size.count == 2 else { throw Failure("--size is WIDTHxHEIGHT") }
        let spec = ImageModelCatalog.fluxSchnell4bit
        let request = ImageRequest(
            prompt: prompt, width: size[0], height: size[1],
            steps: Int(value("--steps") ?? "") ?? spec.defaultSteps,
            seed: UInt64(value("--seed") ?? "") ?? 42
        )
        let store = LibraryLocation.resolve().store
        let directory = store.directory(forRepo: spec.repo)
        let output = URL(fileURLWithPath: value("--out") ?? "generated.png")
        let hardware = HardwareProfile.detect()
        print("\(spec.displayName) on \(hardware.machineKey.displayName), "
            + String(format: "%.1f GB free now", hardware.availableMemoryGB))

        let last = LastPhase()
        let clock = Date()
        let result = try await FluxEngine.generate(request, spec: spec, directory: directory) { progress in
            if last.changed(to: progress.phase + "\(progress.unitsDone)") {
                print(String(format: "  %6.1f s  %3.0f%%  %@ %@", Date().timeIntervalSince(clock), progress.fraction * 100, progress.phase,
                             progress.unitsTotal > 0 ? "\(progress.unitsDone)/\(progress.unitsTotal)" : ""))
            }
        }
        for sample in result.samples {
            print(String(format: "  %-8@ %6.1f s   peak %@", sample.phase as NSString, sample.seconds,
                         gb(sample.peakBytes) as NSString))
        }
        print(String(format: "total %.1f s, peak %@", result.seconds, gb(result.peakBytes)))

        guard let destination = CGImageDestinationCreateWithURL(output as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw Failure("couldn't create \(output.path)")
        }
        CGImageDestinationAddImage(destination, result.image.cgImage, nil)
        guard CGImageDestinationFinalize(destination) else { throw Failure("couldn't write \(output.path)") }
        print("wrote \(output.path)")

        if arguments.contains("--record") {
            var calibration = CalibrationStore.load()
            calibration.merge(result.samples)
            try calibration.save()
            print("recorded \(result.samples.count) measurements")
        }
    }

    final class LastPhase: @unchecked Sendable {
        private let lock = NSLock()
        private var key = ""
        func changed(to newKey: String) -> Bool {
            lock.withLock {
                defer { key = newKey }
                return newKey != key
            }
        }
    }
}
