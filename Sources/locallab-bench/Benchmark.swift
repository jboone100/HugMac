import Foundation
import LocalLabCore
import LocalLabMLX

// The acceptance benchmark from the plan (§6.4): upscale a clip with every setting chosen by
// the resolver, and report per-phase peak memory and time against the ComfyUI baseline.
//
//   locallab-bench <video> [--frames N] [--short-side 768] [--quality balanced] [--model DIR]
//
// `--frames` truncates the clip, so chunking, blending and audio can be verified in minutes
// before committing to a multi-hour full run.

struct Arguments {
    var input: URL
    var frames: Int?
    var shortSide = 768
    var quality: QualityPreset = .balanced
    var modelDirectory: URL
    var outputDirectory: URL

    static func parse() throws -> Arguments {
        var raw = Array(CommandLine.arguments.dropFirst())
        guard let first = raw.first, !first.hasPrefix("--") else {
            throw Failure("usage: locallab-bench <video> [--frames N] [--short-side 768] [--quality fast|balanced|best] [--model DIR]")
        }
        let store = ModelStore()
        var arguments = Arguments(
            input: URL(fileURLWithPath: first),
            modelDirectory: store.directory(forRepo: SeedVR2Variant.threeBInt8.hfRepo),
            outputDirectory: store.outputsDirectory
        )
        raw.removeFirst()
        while let flag = raw.first {
            raw.removeFirst()
            func value() throws -> String {
                guard let next = raw.first else { throw Failure("\(flag) needs a value") }
                raw.removeFirst()
                return next
            }
            switch flag {
            case "--frames": arguments.frames = Int(try value())
            case "--short-side": arguments.shortSide = Int(try value()) ?? 768
            case "--quality":
                arguments.quality = QualityPreset(rawValue: try value()) ?? .balanced
            case "--model": arguments.modelDirectory = URL(fileURLWithPath: try value())
            case "--out": arguments.outputDirectory = URL(fileURLWithPath: try value())
            default: throw Failure("unknown flag \(flag)")
            }
        }
        return arguments
    }
}

struct Failure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

func format(_ seconds: Double) -> String {
    if seconds < 90 { return String(format: "%.1f s", seconds) }
    let total = Int(seconds.rounded())
    return String(format: "%d h %02d m %02d s", total / 3600, (total % 3600) / 60, total % 60)
}

func gb(_ bytes: Int64) -> String {
    String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
}

@main
struct Benchmark {
    static func main() async {
        // A one-hour run piped to a file would otherwise report nothing until it flushed.
        setvbuf(stdout, nil, _IONBF, 0)
        do { try await run() } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }

    static func run() async throws {
        LegacyMigration.moveApplicationSupportIfNeeded()
        let raw = Array(CommandLine.arguments.dropFirst())
        if raw.first == "--install" {
            try await InstallCommand.run(arguments: Array(raw.dropFirst()))
            return
        }
        if raw.first == "--trim" {
            try await Compare.trim(arguments: Array(raw.dropFirst()))
            return
        }
        if raw.first == "--diff" {
            try await Compare.diff(arguments: Array(raw.dropFirst()))
            return
        }
        if raw.first == "--chat" {
            try await ChatCommand.run(arguments: Array(raw.dropFirst()))
            return
        }
        if raw.first == "--probe" {
            try await ProbeCommand.run(arguments: Array(raw.dropFirst()))
            return
        }
        if raw.first == "--profile" {
            ProbeCommand.profile()
            return
        }
        if raw.first == "--extract" {
            try await Extract.run(arguments: Array(raw.dropFirst()))
            return
        }
        let arguments = try Arguments.parse()
        let hardware = HardwareProfile.detect()

        print("── Machine ─────────────────────────────────────────────")
        print("  \(hardware.chipName), \(hardware.gpuCoreCount.map { "\($0) GPU cores" } ?? "GPU cores unknown")")
        print(String(format: "  RAM %.0f GB · available now %.1f GB · single-model ceiling %.1f GB",
                     hardware.totalMemoryGB, hardware.availableMemoryGB, hardware.usableMemoryGB))
        print("  plannable budget \(gb(hardware.plannableMemoryBytes()))")

        var probed = try await VideoIO.probe(arguments.input)
        let fullFrameCount = probed.frameCount
        if let limit = arguments.frames, limit < probed.frameCount {
            probed = VideoMedia(
                url: probed.url, width: probed.width, height: probed.height, fps: probed.fps,
                frameCount: limit, hasAudio: probed.hasAudio, hasAlpha: probed.hasAlpha
            )
        }
        print("\n── Input ───────────────────────────────────────────────")
        print("  \(arguments.input.lastPathComponent): \(probed.width)×\(probed.height), "
            + "\(probed.frameCount) frames @ \(String(format: "%.2f", probed.fps)) fps"
            + (probed.hasAudio ? ", audio" : ", no audio")
            + (probed.hasAlpha ? ", alpha" : ""))

        var store = CalibrationStore.load()
        let alreadyMeasured = store.bytesPerPeakUnit(
            engineID: SeedVR2Resolver.mlxEngineID, phase: "vae-decode", chipName: hardware.chipName
        ) != nil
        store.merge(CalibrationStore.comfyUIBaseline(chipName: hardware.chipName))
        print("  calibration: " + (alreadyMeasured
            ? "using this Mac's measurements"
            : "no prior run here — using the built-in fallbacks"))
        let resolver = SeedVR2Resolver(hardware: hardware, calibration: store)
        let plan = try resolver.plan(
            source: .video(probed),
            target: .shortSide(arguments.shortSide),
            quality: arguments.quality,
            installedVariants: [.threeBInt8]
        )

        print("\n── Plan (every value chosen by the resolver) ───────────")
        print("  output    \(plan.outputWidth)×\(plan.outputHeight)  (working \(plan.paddedWidth)×\(plan.paddedHeight))")
        print("  chunks    \(plan.chunks.count) × up to \(plan.chunks.map(\.length).max() ?? 0) frames, overlap \(plan.temporalOverlap)")
        print("  encode    \(plan.encodeTiling.map { "\($0.tileSize) px tiles / \($0.overlap) px overlap" } ?? "untiled")")
        print("  decode    \(plan.decodeTiling.map { "\($0.tileSize) px tiles / \($0.overlap) px overlap" } ?? "untiled")")
        for phase in plan.phases {
            print(String(format: "  %-11@ predicted peak %@", phase.phase as NSString, gb(phase.peakBytes)))
        }
        print("  predicted overall peak \(gb(plan.peakBytes))")
        print("\n  why:")
        for reason in plan.reasons {
            print("    \(reason.setting) = \(reason.value)  [\(reason.provenance.rawValue)]")
            print("      \(reason.because)")
        }

        let components = SeedVR2Components(directory: arguments.modelDirectory)
        try components.verify()
        print("\n  components verified in \(arguments.modelDirectory.path)")

        let residency = SeedVR2Residency()
        // Working files go in their own folder under jobs/, never beside results, and a fresh
        // one each run: a benchmark measures a cold run, and stale checkpoints would skip phases.
        let scratch = ModelStore().jobsDirectory
            .appendingPathComponent("bench-\(UUID().uuidString)", isDirectory: true)

        // Never overwrite an earlier result, and say in the name when only part of the clip
        // was run — a 5-frame smoke test once overwrote the full 243-frame benchmark output
        // because both were called `Diner_0-1344x768.mp4`.
        var stem = arguments.input.deletingPathExtension().lastPathComponent
            + "-\(plan.outputWidth)x\(plan.outputHeight)"
        if let frames = arguments.frames, frames < fullFrameCount {
            stem += "-first\(probed.frameCount)f"
        }
        let output = ModelStore.uniqueOutputURL(
            in: arguments.outputDirectory, stem: stem, extension: "mp4"
        )

        print("\n── Running ─────────────────────────────────────────────")
        let started = Date()
        let lastPhase = Mutex("")
        let result = try await SeedVR2Engine.upscale(
            video: probed, plan: plan, components: components, residency: residency,
            machine: hardware.machineKey, outputURL: output, scratch: scratch, resume: false
        ) { progress in
            let elapsed = Date().timeIntervalSince(started)
            let changed = lastPhase.exchange(progress.phase) != progress.phase
            if changed || progress.unitsTotal <= 1 || progress.unitsDone % 5 == 0 {
                print(String(
                    format: "  [%6.1fs] %-11@ %3d/%-3d  %.0f%%",
                    elapsed, progress.phase as NSString, progress.unitsDone,
                    progress.unitsTotal, progress.fraction * 100
                ))
            }
        }

        // Succeeded: the checkpoints have done their job.
        try? FileManager.default.removeItem(at: scratch)

        print("\n── Measured ────────────────────────────────────────────")
        for sample in result.measurements.samples {
            let predicted = plan.phases.first { $0.phase == sample.phase }
            print(String(
                format: "  %-11@ %-10@ peak %-9@ (predicted %-9@)  %@ bytes/peak-unit",
                sample.phase as NSString, format(sample.seconds) as NSString,
                gb(sample.peakBytes) as NSString,
                gb(predicted?.peakBytes ?? 0) as NSString,
                String(format: "%.0f", sample.bytesPerPeakUnit ?? 0) as NSString
            ))
        }
        print("  overall peak \(gb(result.measurements.peakBytes))"
            + "  (predicted \(gb(plan.peakBytes)))")
        print("  total        \(format(result.measurements.totalSeconds))")
        print("  output       \(output.path)")
        print("               \(result.video.width)×\(result.video.height), "
            + "\(result.video.frameCount) frames"
            + (result.video.hasAudio ? ", audio kept" : ", no audio"))

        // Compare against the reference run, scaled to the number of frames actually done.
        let baselineFrames = 243.0
        let scale = Double(probed.frameCount) / baselineFrames
        let baselineTotal = CalibrationStore.comfyUIBaselineTotalSeconds * scale
        let baselineSteady = (3795.88 + 794.32 + CalibrationStore.comfyUIBaselineSteadyDecodeSeconds + 13.46) * scale
        print("\n── Against the ComfyUI baseline (PyTorch/MPS, same Mac) ")
        print("  baseline, scaled to \(probed.frameCount) frames: \(format(baselineTotal))"
            + "  (excluding its stalls: \(format(baselineSteady)))")
        let ratio = baselineSteady / max(result.measurements.totalSeconds, 0.001)
        print(String(format: "  LocalLab is %.2f× %@ than the baseline's steady-state time",
                     ratio >= 1 ? ratio : 1 / ratio, ratio >= 1 ? "faster" : "slower"))
        print("  baseline peak 7.57 GB vs measured \(gb(result.measurements.peakBytes))")

        // Feed the measurements back so the next plan on this Mac is predicted, not guessed.
        store.merge(result.measurements.samples)
        do {
            try store.save()
            print("\n  recorded \(result.measurements.samples.count) measurements to "
                + CalibrationStore.defaultURL().path)
        } catch {
            print("\n  could not save calibration: \(error)")
        }
    }
}

/// Minimal mutex so the progress closure can track the previous phase without data races.
final class Mutex<Value>: @unchecked Sendable {
    private var value: Value
    private let lock = NSLock()
    init(_ value: Value) { self.value = value }
    func exchange(_ newValue: Value) -> Value {
        lock.lock(); defer { lock.unlock() }
        let old = value
        value = newValue
        return old
    }
}
