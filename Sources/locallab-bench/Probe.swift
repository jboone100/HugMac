import Foundation
import LocalLabCore
import LocalLabMLX

/// `locallab-bench --probe [--no-save] [--swift]` — run the first-run probes (plan §5.14),
/// print them, and save them beside `calibration.json` as the app would. `--swift` also
/// prints the report as a `ProbeReport` literal, for adding a reference Mac.
///
/// `locallab-bench --profile` — print the machine profile the app would show, from what is
/// already measured.
enum ProbeCommand {
    static func run(arguments: [String]) async throws {
        let hardware = HardwareProfile.detect()
        let store = ModelStore()
        print("Probing \(hardware.machineKey.displayName) — library on \(store.baseDirectory.path)")
        let report = try await ProbeSuite.run(hardware: hardware, library: store.baseDirectory) { progress in
            if let running = progress.running {
                print("  [\(progress.completed + 1)/\(progress.total)] \(running.title)…")
            }
        }
        print("")
        for result in report.results {
            print(String(format: "  %-28@ %8.2f %-6@  %@", result.kind.title as NSString, result.value,
                         result.kind.unit as NSString, result.detail as NSString))
        }
        print(String(format: "\n  %.1f s in total", report.durationSeconds))

        if !arguments.contains("--no-save") {
            let url = ProbeStore.defaultURL()
            var probes = ProbeStore.load(from: url)
            probes.record(report)
            try probes.save(to: url)
            print("  saved to \(url.path)")
        }
        if arguments.contains("--swift") {
            print("\n" + swiftLiteral(report))
        }
    }

    static func profile() {
        let hardware = HardwareProfile.detect()
        let calibration = CalibrationStore.load()
        let profile = MachineProfile.resolve(
            hardware: hardware, calibration: calibration,
            installedUpscalers: SeedVR2Variant.installed(in: ModelStore())
        )
        print(hardware.machineKey.displayName)
        print(String(format: "  %.0f GB/s bandwidth · %.1f GB usable by the GPU · %.1f GB free now",
                     hardware.memoryBandwidthGBps, hardware.usableMemoryGB, hardware.availableMemoryGB))
        print("  " + profile.timing.summary)
        for capability in profile.capabilities {
            print("\n  \(capability.title) — \(capability.example)")
            print("    [\(capability.status)] \(capability.headline)")
            print("    \(capability.detail)")
        }
    }

    static func swiftLiteral(_ report: ProbeReport) -> String {
        let key = report.machine
        let results = report.results.map {
            "            ProbeResult(kind: .\($0.kind.rawValue), value: \(String(format: "%.3f", $0.value)), detail: \"\($0.detail)\"),"
        }.joined(separator: "\n")
        return """
        ProbeReport(
            machine: MachineKey(chipName: "\(key.chipName)", gpuCores: \(key.gpuCores.map(String.init) ?? "nil"), memoryGB: \(key.memoryGB)),
            date: ISO8601DateFormatter().date(from: "\(ISO8601DateFormatter().string(from: report.date))") ?? .distantPast,
            macOSVersion: "\(report.macOSVersion)",
            durationSeconds: \(String(format: "%.1f", report.durationSeconds)),
            results: [
        \(results)
            ]
        )
        """
    }
}
