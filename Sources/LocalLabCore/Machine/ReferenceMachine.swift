import Foundation

/// A Mac whose measurements ship with the app, so a Mac that has never run an engine still
/// gets a time estimate — scaled to it, and labelled as an estimate (plan §5.13, layer 2).
///
/// Grown by review, never by collection: a new entry comes from a benchmark someone ran and
/// shared on purpose (an exported profile, or an opt-in report — plan §5.15).
public struct ReferenceMachine: Sendable, Equatable {
    public let machine: MachineKey
    /// The probe results measured on this Mac with the current suite, when there are any.
    /// Without them, timings are scaled by spec sheet instead.
    public let probes: ProbeReport?
    public let samples: [CalibrationSample]
    /// Where the numbers came from.
    public let source: String

    public init(machine: MachineKey, probes: ProbeReport?, samples: [CalibrationSample], source: String) {
        self.machine = machine
        self.probes = probes
        self.samples = samples
        self.source = source
    }

    public static let all: [ReferenceMachine] = [m2Max30Core32GB]

    // MARK: - Apple M2 Max, 30-core GPU, 32 GB

    /// The owner's Mac, where the SeedVR2 engine was built and first measured (2026-09-17).
    /// Only complete runs are included: a resumed run skips work it still reports, and would
    /// make this Mac look faster than it is.
    public static let m2Max30Core32GB: ReferenceMachine = {
        let machine = MachineKey(chipName: "Apple M2 Max", gpuCores: 30, memoryGB: 32)
        func sample(
            _ phase: String, work: Double, peak: Double, seconds: Double,
            peakBytes: Int64, weights: Int64, note: String
        ) -> CalibrationSample {
            CalibrationSample(
                engineID: SeedVR2Resolver.mlxEngineID, phase: phase, workUnits: work, peakUnits: peak,
                seconds: seconds, peakBytes: peakBytes, weightBytes: weights, machine: machine, note: note
            )
        }
        let clip = "SeedVR2-3B-int8, 1 chunk(s) of up to 5 frames"
        let image = "SeedVR2-3B-int8, 1 chunk(s) of up to 1 frames"
        return ReferenceMachine(
            machine: machine,
            probes: m2Max30Core32GBProbes,
            samples: [
                sample("vae-encode", work: 5898240, peak: 2949120, seconds: 11.63, peakBytes: 11705913338, weights: 501000000, note: clip),
                sample("dit", work: 8064, peak: 8064, seconds: 12.16, peakBytes: 9921987906, weights: 4220000000, note: clip),
                sample("vae-decode", work: 14155776, peak: 737280, seconds: 54.12, peakBytes: 8168743862, weights: 501000000, note: clip),
                sample("vae-encode", work: 5898240, peak: 2949120, seconds: 11.24, peakBytes: 11705913338, weights: 501000000, note: clip),
                sample("dit", work: 8064, peak: 8064, seconds: 11.62, peakBytes: 9922000318, weights: 4220000000, note: clip),
                sample("vae-decode", work: 14155776, peak: 737280, seconds: 53.71, peakBytes: 8168743862, weights: 501000000, note: clip),
                sample("vae-encode", work: 5898240, peak: 2949120, seconds: 12.41, peakBytes: 11705913338, weights: 501000000, note: clip),
                sample("dit", work: 8064, peak: 8064, seconds: 12.09, peakBytes: 9850994282, weights: 4220000000, note: clip),
                sample("vae-decode", work: 14155776, peak: 737280, seconds: 55.63, peakBytes: 8168743862, weights: 501000000, note: clip),
                sample("vae-encode", work: 1032192, peak: 1032192, seconds: 2.28, peakBytes: 5263855618, weights: 501000000, note: image),
                sample("dit", work: 4032, peak: 4032, seconds: 4.69, peakBytes: 6271099590, weights: 4220000000, note: image),
                sample("vae-decode", work: 1032192, peak: 1032192, seconds: 5.32, peakBytes: 9493275574, weights: 501000000, note: image),
            ],
            source: "LocalLab SeedVR2 engine runs on the owner's Mac, 2026-09-17"
        )
    }()

    /// `locallab-bench --probe` on the same Mac, 2026-09-18. GPU probes repeated within ~2%
    /// across three runs; disk and headroom vary with what else is running, and neither is
    /// used for scaling time.
    static let m2Max30Core32GBProbes: ProbeReport? = ProbeReport(
        machine: MachineKey(chipName: "Apple M2 Max", gpuCores: 30, memoryGB: 32),
        date: ISO8601DateFormatter().date(from: "2026-09-18T21:21:58Z") ?? .distantPast,
        macOSVersion: "26.6.2",
        durationSeconds: 6.2,
        results: [
            ProbeResult(kind: .memoryBandwidth, value: 321.495, detail: "sum over 1 GiB fp32, best of 3 × 10"),
            ProbeResult(kind: .matmulF16, value: 8.809, detail: "4096×4096 · 4096×4096 fp16, best of 3 × 10"),
            ProbeResult(kind: .quantizedMatmul4, value: 7.421, detail: "2048×4096 · 4-bit 4096×4096, group 64, best of 3 × 10"),
            ProbeResult(kind: .quantizedMatmul8, value: 7.608, detail: "2048×4096 · 8-bit 4096×4096, group 64, best of 3 × 10"),
            ProbeResult(kind: .attention, value: 7.122, detail: "24 heads × 4096 tokens × 128, fp16, best of 3 × 5"),
            ProbeResult(kind: .conv3d, value: 6.838, detail: "5×128×128×256, 3×3×3 kernel, fp16, best of 3 × 3"),
            ProbeResult(kind: .memoryHeadroom, value: 9.000, detail: "allocated 9.0 GB of 12.4 GB free; all returned to the system afterwards"),
            ProbeResult(kind: .diskRead, value: 2.809, detail: "512 MB uncached on the library's volume; write 2.4 GB/s"),
        ]
    )
}
