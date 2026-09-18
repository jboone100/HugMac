import Foundation

/// A measurement from a real run on this machine: how long a phase took and how much memory
/// it peaked at, for a known amount of work.
///
/// This is what turns HugMac's estimates from guesses into predictions, and it is local-only
/// — no telemetry leaves the machine.
public struct CalibrationSample: Sendable, Codable, Equatable {
    /// Which engine produced this. Distinct engines are never mixed: the owner's ComfyUI
    /// baseline ran PyTorch on MPS, so it cannot be used to predict the MLX engine.
    public let engineID: String
    public let phase: String
    /// Work done, in the phase's natural unit — pixel-frames for VAE phases, latent tokens
    /// for the transformer. Drives **time**, and scales with the whole job.
    public let workUnits: Double
    /// The size of the largest single piece of work: one chunk's tile area × its frames for
    /// the VAE phases, one chunk's tokens for the transformer.
    ///
    /// Separate from `workUnits` because **peak memory does not scale with the job**. A
    /// 61-chunk clip and a 1-chunk clip peak identically at the same chunk size; what drives
    /// peak is how much is live at once. Predicting peak from total work would say a long
    /// clip needs more memory than a short one, which is false.
    public let peakUnits: Double
    public let seconds: Double
    public let peakBytes: Int64
    /// Weights resident during the phase, excluded when deriving bytes-per-unit so the
    /// activation cost can be measured on its own.
    public let weightBytes: Int64
    public let chipName: String
    public let note: String

    public init(
        engineID: String,
        phase: String,
        workUnits: Double,
        peakUnits: Double = 0,
        seconds: Double,
        peakBytes: Int64,
        weightBytes: Int64 = 0,
        chipName: String,
        note: String = ""
    ) {
        self.engineID = engineID
        self.phase = phase
        self.workUnits = workUnits
        self.peakUnits = peakUnits
        self.seconds = seconds
        self.peakBytes = peakBytes
        self.weightBytes = weightBytes
        self.chipName = chipName
        self.note = note
    }

    public var secondsPerUnit: Double { workUnits > 0 ? seconds / workUnits : 0 }

    /// Activation bytes per peak unit, with the phase's weights removed.
    public var bytesPerPeakUnit: Double? {
        guard peakUnits > 0 else { return nil }
        let activations = Double(peakBytes - weightBytes)
        guard activations > 0 else { return nil }
        return activations / peakUnits
    }
}

/// A time estimate that is honest about where it came from. An unmeasured engine says so
/// rather than inventing a number.
public enum TimeEstimate: Sendable, Equatable, Codable {
    case unknown
    case measured(seconds: Double)
    case extrapolated(seconds: Double, fromChip: String)

    public var seconds: Double? {
        switch self {
        case .unknown: nil
        case .measured(let s): s
        case .extrapolated(let s, _): s
        }
    }
}

/// Per-machine calibration history.
public struct CalibrationStore: Sendable {
    public internal(set) var samples: [CalibrationSample]

    public init(samples: [CalibrationSample] = []) {
        self.samples = samples
    }

    public mutating func record(_ sample: CalibrationSample) {
        samples.append(sample)
    }

    /// Seconds per work unit for an engine's phase on this chip, newest samples weighted by
    /// simple mean. Returns nil when this engine has never run here.
    public func secondsPerUnit(engineID: String, phase: String, chipName: String) -> Double? {
        let matches = samples.filter {
            $0.engineID == engineID && $0.phase == phase && $0.chipName == chipName && $0.workUnits > 0
        }
        guard !matches.isEmpty else { return nil }
        return matches.map(\.secondsPerUnit).reduce(0, +) / Double(matches.count)
    }

    /// Measured activation bytes per peak unit, or nil when this engine's phase has never
    /// run here. This is what replaces the fitted constants in the memory model.
    public func bytesPerPeakUnit(engineID: String, phase: String, chipName: String) -> Double? {
        let values = samples.compactMap { sample -> Double? in
            guard sample.engineID == engineID, sample.phase == phase,
                  sample.chipName == chipName else { return nil }
            return sample.bytesPerPeakUnit
        }
        guard !values.isEmpty else { return nil }
        // The maximum, not the mean: under-predicting peak means a job that swaps or dies,
        // while over-predicting only costs a little tiling.
        return values.max()
    }

    /// Predicted activation bytes (peak less weights) for a piece of work of `peakUnits`,
    /// from measurements of this engine's phase on this chip.
    ///
    /// Interpolated between measured sizes rather than scaled from one per-unit rate: the
    /// transformer has a large fixed cost, so its cost *per token* differs by chunk size, and a
    /// single global rate let one small measurement dictate every large plan.
    ///
    /// - Between two measured sizes: linear interpolation of the measured activations.
    /// - Smaller than anything measured: the smallest measurement — an upper bound, since cost
    ///   grows with size.
    /// - Larger than anything measured: the largest measurement's per-unit rate, flagged, so
    ///   the caller can refuse to predict below its own coefficients.
    public func predictedActivation(
        engineID: String, phase: String, chipName: String, peakUnits: Double
    ) -> (bytes: Double, extrapolatedAbove: Bool)? {
        var bySize: [Double: Double] = [:]
        for sample in samples where sample.engineID == engineID && sample.phase == phase
            && sample.chipName == chipName && sample.peakUnits > 0 {
            let activation = Double(sample.peakBytes - sample.weightBytes)
            guard activation > 0 else { continue }
            bySize[sample.peakUnits] = max(bySize[sample.peakUnits] ?? 0, activation)
        }
        let points = bySize.sorted { $0.key < $1.key }
        guard !points.isEmpty, peakUnits > 0 else { return nil }
        if let exact = points.first(where: { abs($0.key - peakUnits) / peakUnits < 0.05 }) {
            return (exact.value, false)
        }
        let below = points.last { $0.key < peakUnits }
        let above = points.first { $0.key > peakUnits }
        switch (below, above) {
        case let (low?, high?):
            let t = (peakUnits - low.key) / (high.key - low.key)
            return (low.value + t * (high.value - low.value), false)
        case let (nil, high?):
            return (high.value, false)
        case let (low?, nil):
            return (low.value / low.key * peakUnits, true)
        case (nil, nil):
            return nil
        }
    }

    public func estimate(
        engineID: String,
        phase: String,
        chipName: String,
        workUnits: Double
    ) -> TimeEstimate {
        if let perUnit = secondsPerUnit(engineID: engineID, phase: phase, chipName: chipName) {
            return .measured(seconds: perUnit * workUnits)
        }
        return .unknown
    }

    // MARK: - The ComfyUI baseline

    /// The owner's ComfyUI run of 2026-09-06, from the SeedVR2 node's debug log:
    /// `Diner_0.mp4`, 243 frames, 672×384 → 1344×768, SeedVR2 3B fp16, temporal batch 5,
    /// overlap 1 (61 batches), VAE tiled 512 px / 128 px overlap (8 tiles per batch).
    ///
    /// Recorded under its own engine id because it is **PyTorch on MPS**, not MLX. It is the
    /// benchmark HugMac's engine is measured against (§6.4 of the plan), never a prediction
    /// of it.
    public static let comfyUIBaselineEngineID = "comfyui-seedvr2-videoupscaler-mps"

    public static func comfyUIBaseline(chipName: String = "Apple M2 Max") -> [CalibrationSample] {
        // 61 batches × 5 frames at 1344×768 output.
        let pixelFrames = 61.0 * 5.0 * 1344.0 * 768.0
        // 61 batches × 2 latent frames × (96/2) × (168/2) patch tokens.
        let latentTokens = 61.0 * 2.0 * 48.0 * 84.0
        return [
            CalibrationSample(
                engineID: comfyUIBaselineEngineID, phase: "vae-encode",
                workUnits: pixelFrames, seconds: 3795.88, peakBytes: 1_320_702_443,
                chipName: chipName, note: "tiled 512/128; 60–71 s per 5-frame batch, steady"
            ),
            CalibrationSample(
                engineID: comfyUIBaselineEngineID, phase: "dit",
                workUnits: latentTokens, seconds: 794.32, peakBytes: 8_128_200_540,
                chipName: chipName, note: "3B fp16; 13–17 s per batch, steady"
            ),
            CalibrationSample(
                engineID: comfyUIBaselineEngineID, phase: "vae-decode",
                workUnits: pixelFrames, seconds: 18980.52, peakBytes: 2_845_415_014,
                chipName: chipName,
                note: "tiled 512/128; median 160 s per batch, 17 of 61 over 2× median (max 1304 s)"
            ),
        ]
    }

    /// Steady-state decode total, excluding the 17 outlier batches — the fairer figure to
    /// beat, since the outliers look like the Mac sleeping rather than the model's cost.
    public static let comfyUIBaselineSteadyDecodeSeconds: Double = 61 * 160

    public static let comfyUIBaselineTotalSeconds: Double = 3795.88 + 794.32 + 18980.52 + 13.46
}

// MARK: - Persistence

public extension CalibrationStore {
    /// Where a machine's measurements live. Local only — this is the app's own record of how
    /// fast and how heavy each phase is *here*, and it never leaves the machine.
    static func defaultURL(appSupport: URL? = nil) -> URL {
        let base = appSupport ?? FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base
            .appendingPathComponent("HugMac", isDirectory: true)
            .appendingPathComponent("calibration.json")
    }

    /// Load measurements, or an empty store when this machine has never run a job.
    static func load(from url: URL? = nil) -> CalibrationStore {
        let location = url ?? defaultURL()
        guard let data = try? Data(contentsOf: location),
              let samples = try? JSONDecoder().decode([CalibrationSample].self, from: data) else {
            return CalibrationStore()
        }
        return CalibrationStore(samples: samples)
    }

    func save(to url: URL? = nil) throws {
        let location = url ?? Self.defaultURL()
        try FileManager.default.createDirectory(
            at: location.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(samples).write(to: location, options: .atomic)
    }

    /// Add a finished run's measurements, keeping the most recent few per (engine, phase,
    /// chip) so a long history doesn't dilute a recent change.
    mutating func merge(_ newSamples: [CalibrationSample], keepPerKey: Int = 5) {
        samples.append(contentsOf: newSamples)
        var kept: [String: Int] = [:]
        var result: [CalibrationSample] = []
        for sample in samples.reversed() {
            let key = "\(sample.engineID)|\(sample.phase)|\(sample.chipName)"
            let count = kept[key, default: 0]
            guard count < keepPerKey else { continue }
            kept[key] = count + 1
            result.append(sample)
        }
        samples = result.reversed()
    }
}
