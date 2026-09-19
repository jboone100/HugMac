import Foundation

/// How well one chat model runs on this Mac at one context length, with the arithmetic.
public struct ChatFit: Sendable, Equatable {
    public enum Grade: Sendable, Equatable {
        /// Fits with headroom in memory free now, and fast enough to read comfortably.
        case green
        /// Runs, with a caveat stated in `because`.
        case yellow(because: String)
        /// Won't run here; `because` says why.
        case red(because: String)

        public var isRed: Bool { if case .red = self { true } else { false } }
    }

    public let spec: ChatModelSpec
    public let context: Int
    public let peakBytes: Int64
    public let tokensPerSecond: Double
    /// Where the speed came from — measured here, or estimated (and from what).
    public let speed: TimeEstimate
    public let grade: Grade
    /// "9.2 GB weights + 2.1 GB context (16k) + 1.0 GB runtime = 12.3 GB · ~24 tok/s"
    public let arithmetic: String

    public var peakGB: Double { Double(peakBytes) / 1_073_741_824 }
    /// Fits in what is free right now, not only in what the GPU could use.
    public let fitsNow: Bool
}

/// Chooses the chat model (plan §5.12): the best installed model that runs well here, or
/// the one the user picked. Never downloads — it names a better model instead.
public struct ChatModelPicker: Sendable {
    /// Keeps the app's first name: it labels measurements already stored in
    /// `calibration.json`, and renaming it would orphan them.
    public static let engineID = "hugmac-llm"
    /// Below this a reply reads as painful; Smart Fit never picks one (plan §5.2).
    public static let usableTokensPerSecond = 8.0
    /// At or above this, reading keeps up comfortably.
    public static let comfortableTokensPerSecond = 15.0
    public static let contextLadder = [32_768, 16_384, 8_192, 4_096]
    /// The runtime, the tokenizer and the prompt's activations, beyond weights and cache.
    public static let runtimeBytes: Int64 = 1_073_741_824

    public let hardware: HardwareProfile
    public let calibration: CalibrationStore

    public init(hardware: HardwareProfile, calibration: CalibrationStore) {
        self.hardware = hardware
        self.calibration = calibration
    }

    // MARK: - One model

    /// Grade a model at a context length, or at the longest one that keeps it green.
    /// `vision`: graded for a conversation with images — the model loaded with its vision
    /// half, which generates more slowly.
    public func fit(_ spec: ChatModelSpec, context requested: Int? = nil, vision: Bool = false) -> ChatFit {
        let context = requested ?? defaultContext(for: spec)
        let kv = spec.kvCacheBytes(context: context)
        let peak = spec.weightBytes + kv + Self.runtimeBytes
        let (tokensPerSecond, speed) = self.speed(of: spec, vision: vision)

        let usable = hardware.usableMemoryBytes
        let plannable = hardware.plannableMemoryBytes()
        let fitsNow = peak <= plannable
        let grade: ChatFit.Grade
        if peak > usable {
            grade = .red(because: String(format: "needs %.1f GB; this Mac's GPU can use %.1f GB",
                                         Double(peak) / 1_073_741_824, hardware.usableMemoryGB))
        } else if tokensPerSecond < Self.usableTokensPerSecond {
            grade = .yellow(because: String(format: "about %.0f tokens a second — slow to read", tokensPerSecond))
        } else if !fitsNow {
            grade = .yellow(because: String(format: "needs %.1f GB and %.1f GB is free — close some apps",
                                            Double(peak) / 1_073_741_824, Double(plannable) / 1_073_741_824))
        } else if Double(peak) > Double(usable) * 0.7 || tokensPerSecond < Self.comfortableTokensPerSecond {
            grade = .yellow(because: tokensPerSecond < Self.comfortableTokensPerSecond
                ? String(format: "about %.0f tokens a second", tokensPerSecond)
                : "fits, with little room for anything else")
        } else {
            grade = .green
        }

        let arithmetic = String(
            format: "%.1f GB weights + %.1f GB context (%@) + %.1f GB runtime = %.1f GB · ~%.0f tok/s",
            spec.weightGB, Double(kv) / 1_073_741_824, Self.contextLabel(context),
            Double(Self.runtimeBytes) / 1_073_741_824, Double(peak) / 1_073_741_824, tokensPerSecond
        )
        return ChatFit(spec: spec, context: context, peakBytes: peak, tokensPerSecond: tokensPerSecond,
                       speed: speed, grade: grade, arithmetic: arithmetic, fitsNow: fitsNow)
    }

    /// The longest context whose cache keeps the model within 70% of what the GPU can use
    /// and within what's free now; 4k when even that doesn't.
    public func defaultContext(for spec: ChatModelSpec) -> Int {
        let ceiling = min(Int64(Double(hardware.usableMemoryBytes) * 0.7), hardware.plannableMemoryBytes())
        for context in Self.contextLadder where context <= spec.maxContext {
            if spec.weightBytes + spec.kvCacheBytes(context: context) + Self.runtimeBytes <= ceiling {
                return context
            }
        }
        return Self.contextLadder.last ?? 4_096
    }

    /// Tokens a second when generating. Measured here when this Mac has run a chat model —
    /// rates are kept per byte read, so one model's measurement prices another — otherwise
    /// estimated from memory bandwidth: the probe's measured figure when there is one, the
    /// spec sheet's otherwise.
    /// Vision is kept apart: the runtime's vision load of Qwen3.5 9B generated at ~16 tok/s
    /// where its text load did ~38 (M2 Max, 2026-09-18), so a vision reply measured here
    /// prices vision replies only, and an unmeasured one is taken as 45% of the text speed.
    public static let visionSpeedFactor = 0.45

    public func speed(of spec: ChatModelSpec, vision: Bool = false) -> (tokensPerSecond: Double, source: TimeEstimate) {
        if vision {
            let gbPerToken = Self.gbReadPerToken(spec)
            let measured = calibration.estimate(
                engineID: Self.engineID, phase: Self.visionPhase, machine: hardware.machineKey, workUnits: gbPerToken
            )
            if let seconds = measured.seconds, seconds > 0, case .measured = measured {
                return (1 / seconds, measured)
            }
            let text = speed(of: spec)
            let seconds = 1 / (text.tokensPerSecond * Self.visionSpeedFactor)
            return (1 / seconds, .extrapolated(seconds: seconds, fromChip: hardware.chipName, basis: .specs))
        }
        let gbPerToken = Self.gbReadPerToken(spec)
        let estimate = calibration.estimate(
            engineID: Self.engineID, phase: "decode", machine: hardware.machineKey, workUnits: gbPerToken
        )
        if let seconds = estimate.seconds, seconds > 0 {
            return (1 / seconds, estimate)
        }
        // Token generation streams the active weights once per token. MLX reaches roughly
        // three quarters of the probe's measured bandwidth on 4-bit weights, and about 60% of
        // the spec sheet's — starting values, replaced by the first measured chat here.
        //
        // Plus a fixed ~1.5 ms a token — sampling, detokenizing, kernel launches — which
        // bandwidth alone misses: without it a 0.5B model "runs" at 900 tokens a second.
        let overhead = 0.0015
        let seconds: Double
        let basis: ScalingBasis
        if let measured = calibration.probes.latest(for: hardware.machineKey)?.value(.memoryBandwidth) {
            seconds = gbPerToken / (measured * 0.75) + overhead
            basis = .probes
        } else {
            seconds = gbPerToken / (hardware.memoryBandwidthGBps * 0.6) + overhead
            basis = .specs
        }
        return (1 / seconds, .extrapolated(seconds: seconds, fromChip: hardware.chipName, basis: basis))
    }

    public static let visionPhase = "decode-vision"

    /// Gigabytes read per generated token. Expert weights are gathered, not streamed: a
    /// mixture of experts reaches roughly half a dense model's bandwidth, priced as extra
    /// bytes so a dense model's measured rate still applies.
    public static func gbReadPerToken(_ spec: ChatModelSpec) -> Double {
        spec.activeWeightBytes / 1e9 / (spec.isMixtureOfExperts ? 0.55 : 1)
    }

    // MARK: - Choosing

    /// What **Smart Fit** loads — LocalLab's name for choosing from the hardware profile: among installed models that aren't red and aren't painfully
    /// slow, prefer one that runs *well* (green), then one that fits in memory free now,
    /// then the highest quality, then the fastest. Nil when nothing installed will do.
    ///
    /// Green first, not quality first: with 19 GB free on a 32 GB Mac, quality first picks
    /// a 27B squeezed to a 4k context at ~15 tokens a second over a 9B at 32k and ~40.
    public func smartFit(installed: Set<String>) -> ChatFit? {
        smartFit(candidates: ChatCatalog.models.filter { installed.contains($0.repo) })
    }

    /// Smart Fit over any installed models — curated or estimated from their repos.
    ///
    /// `vision`: the conversation has images, so only models that can see them qualify.
    public func smartFit(candidates specs: [ChatModelSpec], vision: Bool = false) -> ChatFit? {
        let candidates = specs
            .filter { !vision || $0.seesImages }
            .map { fit($0, vision: vision) }
            .filter { !$0.grade.isRed && $0.tokensPerSecond >= Self.usableTokensPerSecond }
        return candidates.max { a, b in
            let aGreen = a.grade == .green, bGreen = b.grade == .green
            if aGreen != bGreen { return !aGreen }
            if a.fitsNow != b.fitsNow { return !a.fitsNow }
            if a.spec.qualityScore != b.spec.qualityScore { return a.spec.qualityScore < b.spec.qualityScore }
            return a.tokensPerSecond < b.tokensPerSecond
        }
    }

    /// The best model this Mac could run comfortably — green against everything the GPU may
    /// use, at 8k context or more — best first. What a fresh install recommends.
    public func recommendations(limit: Int = 2, excluding installed: Set<String> = []) -> [ChatFit] {
        let emptied = ChatModelPicker(
            hardware: hardware.withAvailableMemory(Int64(Double(hardware.usableMemoryBytes) / 0.85)),
            calibration: calibration
        )
        return ChatCatalog.models
            .filter { !installed.contains($0.repo) }
            .map { emptied.fit($0) }
            .filter { $0.grade == .green && $0.context >= 8_192 }
            .sorted { $0.spec.qualityScore > $1.spec.qualityScore }
            .prefix(limit)
            .map { $0 }
    }

    /// A model worth installing because it would beat `current` here — or nil.
    public func betterThan(_ current: ChatFit?, installed: Set<String>) -> ChatFit? {
        guard let best = recommendations(limit: 1, excluding: installed).first else { return nil }
        guard let current else { return best }
        return best.spec.qualityScore > current.spec.qualityScore ? best : nil
    }

    public static func contextLabel(_ tokens: Int) -> String {
        tokens >= 1_024 ? "\(tokens / 1_024)k" : "\(tokens)"
    }
}
