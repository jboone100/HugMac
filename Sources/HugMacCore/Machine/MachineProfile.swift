import Foundation

/// What HugMac knows about the Mac it is running on, and what it can do here — resolved at
/// every launch from three layers, most trusted last (plan §5.13):
///
/// 1. **Rules** — hardware facts: memory, the GPU's wired limit, cores, bandwidth.
/// 2. **Reference Macs** — measurements that ship with the app, scaled to this Mac by its
///    first-run probes, or by spec sheet before it has any.
/// 3. **This Mac's calibration** — its own runs, which replace everything above them.
///
/// It is **computed, not looked up**: nothing here is a table of Mac models. A chip released
/// after this build still gets a profile, because every figure comes from the same resolver
/// the task screens use.
public struct MachineProfile: Sendable {
    public let hardware: HardwareProfile
    public let power: PowerState
    public let probes: ProbeReport?
    public let timing: TimingSource
    public let capabilities: [Capability]

    public var machine: MachineKey { hardware.machineKey }

    /// Where this Mac's time estimates come from, for the one-line summary.
    public enum TimingSource: Sendable, Equatable {
        /// This Mac has run the engines itself.
        case measuredHere
        /// Scaled from a reference Mac by comparing the two Macs' probe results.
        case probes(fromChip: String)
        /// Scaled from a reference Mac by spec sheet — the first-run probes haven't run.
        case specs(fromChip: String)
        /// Nothing to go on yet.
        case none

        public var summary: String {
            switch self {
            case .measuredHere: "Times are measured on this Mac."
            case .probes(let chip): "Times are estimated from an \(chip), scaled by this Mac's measured speed."
            case .specs(let chip): "Times are estimated from an \(chip), scaled by spec sheet until this Mac is measured."
            case .none: "No time estimates yet — they appear after the first run."
            }
        }
    }

    public static func resolve(
        hardware: HardwareProfile,
        calibration: CalibrationStore,
        installedUpscalers: Set<SeedVR2Variant>,
        power: PowerState = .current()
    ) -> MachineProfile {
        let machine = hardware.machineKey
        let probes = calibration.probes.latest(for: machine)
        return MachineProfile(
            hardware: hardware,
            power: power,
            probes: probes,
            timing: timingSource(machine: machine, probes: probes, calibration: calibration),
            capabilities: [
                upscaleImage(hardware: hardware, calibration: calibration, installed: installedUpscalers),
                upscaleVideo(hardware: hardware, calibration: calibration, installed: installedUpscalers),
                chat(hardware: hardware),
                videoGeneration(hardware: hardware),
            ]
        )
    }

    static func timingSource(
        machine: MachineKey, probes: ProbeReport?, calibration: CalibrationStore
    ) -> TimingSource {
        if calibration.samples.contains(where: {
            $0.engineID == SeedVR2Resolver.mlxEngineID && $0.machine.runsLike(machine)
        }) {
            return .measuredHere
        }
        guard let reference = calibration.reference.first(where: { $0.machine.chipName == machine.chipName })
                ?? calibration.reference.first else { return .none }
        if probes != nil, reference.probes != nil { return .probes(fromChip: reference.machine.chipName) }
        return .specs(fromChip: reference.machine.chipName)
    }

    // MARK: - Capabilities

    /// One thing this Mac can or can't do, with the arithmetic that decided it.
    public struct Capability: Sendable, Identifiable, Equatable {
        public enum Status: Sendable, Equatable {
            /// Fits in the memory free right now.
            case ready
            /// Fits in what the GPU can use, but not in what's free — close some apps.
            case closeApps
            /// Doesn't fit on this Mac at all.
            case tooLarge
            /// HugMac can't do this yet; the figures are the plan's estimate.
            case notBuilt
        }

        public let id: String
        public let title: String
        /// The worked example the verdict is for — "a 1080p photo to 4K".
        public let example: String
        public let status: Status
        public let headline: String
        public let detail: String
        public let time: TimeEstimate?

        public init(
            id: String, title: String, example: String, status: Status,
            headline: String, detail: String, time: TimeEstimate? = nil
        ) {
            self.id = id
            self.title = title
            self.example = example
            self.status = status
            self.headline = headline
            self.detail = detail
            self.time = time
        }
    }

    static func upscaleImage(
        hardware: HardwareProfile, calibration: CalibrationStore, installed: Set<SeedVR2Variant>
    ) -> Capability {
        // Not 1080p → 4K: SeedVR2's transformer can't tile, and a 4K frame's ~32k tokens
        // need about 30 GB in that phase alone — beyond most Macs, however many apps close.
        upscale(
            id: "upscale-image", title: "Upscale an image", example: "a 1024×1024 image to 2048×2048",
            source: .image(width: 1024, height: 1024), target: .scale(2),
            hardware: hardware, calibration: calibration, installed: installed
        )
    }

    static func upscaleVideo(
        hardware: HardwareProfile, calibration: CalibrationStore, installed: Set<SeedVR2Variant>
    ) -> Capability {
        // The owner's reference clip, cut to five seconds: 672×384 → 1344×768.
        let clip = VideoMedia(
            url: URL(fileURLWithPath: "/dev/null"), width: 672, height: 384,
            fps: 24, frameCount: 121, hasAudio: false
        )
        return upscale(
            id: "upscale-video", title: "Upscale a video", example: "5 s of 384p video to 768p",
            source: .video(clip), target: .scale(2),
            hardware: hardware, calibration: calibration, installed: installed
        )
    }

    /// The verdict for one upscale example, in two parts:
    ///
    /// - **What runs now** — with the checkpoints installed, against memory free now, or
    ///   failing that against everything the GPU may use ("close some apps").
    /// - **What this Mac could run** — the best checkpoint that fits at all, named when it is
    ///   better than what's installed. Never downloaded on the user's behalf.
    ///
    /// A time is shown only for a checkpoint that has been measured — here or on a reference
    /// Mac. Per-unit rates are per checkpoint: pricing a 7B transformer at a 3B's measured
    /// speed would under-state it by more than half.
    static func upscale(
        id: String, title: String, example: String,
        source: SeedVR2Resolver.Source, target: UpscaleTarget,
        hardware: HardwareProfile, calibration: CalibrationStore, installed: Set<SeedVR2Variant>
    ) -> Capability {
        let emptied = hardware.withAvailableMemory(Int64(Double(hardware.usableMemoryBytes) / 0.85))
        func plan(_ hw: HardwareProfile, _ variants: Set<SeedVR2Variant>) throws -> SeedVR2Plan {
            try SeedVR2Resolver(hardware: hw, calibration: calibration)
                .plan(source: source, target: target, quality: .balanced, installedVariants: variants)
        }
        let bestPossible = try? plan(emptied, Set(SeedVR2Variant.allCases))

        func describe(_ plan: SeedVR2Plan, status: Capability.Status, lead: String) -> Capability {
            let timed = calibration.hasMeasurements(of: plan.variant)
            let time = timed ? plan.totalTime : .unknown
            var headline = plan.variant.displayName
            if let seconds = time.seconds { headline += " · about " + Self.duration(seconds) }
            var detail = lead + String(format: " Peak %.1f GB. ", plan.peakGB)
            detail += timed ? Self.provenance(time) : "Time known after its first run here."
            if let better = bestPossible?.variant, better != plan.variant,
               SeedVR2Variant.byQualityDescending.firstIndex(of: better)
                   ?? .max < SeedVR2Variant.byQualityDescending.firstIndex(of: plan.variant) ?? .max {
                detail += " \(better.displayName) would also fit, for higher quality — "
                    + String(format: "%.1f GB to download.", Double(better.downloadBytes) / 1_073_741_824)
            }
            return Capability(
                id: id, title: title, example: example, status: status,
                headline: headline, detail: detail, time: time
            )
        }

        guard !installed.isEmpty else {
            guard let bestPossible else { return tooLarge(id: id, title: title, example: example,
                                                          hardware: hardware, error: nil) }
            let variant = bestPossible.variant
            return Capability(
                id: id, title: title, example: example,
                status: (try? plan(hardware, [variant])) != nil ? .ready : .closeApps,
                headline: "Install \(variant.displayName) to start",
                detail: String(format: "The best checkpoint this Mac can run for this — a %.1f GB download. Peak %.1f GB.",
                               Double(variant.downloadBytes) / 1_073_741_824, bestPossible.peakGB)
            )
        }
        if let now = try? plan(hardware, installed) {
            return describe(now, status: .ready,
                            lead: "The best installed checkpoint that fits in the memory free now.")
        }
        do {
            return describe(try plan(emptied, installed), status: .closeApps,
                            lead: String(format: "Needs more than the %.1f GB free now; fits once other apps are closed.",
                                         hardware.availableMemoryGB))
        } catch {
            return tooLarge(id: id, title: title, example: example, hardware: hardware, error: error)
        }
    }

    static func tooLarge(
        id: String, title: String, example: String, hardware: HardwareProfile, error: Error?
    ) -> Capability {
        if case StageError.insufficientMemory(let requiredGB, _)? = error {
            return Capability(
                id: id, title: title, example: example, status: .tooLarge,
                headline: String(format: "Too large — needs %.1f GB", requiredGB),
                detail: String(format: "Its smallest plan needs %.1f GB; this Mac's GPU can use %.1f GB.",
                               requiredGB, hardware.usableMemoryGB)
            )
        }
        return Capability(
            id: id, title: title, example: example, status: .tooLarge,
            headline: "Too large for this Mac",
            detail: error?.localizedDescription
                ?? String(format: "No SeedVR2 checkpoint fits in the %.1f GB this Mac's GPU can use.",
                          hardware.usableMemoryGB)
        )
    }

    /// Chat isn't built yet, so this is the plan's rule of thumb (§5.2), labelled as one:
    /// 4-bit weights at 4.5 bits each, plus ~2 GB for an 8k context and runtime, within 70%
    /// of the memory the GPU can use.
    static func chat(hardware: HardwareProfile) -> Capability {
        let usable = hardware.usableMemoryGB
        let budget = usable * 0.7 - 2
        let bytesPerParamGB = 4.5 / 8 * 1e9 / 1_073_741_824
        let largest = budget / bytesPerParamGB
        let classes: [Double] = [1, 3, 4, 8, 14, 24, 32, 70, 120, 235]
        let fits = classes.last { $0 <= largest }
        let headline = fits.map { "Up to about \(Int($0))B parameters at 4-bit" } ?? "Only the smallest models"
        return Capability(
            id: "chat", title: "Chat", example: "a 4-bit model with an 8k context", status: .notBuilt,
            headline: headline,
            detail: String(format: "Coming in a later phase. Rule of thumb: 4.5 bits per weight plus about 2 GB for context, within 70%% of the %.1f GB the GPU can use.", usable)
        )
    }

    /// Text-to-video isn't built yet. MiniMax-H3 at 384p peaked at about 15 GB in the plan's
    /// estimate (§6.2, §5.11).
    static func videoGeneration(hardware: HardwareProfile) -> Capability {
        let needGB = 15.0
        let fits = hardware.usableMemoryGB >= needGB
        return Capability(
            id: "text-to-video", title: "Generate video", example: "MiniMax-H3 at 384p", status: .notBuilt,
            headline: fits ? "Should fit — about 15 GB" : "Won't fit — needs about 15 GB",
            detail: String(format: "Coming in a later phase. The plan's estimate is about %.0f GB at 384p, against %.1f GB the GPU can use here.", needGB, hardware.usableMemoryGB)
        )
    }

    static func provenance(_ time: TimeEstimate) -> String {
        switch time {
        case .unknown: "Time unknown until it has run here."
        case .measured: "Time measured on this Mac."
        case .extrapolated(_, let chip, .probes): "Time estimated from an \(chip), scaled by this Mac's measured speed."
        case .extrapolated(_, let chip, .specs): "Time estimated from an \(chip), scaled by spec sheet."
        }
    }

    static func duration(_ seconds: Double) -> String {
        if seconds < 90 { return "\(Int(seconds.rounded())) s" }
        if seconds < 5400 { return "\(Int((seconds / 60).rounded())) min" }
        return String(format: "%.1f h", seconds / 3600)
    }
}

public extension SeedVR2Variant {
    /// "SeedVR2 3B int8"
    var displayName: String { rawValue.replacingOccurrences(of: "-", with: " ") }

    /// The checkpoints installed in `store`'s library.
    static func installed(in store: ModelStore) -> Set<SeedVR2Variant> {
        Set(allCases.filter { store.isInstalled(repo: $0.hfRepo) })
    }

    /// The checkpoint a sample was measured with, read from its note ("SeedVR2-3B-int8, …").
    init?(sampleNote note: String) {
        guard let name = note.split(separator: ",").first.map(String.init) else { return nil }
        self.init(rawValue: name)
    }
}

public extension CalibrationStore {
    /// Whether any timing exists for this checkpoint — measured here or on a reference Mac.
    func hasMeasurements(of variant: SeedVR2Variant) -> Bool {
        (samples + reference.flatMap(\.samples)).contains {
            $0.engineID == SeedVR2Resolver.mlxEngineID && SeedVR2Variant(sampleNote: $0.note) == variant
        }
    }
}

public extension HardwareProfile {
    /// The same Mac with a different amount of free memory — for asking "would it fit if
    /// other apps were closed?".
    func withAvailableMemory(_ bytes: Int64) -> HardwareProfile {
        HardwareProfile(
            chipName: chipName, generation: generation, tier: tier, gpuCoreCount: gpuCoreCount,
            memoryBandwidthGBps: memoryBandwidthGBps, totalMemoryBytes: totalMemoryBytes,
            availableMemoryBytes: bytes, gpuWiredLimitBytes: gpuWiredLimitBytes,
            macOSVersion: macOSVersion
        )
    }

    var macOSVersionString: String {
        "\(macOSVersion.majorVersion).\(macOSVersion.minorVersion).\(macOSVersion.patchVersion)"
    }
}
