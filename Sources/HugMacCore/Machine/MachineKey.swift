import Foundation

/// Which Mac a measurement belongs to: enough to tell two Macs apart when their speed
/// differs, and no more — no serial number, no hardware UUID.
///
/// The CPU brand string alone conflates variants: an M2 Max ships with 30 or 38 GPU cores,
/// and those run a diffusion phase at different speeds. Memory is recorded too, though it
/// doesn't change speed — it's what decides what *fits*, and a profile shows both.
public struct MachineKey: Sendable, Codable, Hashable {
    /// e.g. "Apple M2 Max"
    public let chipName: String
    public let gpuCores: Int?
    /// Physical memory, rounded to whole GB. `0` when unknown.
    public let memoryGB: Int

    public init(chipName: String, gpuCores: Int? = nil, memoryGB: Int = 0) {
        self.chipName = chipName
        self.gpuCores = gpuCores
        self.memoryGB = memoryGB
    }

    /// True when timings from `other` can be used here unscaled: the same chip and, where
    /// both are known, the same GPU core count. Memory is ignored — it doesn't change speed.
    public func runsLike(_ other: MachineKey) -> Bool {
        guard chipName == other.chipName else { return false }
        guard let mine = gpuCores, let theirs = other.gpuCores else { return true }
        return mine == theirs
    }

    public var generation: Int? { HardwareProfile.generation(from: chipName) }
    public var tier: HardwareProfile.Tier { HardwareProfile.tier(from: chipName) }
    public var bandwidthGBps: Double { HardwareProfile.bandwidth(generation: generation, tier: tier) }

    /// "Apple M2 Max · 30-core GPU · 32 GB"
    public var displayName: String {
        var parts = [chipName]
        if let gpuCores { parts.append("\(gpuCores)-core GPU") }
        if memoryGB > 0 { parts.append("\(memoryGB) GB") }
        return parts.joined(separator: " · ")
    }
}

public extension HardwareProfile {
    var machineKey: MachineKey {
        MachineKey(
            chipName: chipName, gpuCores: gpuCoreCount,
            memoryGB: Int((Double(totalMemoryBytes) / 1_073_741_824).rounded())
        )
    }
}

/// Carries a timing measured on one Mac to another.
///
/// Memory measurements travel between Macs almost unchanged — the same model, the same MLX
/// kernels, the same shapes. Time does not: it depends on GPU cores, clocks and bandwidth.
/// So a reference Mac's seconds are multiplied by how much slower this Mac is at the kind of
/// work the phase does.
public enum MachineScaling {
    /// The multiplier to apply to a reference Mac's seconds, and how it was arrived at.
    public static func factor(
        from reference: MachineKey, fromProbes referenceProbes: ProbeReport?,
        to machine: MachineKey, toProbes machineProbes: ProbeReport?,
        phase: String
    ) -> (factor: Double, basis: ScalingBasis) {
        let kind = ProbeKind.forPhase(phase)
        if let kind,
           let theirs = referenceProbes?.value(kind), let mine = machineProbes?.value(kind),
           theirs > 0, mine > 0 {
            // Probe values are throughputs: twice the throughput, half the time.
            return (theirs / mine, .probes)
        }
        if reference.runsLike(machine) { return (1, .specs) }
        switch kind?.specBasis ?? .compute {
        case .bandwidth:
            return (reference.bandwidthGBps / max(machine.bandwidthGBps, 1), .specs)
        case .compute:
            guard let theirs = computeScore(reference), let mine = computeScore(machine) else {
                return (reference.bandwidthGBps / max(machine.bandwidthGBps, 1), .specs)
            }
            return (theirs / mine, .specs)
        }
    }

    /// GPU cores × a per-generation factor for clock speed and architecture. **A guess**,
    /// labelled as one wherever it is used, and replaced by probe ratios once this Mac has
    /// run its first-run probes (plan §5.14).
    static func computeScore(_ machine: MachineKey) -> Double? {
        guard let cores = machine.gpuCores else { return nil }
        let perCore: Double
        switch machine.generation ?? 1 {
        case ...1: perCore = 1.0
        case 2: perCore = 1.1
        case 3: perCore = 1.25
        case 4: perCore = 1.4
        default: perCore = 1.6
        }
        return Double(cores) * perCore
    }
}
