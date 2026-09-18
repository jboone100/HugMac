import Foundation
import IOKit

/// What this Mac actually is, and what it has free right now.
///
/// Replaces MLXUI's `SystemInfo`, whose chip detection mapped `hw.model` through a
/// hand-written table covering only `Mac14,*`/`Mac15,*`/`Mac16,*`; everything else — every
/// M1 Mac, M1 Ultra (800 GB/s, reported as 100), M3 Ultra — fell through to a 100 GB/s
/// default. `machdep.cpu.brand_string` names the chip directly, so the table isn't needed.
public struct HardwareProfile: Sendable, Equatable {
    public enum Tier: String, Sendable, Equatable {
        case base, pro, max, ultra
    }

    /// e.g. "Apple M2 Max"
    public let chipName: String
    /// e.g. 2 for M2
    public let generation: Int?
    public let tier: Tier
    public let gpuCoreCount: Int?
    public let memoryBandwidthGBps: Double
    public let totalMemoryBytes: Int64
    /// Memory that can be allocated now without pushing the system into pressure. The
    /// conservative figure — the resolver plans against this, not `totalMemoryBytes`.
    public let availableMemoryBytes: Int64
    /// The GPU wired limit (`iogpu.wired_limit_mb`). A model above this fails even when RAM
    /// looks free. `0` from sysctl means "system default", which we compute rather than
    /// report as zero.
    public let gpuWiredLimitBytes: Int64
    public let macOSVersion: OperatingSystemVersion

    public static func detect() -> HardwareProfile {
        let brand = Self.sysctlString("machdep.cpu.brand_string") ?? "Apple Silicon"
        let tier = Self.tier(from: brand)
        let generation = Self.generation(from: brand)
        let total = Int64(ProcessInfo.processInfo.physicalMemory)

        return HardwareProfile(
            chipName: brand,
            generation: generation,
            tier: tier,
            gpuCoreCount: Self.gpuCoreCount(),
            memoryBandwidthGBps: Self.bandwidth(generation: generation, tier: tier),
            totalMemoryBytes: total,
            availableMemoryBytes: Self.availableMemory(total: total),
            gpuWiredLimitBytes: Self.wiredLimit(total: total),
            macOSVersion: ProcessInfo.processInfo.operatingSystemVersion
        )
    }

    public init(
        chipName: String,
        generation: Int?,
        tier: Tier,
        gpuCoreCount: Int?,
        memoryBandwidthGBps: Double,
        totalMemoryBytes: Int64,
        availableMemoryBytes: Int64,
        gpuWiredLimitBytes: Int64,
        macOSVersion: OperatingSystemVersion
    ) {
        self.chipName = chipName
        self.generation = generation
        self.tier = tier
        self.gpuCoreCount = gpuCoreCount
        self.memoryBandwidthGBps = memoryBandwidthGBps
        self.totalMemoryBytes = totalMemoryBytes
        self.availableMemoryBytes = availableMemoryBytes
        self.gpuWiredLimitBytes = gpuWiredLimitBytes
        self.macOSVersion = macOSVersion
    }

    public static func == (lhs: HardwareProfile, rhs: HardwareProfile) -> Bool {
        lhs.chipName == rhs.chipName
            && lhs.totalMemoryBytes == rhs.totalMemoryBytes
            && lhs.availableMemoryBytes == rhs.availableMemoryBytes
    }

    // MARK: - Derived budgets

    public var totalMemoryGB: Double { Double(totalMemoryBytes) / 1_073_741_824 }
    public var availableMemoryGB: Double { Double(availableMemoryBytes) / 1_073_741_824 }

    /// The ceiling a single model may not cross: the lower of the GPU wired limit and
    /// total RAM less an OS floor.
    public var usableMemoryBytes: Int64 {
        let osFloor: Int64 = 2 * 1_073_741_824
        return min(gpuWiredLimitBytes, max(0, totalMemoryBytes - osFloor))
    }

    public var usableMemoryGB: Double { Double(usableMemoryBytes) / 1_073_741_824 }

    /// What the resolver plans against: what is free now, less a safety margin, and never
    /// above the single-model ceiling.
    public func plannableMemoryBytes(marginFraction: Double = 0.15) -> Int64 {
        let margin = 1.0 - max(0, min(marginFraction, 0.9))
        return min(usableMemoryBytes, Int64(Double(availableMemoryBytes) * margin))
    }

    public var isAppleSilicon: Bool { chipName.hasPrefix("Apple M") }

    // MARK: - sysctl / IOKit

    static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        // sysctl strings are null-terminated; drop the terminator before decoding.
        let bytes = buffer.prefix { $0 != 0 }
        return String(decoding: bytes, as: UTF8.self)
    }

    static func sysctlInt(_ name: String) -> Int64? {
        var value: Int64 = 0
        var size = MemoryLayout<Int64>.size
        if sysctlbyname(name, &value, &size, nil, 0) == 0 { return value }
        var small: Int32 = 0
        var smallSize = MemoryLayout<Int32>.size
        if sysctlbyname(name, &small, &smallSize, nil, 0) == 0 { return Int64(small) }
        return nil
    }

    static func tier(from brand: String) -> Tier {
        let lower = brand.lowercased()
        if lower.contains("ultra") { return .ultra }
        if lower.contains("max") { return .max }
        if lower.contains("pro") { return .pro }
        return .base
    }

    static func generation(from brand: String) -> Int? {
        // "Apple M2 Max" → 2. Tolerates future "Apple M5 Ultra" without a table update.
        guard let range = brand.range(of: #"M(\d+)"#, options: .regularExpression) else { return nil }
        return Int(brand[range].dropFirst())
    }

    /// Memory bandwidth by generation + tier, in GB/s. Keyed on what the chip *is* rather
    /// than on a model identifier, so an unknown machine of a known family is still right.
    static func bandwidth(generation: Int?, tier: Tier) -> Double {
        guard let generation else { return 100 }
        let byTier: [Tier: Double]
        switch generation {
        case 1:  byTier = [.base: 68,  .pro: 200, .max: 400, .ultra: 800]
        case 2:  byTier = [.base: 100, .pro: 200, .max: 400, .ultra: 800]
        case 3:  byTier = [.base: 100, .pro: 150, .max: 400, .ultra: 800]
        case 4:  byTier = [.base: 120, .pro: 273, .max: 546, .ultra: 1092]
        default: byTier = [.base: 120, .pro: 273, .max: 546, .ultra: 1092]
        }
        return byTier[tier] ?? 100
    }

    /// GPU core count from the IORegistry. Metal exposes no core count, and core count
    /// predicts diffusion and video throughput better than bandwidth does.
    static func gpuCoreCount() -> Int? {
        var iterator: io_iterator_t = 0
        guard let match = IOServiceMatching("IOAccelerator") else { return nil }
        guard IOServiceGetMatchingServices(kIOMainPortDefault, match, &iterator) == KERN_SUCCESS else {
            return nil
        }
        defer { IOObjectRelease(iterator) }
        var service = IOIteratorNext(iterator)
        while service != 0 {
            defer {
                IOObjectRelease(service)
                service = IOIteratorNext(iterator)
            }
            if let value = IORegistryEntrySearchCFProperty(
                service, kIOServicePlane, "gpu-core-count" as CFString,
                kCFAllocatorDefault, IOOptionBits(kIORegistryIterateRecursively)
            ) as? Int {
                return value
            }
        }
        return nil
    }

    /// free + inactive + purgeable pages: what can be handed out before the system starts
    /// evicting things that matter.
    static func availableMemory(total: Int64) -> Int64 {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return total / 2 }
        var pageSize: vm_size_t = 0
        guard host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS else { return total / 2 }
        let pages = Int64(stats.free_count) + Int64(stats.inactive_count) + Int64(stats.purgeable_count)
        return pages * Int64(pageSize)
    }

    /// `iogpu.wired_limit_mb` when set; otherwise macOS's own default, which is a fraction
    /// of physical memory that rises on larger machines.
    static func wiredLimit(total: Int64) -> Int64 {
        if let mb = sysctlInt("iogpu.wired_limit_mb"), mb > 0 {
            return mb * 1_048_576
        }
        let totalGB = Double(total) / 1_073_741_824
        let fraction: Double = totalGB >= 64 ? 0.80 : (totalGB >= 32 ? 0.75 : 0.70)
        return Int64(Double(total) * fraction)
    }
}
