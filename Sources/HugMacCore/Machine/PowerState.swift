import Foundation
import IOKit.ps

/// Whether the Mac is on battery, and how full it is. The first-run probes don't start on a
/// nearly flat battery, and a laptop's profile says it is one (plan §5.14).
public struct PowerState: Sendable, Equatable {
    public let onBattery: Bool
    /// 0–100, or nil on a Mac without a battery.
    public let batteryPercent: Int?

    public init(onBattery: Bool, batteryPercent: Int?) {
        self.onBattery = onBattery
        self.batteryPercent = batteryPercent
    }

    public static func current() -> PowerState {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue() else {
            return PowerState(onBattery: false, batteryPercent: nil)
        }
        let providing = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() as String?
        let onBattery = providing == kIOPSBatteryPowerValue
        var percent: Int?
        if let list = IOPSCopyPowerSourcesList(info)?.takeRetainedValue() as? [CFTypeRef] {
            for source in list {
                guard let description = IOPSGetPowerSourceDescription(info, source)?
                        .takeUnretainedValue() as? [String: Any],
                      description[kIOPSTypeKey] as? String == kIOPSInternalBatteryType,
                      let current = description[kIOPSCurrentCapacityKey] as? Int,
                      let max = description[kIOPSMaxCapacityKey] as? Int, max > 0 else { continue }
                percent = Int((Double(current) / Double(max) * 100).rounded())
            }
        }
        return PowerState(onBattery: onBattery, batteryPercent: percent)
    }

    public var hasBattery: Bool { batteryPercent != nil }
}
