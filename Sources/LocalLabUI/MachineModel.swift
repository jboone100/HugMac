import Foundation
import LocalLabCore
import LocalLabMLX
import Observation

/// The *This Mac* screen: what this Mac is, what it can do, and its first-run measurement
/// (plan §5.13, §5.14). Re-resolved at launch, after the probes, and after every job — each
/// can change what the profile says.
@MainActor
@Observable
public final class MachineModel {
    public enum ProbeState: Equatable {
        case idle
        case running(completed: Int, total: Int, current: ProbeKind?)
        case failed(String)
    }

    public private(set) var profile: MachineProfile
    public private(set) var probeState: ProbeState = .idle
    /// Why the first-run measurement didn't start by itself, when it didn't.
    public private(set) var autoMeasureBlocked: String?
    public private(set) var libraryFreeBytes: Int64?

    private let store: ModelStore
    private let queue: JobQueue
    private let calibrationURL: URL
    private let defaults: UserDefaults
    @ObservationIgnored private var probeTask: Task<Void, Never>?

    static let declinedKey = "LocalLab.firstRunProbesDeclined"

    public init(
        store: ModelStore, queue: JobQueue,
        calibrationURL: URL = CalibrationStore.defaultURL(),
        defaults: UserDefaults = .standard
    ) {
        self.store = store
        self.queue = queue
        self.calibrationURL = calibrationURL
        self.defaults = defaults
        profile = MachineProfile.resolve(
            hardware: .detect(), calibration: CalibrationStore.load(from: calibrationURL),
            installedUpscalers: SeedVR2Variant.installed(in: store)
        )
        libraryFreeBytes = Self.freeBytes(at: store.baseDirectory)
    }

    public var probes: ProbeReport? { profile.probes }
    public var isMeasuring: Bool {
        if case .running = probeState { return true }
        return false
    }

    /// Nothing measured yet on this Mac, under this suite and this macOS.
    public var needsMeasuring: Bool {
        CalibrationStore.load(from: calibrationURL).probes.needsMeasuring(
            profile.machine, macOSMajor: profile.hardware.macOSVersion.majorVersion
        )
    }

    public func refresh() {
        profile = MachineProfile.resolve(
            hardware: .detect(), calibration: CalibrationStore.load(from: calibrationURL),
            installedUpscalers: SeedVR2Variant.installed(in: store)
        )
        libraryFreeBytes = Self.freeBytes(at: store.baseDirectory)
    }

    /// Run the probes on first launch — or after a macOS upgrade or a probe-suite change —
    /// unless the user stopped them before, or now is a bad time.
    public func measureIfNeeded() {
        guard needsMeasuring, !isMeasuring else { return }
        guard !defaults.bool(forKey: Self.declinedKey) else {
            autoMeasureBlocked = "You stopped the first measurement, so it won't start by itself."
            return
        }
        if let reason = blocker() {
            autoMeasureBlocked = reason + " It will start by itself next time LocalLab opens."
            return
        }
        autoMeasureBlocked = nil
        measure(automatic: true)
    }

    /// Why measuring now would give wrong numbers or cost the user something, if it would.
    public func blocker() -> String? {
        if queue.isRunningJob { return "A job is running, and measuring now would measure the job." }
        let power = PowerState.current()
        if power.onBattery, let percent = power.batteryPercent, percent < 30 {
            return "The battery is at \(percent)%."
        }
        switch ProcessInfo.processInfo.thermalState {
        case .serious, .critical: return "The Mac is too hot to measure fairly."
        default: return nil
        }
    }

    public func measure(automatic: Bool = false) {
        guard !isMeasuring else { return }
        if let reason = blocker() {
            probeState = .failed(reason)
            return
        }
        let hardware = HardwareProfile.detect()
        let library = store.baseDirectory
        let url = ProbeStore.url(besideCalibration: calibrationURL)
        probeState = .running(completed: 0, total: ProbeKind.allCases.count, current: ProbeKind.allCases.first)
        probeTask = Task {
            do {
                // Off the main actor: the probes block on the GPU. The cancel is forwarded,
                // since a detached task doesn't inherit it.
                let work = Task.detached(priority: .userInitiated) {
                    try await ProbeSuite.run(hardware: hardware, library: library) { progress in
                        Task { @MainActor [weak self] in
                            guard let self, self.isMeasuring else { return }
                            self.probeState = .running(
                                completed: progress.completed, total: progress.total, current: progress.running
                            )
                        }
                    }
                }
                let report = try await withTaskCancellationHandler {
                    try await work.value
                } onCancel: {
                    work.cancel()
                }
                var probes = ProbeStore.load(from: url)
                probes.record(report)
                try probes.save(to: url)
                queue.updateProbes(probes)
                defaults.set(false, forKey: Self.declinedKey)
                probeState = .idle
                autoMeasureBlocked = nil
                refresh()
            } catch is CancellationError {
                if automatic { defaults.set(true, forKey: Self.declinedKey) }
                probeState = .idle
            } catch {
                probeState = .failed(error.localizedDescription)
            }
        }
    }

    public func stopMeasuring() {
        probeTask?.cancel()
    }

    static func freeBytes(at url: URL) -> Int64? {
        let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }
}
