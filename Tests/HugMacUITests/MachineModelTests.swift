import Foundation
import HugMacCore
import Testing
@testable import HugMacUI

@Suite("This Mac screen model")
@MainActor
struct MachineModelTests {
    func defaults() -> UserDefaults {
        let name = "hugmac-tests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name) ?? .standard
        defaults.removePersistentDomain(forName: name)
        return defaults
    }

    @Test("First launch measures; a Mac already measured under this macOS doesn't")
    func firstLaunchOnly() throws {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        let queue = JobQueue(store: workspace.store, executor: FakeExecutor(), activity: NoActivity(),
                             calibrationURL: workspace.calibrationURL)
        let model = MachineModel(store: workspace.store, queue: queue,
                                 calibrationURL: workspace.calibrationURL, defaults: defaults())
        #expect(model.needsMeasuring)

        let hardware = HardwareProfile.detect()
        var probes = ProbeStore()
        probes.record(ProbeReport(
            machine: hardware.machineKey, macOSVersion: hardware.macOSVersionString,
            durationSeconds: 5, results: [ProbeResult(kind: .conv3d, value: 7, detail: "test")]
        ))
        try probes.save(to: ProbeStore.url(besideCalibration: workspace.calibrationURL))
        #expect(!model.needsMeasuring)
        model.refresh()
        #expect(model.probes?.value(.conv3d) == 7)
    }

    @Test("Stopping the first measurement means it never starts by itself again")
    func declinedStaysDeclined() {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        let queue = JobQueue(store: workspace.store, executor: FakeExecutor(), activity: NoActivity(),
                             calibrationURL: workspace.calibrationURL)
        let settings = defaults()
        settings.set(true, forKey: MachineModel.declinedKey)
        let model = MachineModel(store: workspace.store, queue: queue,
                                 calibrationURL: workspace.calibrationURL, defaults: settings)
        model.measureIfNeeded()
        #expect(!model.isMeasuring)
        #expect(model.autoMeasureBlocked?.contains("won't start by itself") == true)
    }

    @Test("The profile names the installed checkpoint")
    func installedCheckpoint() throws {
        let workspace = Workspace()
        defer { workspace.cleanUp() }
        workspace.markInstalled(.threeBInt8)
        let queue = JobQueue(store: workspace.store, executor: FakeExecutor(), activity: NoActivity(),
                             calibrationURL: workspace.calibrationURL)
        let model = MachineModel(store: workspace.store, queue: queue,
                                 calibrationURL: workspace.calibrationURL, defaults: defaults())
        let video = try #require(model.profile.capabilities.first { $0.id == "upscale-video" })
        if video.status != .tooLarge {
            #expect(video.headline.hasPrefix("SeedVR2 3B int8"))
        }
    }
}
