import Foundation
import Testing
@testable import LocalLabCore

private let gib = 1_073_741_824.0

/// A Mac with the facts the profile reads. Wired limit follows macOS's default fraction.
private func mac(
    _ chip: String, cores: Int?, memoryGB: Double, availableGB: Double? = nil
) -> HardwareProfile {
    let total = Int64(memoryGB * gib)
    return HardwareProfile(
        chipName: chip,
        generation: HardwareProfile.generation(from: chip),
        tier: HardwareProfile.tier(from: chip),
        gpuCoreCount: cores,
        memoryBandwidthGBps: HardwareProfile.bandwidth(
            generation: HardwareProfile.generation(from: chip), tier: HardwareProfile.tier(from: chip)
        ),
        totalMemoryBytes: total,
        availableMemoryBytes: Int64((availableGB ?? memoryGB * 0.6) * gib),
        gpuWiredLimitBytes: HardwareProfile.wiredLimit(total: total),
        macOSVersion: .init(majorVersion: 26, minorVersion: 6, patchVersion: 2)
    )
}

private let mains = PowerState(onBattery: false, batteryPercent: nil)
private let m2Max = MachineKey(chipName: "Apple M2 Max", gpuCores: 30, memoryGB: 32)

private func probes(_ machine: MachineKey, scale: Double, macOS: String = "26.6.2",
                    date: Date = Date(), suite: Int = ProbeReport.currentSuiteVersion) -> ProbeReport {
    ProbeReport(
        machine: machine, date: date, suiteVersion: suite, macOSVersion: macOS, durationSeconds: 5,
        results: ProbeKind.allCases.map { ProbeResult(kind: $0, value: 10 * scale, detail: "test") }
    )
}

/// One reference Mac with a known per-unit rate per phase: 1 s per unit for every phase.
private func reference(probes report: ProbeReport? = nil) -> ReferenceMachine {
    ReferenceMachine(
        machine: m2Max, probes: report,
        samples: ["vae-encode", "dit", "vae-decode"].map {
            CalibrationSample(engineID: SeedVR2Resolver.mlxEngineID, phase: $0, workUnits: 100,
                              seconds: 100, peakBytes: 0, machine: m2Max,
                              note: "SeedVR2-3B-int8, 1 chunk(s) of up to 5 frames")
        },
        source: "test"
    )
}

@Suite("Machine key")
struct MachineKeyTests {
    @Test func samePartNumbersRunAlike() {
        let other30 = MachineKey(chipName: "Apple M2 Max", gpuCores: 30, memoryGB: 96)
        let other38 = MachineKey(chipName: "Apple M2 Max", gpuCores: 38, memoryGB: 32)
        #expect(m2Max.runsLike(other30), "memory doesn't change speed")
        #expect(!m2Max.runsLike(other38), "an M2 Max comes with 30 or 38 GPU cores, and they differ")
        #expect(!m2Max.runsLike(MachineKey(chipName: "Apple M3 Max", gpuCores: 30, memoryGB: 32)))
        #expect(m2Max.runsLike(MachineKey(chipName: "Apple M2 Max")), "an old sample without cores still counts")
    }

    @Test func samplesRecordedBeforeKeyingStillDecode() throws {
        let old = """
        [{"engineID":"hugmac-seedvr2-mlx","phase":"dit","workUnits":10,"peakUnits":0,"seconds":5,
          "peakBytes":1,"weightBytes":0,"chipName":"Apple M2 Max","note":""}]
        """
        let samples = try JSONDecoder().decode([CalibrationSample].self, from: Data(old.utf8))
        #expect(samples[0].gpuCores == nil)
        let store = CalibrationStore(samples: samples)
        #expect(store.secondsPerUnit(engineID: SeedVR2Resolver.mlxEngineID, phase: "dit", machine: m2Max) == 0.5)
    }

    @Test func profileKeyComesFromTheHardware() {
        let key = mac("Apple M2 Max", cores: 30, memoryGB: 32).machineKey
        #expect(key == m2Max)
        #expect(key.displayName == "Apple M2 Max · 30-core GPU · 32 GB")
    }
}

@Suite("Carrying time between Macs")
struct ScalingTests {
    let engine = SeedVR2Resolver.mlxEngineID

    @Test func thisMacsOwnMeasurementWins() {
        let own = CalibrationSample(engineID: engine, phase: "dit", workUnits: 100, seconds: 300,
                                    peakBytes: 0, machine: m2Max)
        let store = CalibrationStore(samples: [own], reference: [reference()])
        #expect(store.estimate(engineID: engine, phase: "dit", machine: m2Max, workUnits: 10) == .measured(seconds: 30))
    }

    @Test func withoutMeasurementsAReferenceMacIsScaledAndSaysSo() {
        let store = CalibrationStore(reference: [reference()])
        let estimate = store.estimate(engineID: engine, phase: "dit", machine: m2Max, workUnits: 10)
        #expect(estimate == .extrapolated(seconds: 10, fromChip: "Apple M2 Max", basis: .specs),
                "the same kind of Mac: factor 1, but still an estimate")
    }

    @Test func probesScaleByTheKindOfWorkThePhaseDoes() {
        let slower = MachineKey(chipName: "Apple M1 Pro", gpuCores: 16, memoryGB: 16)
        // This Mac measures half the reference's throughput on every probe.
        let store = CalibrationStore(
            reference: [reference(probes: probes(m2Max, scale: 1))],
            probes: ProbeStore(reports: [probes(slower, scale: 0.5)])
        )
        let encode = store.estimate(engineID: engine, phase: "vae-encode", machine: slower, workUnits: 10)
        #expect(encode == .extrapolated(seconds: 20, fromChip: "Apple M2 Max", basis: .probes))
    }

    @Test func specSheetScalingUsesCoresTimesGeneration() {
        let m1Pro = MachineKey(chipName: "Apple M1 Pro", gpuCores: 16, memoryGB: 16)
        let factor = MachineScaling.factor(from: m2Max, fromProbes: nil, to: m1Pro, toProbes: nil, phase: "dit")
        #expect(factor.basis == .specs)
        #expect(abs(factor.factor - (30 * 1.1) / (16 * 1.0)) < 1e-9)
    }

    @Test func tokenGenerationScalesWithBandwidth() {
        let m3Pro = MachineKey(chipName: "Apple M3 Pro", gpuCores: 18, memoryGB: 18)
        let factor = MachineScaling.factor(from: m2Max, fromProbes: nil, to: m3Pro, toProbes: nil, phase: "decode")
        #expect(abs(factor.factor - 400.0 / 150.0) < 1e-9)
    }

    @Test func totalsAreAsTrustworthyAsTheirWeakestPart() {
        #expect(TimeEstimate.sum([.measured(seconds: 1), .measured(seconds: 2)]) == .measured(seconds: 3))
        #expect(TimeEstimate.sum([
            .measured(seconds: 1),
            .extrapolated(seconds: 2, fromChip: "A", basis: .probes),
            .extrapolated(seconds: 3, fromChip: "B", basis: .specs),
        ]) == .extrapolated(seconds: 6, fromChip: "B", basis: .specs))
        #expect(TimeEstimate.sum([.measured(seconds: 1), .unknown]) == .unknown)
    }

    @Test func phasesMapToTheProbeTheyResemble() {
        #expect(ProbeKind.forPhase("vae-encode") == .conv3d)
        #expect(ProbeKind.forPhase("vae-decode") == .conv3d)
        #expect(ProbeKind.forPhase("dit") == .matmulF16)
        #expect(ProbeKind.forPhase("decode") == .memoryBandwidth)
        #expect(ProbeKind.forPhase("mystery") == nil)
    }

    @Test func theBundledReferenceIsCompleteRunsOnly() {
        let bundled = ReferenceMachine.m2Max30Core32GB
        #expect(bundled.machine == m2Max)
        #expect(bundled.probes != nil)
        // A resumed run reported its skipped work as done: 0.1 s to encode 5 chunks.
        #expect(bundled.samples.allSatisfy { $0.seconds > 1 })
        #expect(bundled.samples.allSatisfy { SeedVR2Variant(sampleNote: $0.note) == .threeBInt8 })
    }
}

@Suite("Probe store")
struct ProbeStoreTests {
    @Test func measuresOnFirstLaunchAndAfterAMacOSUpgrade() {
        var store = ProbeStore()
        #expect(store.needsMeasuring(m2Max, macOSMajor: 26))
        store.record(probes(m2Max, scale: 1, macOS: "26.6.2"))
        #expect(!store.needsMeasuring(m2Max, macOSMajor: 26))
        #expect(store.needsMeasuring(m2Max, macOSMajor: 27), "a major upgrade changes the kernels")
        #expect(store.needsMeasuring(m2Max, macOSMajor: 26, suiteVersion: ProbeReport.currentSuiteVersion + 1))
        #expect(store.needsMeasuring(MachineKey(chipName: "Apple M2 Max", gpuCores: 38, memoryGB: 32), macOSMajor: 26))
    }

    @Test func keepsTheNewestThreePerMac() {
        var store = ProbeStore()
        for day in 0 ..< 5 {
            store.record(probes(m2Max, scale: Double(day + 1), date: Date(timeIntervalSince1970: Double(day) * 86_400)))
        }
        #expect(store.reports.count == 3)
        #expect(store.latest(for: m2Max)?.value(.conv3d) == 50)
    }

    @Test func roundTripsThroughDisk() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("locallab-probes-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        var store = ProbeStore()
        store.record(probes(m2Max, scale: 1, date: Date(timeIntervalSince1970: 1_800_000_000)))
        try store.save(to: url)
        #expect(ProbeStore.load(from: url) == store)
    }

    @Test func loadingCalibrationBringsProbesAndReferences() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("locallab-cal-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let calibrationURL = directory.appendingPathComponent("calibration.json")
        var store = ProbeStore()
        store.record(probes(m2Max, scale: 1))
        try store.save(to: ProbeStore.url(besideCalibration: calibrationURL))
        let loaded = CalibrationStore.load(from: calibrationURL)
        #expect(loaded.probes.latest(for: m2Max) != nil)
        #expect(loaded.reference == ReferenceMachine.all)
        #expect(loaded.samples.isEmpty)
    }
}

@Suite("Machine profile")
struct MachineProfileTests {
    @Test func chatRuleOfThumbGrowsWithMemory() {
        func chat(_ gb: Double) -> String {
            MachineProfile.chat(hardware: mac("Apple M4 Max", cores: 40, memoryGB: gb)).headline
        }
        #expect(chat(8).contains("3B"))
        #expect(chat(16).contains("8B"))
        #expect(chat(32).contains("24B"))
        #expect(chat(128).contains("120B"))
    }

    @Test func aFreshMacIsToldWhatToInstallNotGivenIt() {
        let big = mac("Apple M4 Max", cores: 40, memoryGB: 128, availableGB: 100)
        let profile = MachineProfile.resolve(
            hardware: big, calibration: CalibrationStore(reference: ReferenceMachine.all),
            installedUpscalers: [], power: mains
        )
        let video = try? #require(profile.capabilities.first { $0.id == "upscale-video" })
        #expect(video?.status == .ready)
        #expect(video?.headline == "Install SeedVR2 7B fp16 to start")
    }

    @Test func aSmallMacIsToldWhyAnImageWontFit() {
        let small = mac("Apple M1", cores: 8, memoryGB: 8, availableGB: 5)
        let profile = MachineProfile.resolve(
            hardware: small, calibration: CalibrationStore(reference: ReferenceMachine.all),
            installedUpscalers: [.threeBInt8], power: mains
        )
        let image = profile.capabilities.first { $0.id == "upscale-image" }
        #expect(image?.status == .tooLarge)
        #expect(image?.headline.hasPrefix("Too large — needs") == true)
    }

    @Test func installedModelTimesComeFromTheReferenceMacUntilMeasured() throws {
        let hardware = mac("Apple M3 Pro", cores: 18, memoryGB: 36, availableGB: 28)
        let profile = MachineProfile.resolve(
            hardware: hardware, calibration: CalibrationStore(reference: ReferenceMachine.all),
            installedUpscalers: [.threeBInt8], power: mains
        )
        #expect(profile.timing == .specs(fromChip: "Apple M2 Max"))
        let video = try #require(profile.capabilities.first { $0.id == "upscale-video" })
        #expect(video.status == .ready)
        guard case .extrapolated(_, "Apple M2 Max", .specs)? = video.time else {
            Issue.record("expected a spec-scaled estimate, got \(String(describing: video.time))")
            return
        }
        #expect(video.detail.contains("scaled by spec sheet"))
    }

    @Test func anUnmeasuredCheckpointGetsNoTimeRatherThanAWrongOne() throws {
        let hardware = mac("Apple M4 Max", cores: 40, memoryGB: 64, availableGB: 50)
        let profile = MachineProfile.resolve(
            hardware: hardware, calibration: CalibrationStore(reference: ReferenceMachine.all),
            installedUpscalers: [.sevenBFP16], power: mains
        )
        let video = try #require(profile.capabilities.first { $0.id == "upscale-video" })
        #expect(video.headline == "SeedVR2 7B fp16", "no time: only 3B int8 has been measured anywhere")
        #expect(video.time == .unknown)
        #expect(video.detail.contains("Time known after its first run here."))
    }

    @Test func aBetterCheckpointThatWouldFitIsNamed() throws {
        let hardware = mac("Apple M2 Max", cores: 30, memoryGB: 64, availableGB: 50)
        let profile = MachineProfile.resolve(
            hardware: hardware, calibration: CalibrationStore(reference: ReferenceMachine.all),
            installedUpscalers: [.threeBInt8], power: mains
        )
        let video = try #require(profile.capabilities.first { $0.id == "upscale-video" })
        #expect(video.detail.contains("would also fit, for higher quality"))
    }

    @Test func timingSourceFollowsWhatHasBeenMeasured() {
        let key = m2Max
        let reference = [ReferenceMachine.m2Max30Core32GB]
        #expect(MachineProfile.timingSource(machine: key, probes: nil,
                                            calibration: CalibrationStore(reference: reference))
                == .specs(fromChip: "Apple M2 Max"))
        #expect(MachineProfile.timingSource(machine: key, probes: probes(key, scale: 1),
                                            calibration: CalibrationStore(reference: reference))
                == .probes(fromChip: "Apple M2 Max"))
        let own = CalibrationSample(engineID: SeedVR2Resolver.mlxEngineID, phase: "dit", workUnits: 1,
                                    seconds: 1, peakBytes: 0, machine: key)
        #expect(MachineProfile.timingSource(machine: key, probes: nil,
                                            calibration: CalibrationStore(samples: [own], reference: reference))
                == .measuredHere)
        #expect(MachineProfile.timingSource(machine: key, probes: nil, calibration: CalibrationStore()) == .none)
    }
}
