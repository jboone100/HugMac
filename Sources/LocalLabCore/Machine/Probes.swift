import Foundation

/// One of the synthetic measurements LocalLab takes on first launch (plan §5.14): small MLX
/// kernels shaped like the work its engines do, run without downloading a model.
public enum ProbeKind: String, Sendable, Codable, CaseIterable, Identifiable {
    case memoryBandwidth
    case matmulF16
    case quantizedMatmul4
    case quantizedMatmul8
    case attention
    case conv3d
    case memoryHeadroom
    case diskRead

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .memoryBandwidth: "Memory bandwidth"
        case .matmulF16: "Matrix multiply (fp16)"
        case .quantizedMatmul4: "Quantized multiply (4-bit)"
        case .quantizedMatmul8: "Quantized multiply (8-bit)"
        case .attention: "Attention"
        case .conv3d: "Video convolution"
        case .memoryHeadroom: "GPU memory headroom"
        case .diskRead: "Library disk read"
        }
    }

    public var unit: String {
        switch self {
        case .memoryBandwidth, .diskRead: "GB/s"
        case .matmulF16, .quantizedMatmul4, .quantizedMatmul8, .attention, .conv3d: "TFLOPS"
        case .memoryHeadroom: "GB"
        }
    }

    /// What the probe stands in for, in a user's terms.
    public var standsInFor: String {
        switch self {
        case .memoryBandwidth: "chat speed, model loading"
        case .matmulF16: "upscaler and video transformers"
        case .quantizedMatmul4: "4-bit chat models"
        case .quantizedMatmul8: "8-bit models, SeedVR2 int8"
        case .attention: "long prompts, video attention"
        case .conv3d: "video encode and decode (the upscaler's slowest phases)"
        case .memoryHeadroom: "how much the GPU could hold at the time, and whether it gave it back"
        case .diskRead: "how fast models load from your library"
        }
    }

    /// Whether a larger value means faster. Headroom is a capacity, not a speed.
    public var isThroughput: Bool { self != .memoryHeadroom }

    enum SpecBasis { case compute, bandwidth }

    /// The spec-sheet figure this probe's work scales with, when there's no probe to compare.
    var specBasis: SpecBasis {
        switch self {
        case .memoryBandwidth, .quantizedMatmul4, .quantizedMatmul8, .diskRead, .memoryHeadroom: .bandwidth
        case .matmulF16, .attention, .conv3d: .compute
        }
    }

    /// The probe an engine phase resembles. VAE phases are 3-D convolutions over video
    /// tensors; a diffusion transformer is dominated by dense matrix multiplies; LLM token
    /// generation streams the weights through memory once per token.
    public static func forPhase(_ phase: String) -> ProbeKind? {
        let name = phase.lowercased()
        if name.hasPrefix("vae") { return .conv3d }
        if name == "dit" || name.contains("prefill") || name.contains("transformer") { return .matmulF16 }
        if name.contains("token") || name == "decode" { return .memoryBandwidth }
        return nil
    }
}

public struct ProbeResult: Sendable, Codable, Equatable {
    public let kind: ProbeKind
    public let value: Double
    /// What exactly was run, so a number can be reproduced — "4096×4096 fp16, best of 3".
    public let detail: String

    public init(kind: ProbeKind, value: Double, detail: String) {
        self.kind = kind
        self.value = value
        self.detail = detail
    }
}

/// One run of the probe suite on one Mac.
public struct ProbeReport: Sendable, Codable, Equatable {
    /// Bumped whenever a probe's workload changes, since results from different workloads
    /// don't compare.
    public static let currentSuiteVersion = 1

    public let machine: MachineKey
    public let date: Date
    public let suiteVersion: Int
    /// "26.6.2"
    public let macOSVersion: String
    public let durationSeconds: Double
    public let results: [ProbeResult]

    public init(
        machine: MachineKey, date: Date = Date(), suiteVersion: Int = ProbeReport.currentSuiteVersion,
        macOSVersion: String, durationSeconds: Double, results: [ProbeResult]
    ) {
        self.machine = machine
        self.date = date
        self.suiteVersion = suiteVersion
        self.macOSVersion = macOSVersion
        self.durationSeconds = durationSeconds
        self.results = results
    }

    public func value(_ kind: ProbeKind) -> Double? {
        results.first { $0.kind == kind }?.value
    }

    public var macOSMajor: Int? {
        macOSVersion.split(separator: ".").first.flatMap { Int($0) }
    }
}

/// First-run probe results, kept beside `calibration.json` and local like it.
public struct ProbeStore: Sendable, Equatable {
    public private(set) var reports: [ProbeReport]

    public init(reports: [ProbeReport] = []) {
        self.reports = reports
    }

    /// The newest report for a Mac that runs like `machine`.
    public func latest(for machine: MachineKey) -> ProbeReport? {
        reports.filter { $0.machine.runsLike(machine) }.max { $0.date < $1.date }
    }

    /// Whether the probes should run (again): never run here, or run under a different
    /// suite version or macOS major version — both change what the numbers mean.
    public func needsMeasuring(
        _ machine: MachineKey, macOSMajor: Int, suiteVersion: Int = ProbeReport.currentSuiteVersion
    ) -> Bool {
        guard let latest = latest(for: machine) else { return true }
        return latest.suiteVersion != suiteVersion || latest.macOSMajor != macOSMajor
    }

    /// Keep the three newest reports per Mac — enough to see a change, not a history.
    public mutating func record(_ report: ProbeReport) {
        reports.append(report)
        var kept: [MachineKey: Int] = [:]
        reports = reports.sorted { $0.date > $1.date }.filter { report in
            let count = kept[report.machine, default: 0]
            kept[report.machine] = count + 1
            return count < 3
        }.reversed()
    }

    public static func url(besideCalibration calibration: URL) -> URL {
        calibration.deletingLastPathComponent().appendingPathComponent("probes.json")
    }

    public static func defaultURL() -> URL {
        url(besideCalibration: CalibrationStore.defaultURL())
    }

    public static func load(from url: URL) -> ProbeStore {
        guard let data = try? Data(contentsOf: url) else { return ProbeStore() }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return ProbeStore(reports: (try? decoder.decode([ProbeReport].self, from: data)) ?? [])
    }

    public func save(to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(reports).write(to: url, options: .atomic)
    }
}
