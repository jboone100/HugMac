import Foundation

/// The sizes Create Image offers. All about one megapixel or less — what FLUX.1 was trained
/// at, and where its VAE's peak (≈1.1 GB + 5.6 KB a pixel) stays below the transformer's.
public enum ImageSize: String, CaseIterable, Identifiable, Sendable, Codable {
    case square512, square768, square1024, landscape, portrait

    public var id: String { rawValue }

    public var width: Int {
        switch self {
        case .square512: 512
        case .square768: 768
        case .square1024: 1024
        case .landscape: 1344
        case .portrait: 768
        }
    }

    public var height: Int {
        switch self {
        case .square512: 512
        case .square768: 768
        case .square1024: 1024
        case .landscape: 768
        case .portrait: 1344
        }
    }

    public var label: String {
        switch self {
        case .square512: "Square, small"
        case .square768: "Square, medium"
        case .square1024: "Square, large"
        case .landscape: "Landscape"
        case .portrait: "Portrait"
        }
    }

    public var dimensions: String { "\(width) × \(height)" }
}

/// Smart Fit for one image model at one size: will it fit, and how long will an image take.
public struct ImageFit: Sendable, Equatable {
    public enum Grade: Sendable, Equatable {
        case green
        case yellow(because: String)
        case red(because: String)
    }

    public let grade: Grade
    /// The largest of the three phases: each one's weights are released before the next loads.
    public let peakBytes: Int64
    public let time: TimeEstimate
    public let fitsNow: Bool
    /// "6.2 GB transformer + 0.6 GB working = 6.8 GB peak · about 75 s at 1024 × 1024"
    public let arithmetic: String
}

public struct ImageModelFitter: Sendable {
    public let hardware: HardwareProfile
    public let calibration: CalibrationStore

    public init(hardware: HardwareProfile, calibration: CalibrationStore) {
        self.hardware = hardware
        self.calibration = calibration
    }

    public static let engineID = "flux-schnell"
    /// Longer than this for one image and the grade says so.
    public static let comfortableSeconds: Double = 150

    /// Working memory beyond each phase's weights, measured on the M2 Max (2026-09-18) at
    /// 512², 768², 1024² and 1344×768 — see `ReferenceMachine.m2Max30Core32GB`. Flat for the
    /// encoders and the transformer (1.0–1.1 GB, most of it transient while the weights load,
    /// so it falls slightly as the image grows), linear in pixels for the VAE. A Mac that has
    /// run the model uses its own measurements instead.
    static let builtInWorkingBytes: [String: (fixed: Double, perUnit: Double)] = [
        "text-encode": (281_000_000, 0),
        "transformer": (1_115_000_000, 0),
        "vae-decode": (1_147_000_000, 5_647),
    ]

    public static func imageTokens(width: Int, height: Int) -> Int {
        (width / 16) * (height / 16)
    }

    /// The phases as the engine records them: (phase, weights, peak units, work units).
    static func phases(_ spec: ImageModelSpec, width: Int, height: Int, steps: Int)
        -> [(phase: String, weights: Int64, peakUnits: Double, workUnits: Double)] {
        let tokens = Double(imageTokens(width: width, height: height))
        let pixels = Double(width * height)
        return [
            ("text-encode", spec.textEncoderBytes, 1, 1),
            ("transformer", spec.transformerBytes, tokens, tokens * Double(steps)),
            ("vae-decode", spec.vaeBytes, pixels, pixels),
        ]
    }

    public func peakBytes(_ spec: ImageModelSpec, width: Int, height: Int) -> Int64 {
        Self.phases(spec, width: width, height: height, steps: 1).map { phase in
            let working = calibration.predictedActivation(
                engineID: Self.engineID, phase: phase.phase, machine: hardware.machineKey,
                peakUnits: phase.peakUnits, singleFrame: true
            )?.bytes ?? Self.builtInWorkingBytes[phase.phase].map { $0.fixed + $0.perUnit * phase.peakUnits } ?? 0
            return phase.weights + Int64(working)
        }.max() ?? 0
    }

    public func time(_ spec: ImageModelSpec, width: Int, height: Int, steps: Int) -> TimeEstimate {
        TimeEstimate.sum(Self.phases(spec, width: width, height: height, steps: steps).map {
            calibration.estimate(engineID: Self.engineID, phase: $0.phase, machine: hardware.machineKey,
                                 workUnits: $0.workUnits)
        })
    }

    public func fit(_ spec: ImageModelSpec, size: ImageSize = .square1024, steps: Int? = nil) -> ImageFit {
        fit(spec, width: size.width, height: size.height, steps: steps ?? spec.defaultSteps)
    }

    public func fit(_ spec: ImageModelSpec, width: Int, height: Int, steps: Int) -> ImageFit {
        let peak = peakBytes(spec, width: width, height: height)
        let time = time(spec, width: width, height: height, steps: steps)
        let usable = hardware.usableMemoryBytes
        let plannable = hardware.plannableMemoryBytes()
        let fitsNow = peak <= plannable
        let gigabytes = { (bytes: Int64) in Double(bytes) / 1_073_741_824 }

        let grade: ImageFit.Grade
        if peak > usable {
            grade = .red(because: String(format: "needs %.1f GB; this Mac's GPU can use %.1f GB",
                                         gigabytes(peak), hardware.usableMemoryGB))
        } else if let seconds = time.seconds, seconds > Self.comfortableSeconds {
            grade = .yellow(because: "about \(Self.duration(seconds)) an image")
        } else if !fitsNow {
            grade = .yellow(because: String(format: "needs %.1f GB and %.1f GB is free — close some apps",
                                            gigabytes(peak), gigabytes(plannable)))
        } else {
            grade = .green
        }

        let working = peak - spec.transformerBytes
        var arithmetic = String(format: "%.1f GB transformer + %.1f GB working = %.1f GB peak",
                                gigabytes(spec.transformerBytes), gigabytes(max(working, 0)), gigabytes(peak))
        arithmetic += " (the text encoders and VAE load only in their turn)"
        if let seconds = time.seconds {
            arithmetic += " · about \(Self.duration(seconds)) at \(width) × \(height)"
            if case .extrapolated(_, let chip, _) = time { arithmetic += ", estimated from \(chip)" }
        }
        return ImageFit(grade: grade, peakBytes: peak, time: time, fitsNow: fitsNow, arithmetic: arithmetic)
    }

    /// "40 s", "2 min", "4½ min".
    public static func duration(_ seconds: Double) -> String {
        if seconds < 90 { return "\(Int((seconds / 5).rounded() * 5)) s" }
        let halves = (seconds / 30).rounded()
        let whole = Int(halves / 2)
        return halves.truncatingRemainder(dividingBy: 2) == 0 ? "\(whole) min" : "\(whole)½ min"
    }
}
