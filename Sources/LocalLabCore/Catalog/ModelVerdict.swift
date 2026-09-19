import Foundation

/// How one catalog model fares on this Mac — Smart Fit's verdict, with its arithmetic
/// (plan §5.2: the dot is not the feature, the arithmetic is).
public struct ModelVerdict: Sendable, Equatable {
    public enum Grade: Sendable, Equatable {
        case green, yellow, red
        /// LocalLab can't run it yet; `fitsIfSupported` says whether memory would allow it.
        case notRunnable(fitsIfSupported: Bool?)
    }

    public let runner: Runner
    public let grade: Grade
    /// "Runs well here · ~41 tok/s" — the row's one line.
    public let headline: String
    /// "5.5 GB weights + 1.0 GB context (32k) + 1.0 GB runtime = 7.5 GB · ~41 tok/s"
    public let arithmetic: String
    /// The caveat, when the grade isn't green.
    public let caveat: String?
    /// Figures come from the search listing, not the model's own files — refined when opened.
    public let isEstimate: Bool
    public let weightBytes: Int64?

    public var sortRank: Int {
        switch grade {
        case .green: 0
        // A model whose size is unknown can't be rated yet: after the ones that can.
        case .yellow: weightBytes == nil ? 2 : 1
        case .red: 3
        case .notRunnable: 4
        }
    }
}

public enum ModelGrader {
    public static func verdict(
        for entry: CatalogEntry, details: CatalogDetails? = nil,
        hardware: HardwareProfile, calibration: CalibrationStore
    ) -> ModelVerdict {
        let runner = RunnerSupport.runner(for: entry)
        let weights = details?.weightBytes ?? entry.estimatedWeightBytes
        let isEstimate = details == nil

        switch runner {
        case .chat:
            guard let weights else {
                return ModelVerdict(runner: runner, grade: .yellow, headline: "Size unknown until opened",
                                    arithmetic: "", caveat: "The listing didn't say how big it is.",
                                    isEstimate: true, weightBytes: nil)
            }
            let spec = ChatCatalog.spec(forRepo: entry.repo)
                ?? .estimated(repo: entry.repo, weightBytes: weights, config: details?.config,
                              parameterCount: entry.parameterCount > 0 ? entry.parameterCount : nil,
                              license: entry.license, revision: details?.revision)
            let fit = ChatModelPicker(hardware: hardware, calibration: calibration).fit(spec)
            let grade: ModelVerdict.Grade
            let caveat: String?
            switch fit.grade {
            case .green: grade = .green; caveat = nil
            case .yellow(let because): grade = .yellow; caveat = because
            case .red(let because): grade = .red; caveat = because
            }
            let headline: String = switch fit.grade {
            case .green: String(format: "Runs well here · ~%.0f tok/s", fit.tokensPerSecond)
            case .yellow: String(format: "Runs, with a caveat · ~%.0f tok/s", fit.tokensPerSecond)
            case .red: "Too large for this Mac"
            }
            // Without config.json the context cache is a guess; say so.
            let exact = !isEstimate && details?.config != nil || spec.isCurated
            return ModelVerdict(runner: runner, grade: grade, headline: headline,
                                arithmetic: fit.arithmetic, caveat: caveat, isEstimate: !exact, weightBytes: weights)

        case .upscale(let variant):
            let capability = MachineProfile.upscaleVideo(hardware: hardware, calibration: calibration, installed: [variant])
            let grade: ModelVerdict.Grade = switch capability.status {
            case .ready: .green
            case .closeApps: .yellow
            case .tooLarge, .notBuilt: .red
            }
            return ModelVerdict(runner: runner, grade: grade,
                                headline: capability.status == .tooLarge ? capability.headline : "Upscales here — " + capability.example,
                                arithmetic: capability.detail, caveat: grade == .yellow ? "Close other apps first." : nil,
                                isEstimate: false, weightBytes: variant.downloadBytes)

        case .createImage(let spec):
            let fit = ImageModelFitter(hardware: hardware, calibration: calibration).fit(spec)
            let time = fit.time.seconds.map { "about " + ImageModelFitter.duration($0) + " an image" }
            let grade: ModelVerdict.Grade
            let caveat: String?
            let headline: String
            switch fit.grade {
            case .green:
                (grade, caveat, headline) = (.green, nil, "Creates images here" + (time.map { " · " + $0 } ?? ""))
            case .yellow(let because):
                (grade, caveat, headline) = (.yellow, because, "Creates images, with a caveat" + (time.map { " · " + $0 } ?? ""))
            case .red(let because):
                (grade, caveat, headline) = (.red, because, "Too large for this Mac")
            }
            return ModelVerdict(runner: runner, grade: grade, headline: headline, arithmetic: fit.arithmetic,
                                caveat: caveat, isEstimate: false, weightBytes: spec.downloadBytes)

        case .notYet(let reason):
            var fits: Bool?
            var arithmetic = ""
            if let weights {
                let peak = weights + ChatModelPicker.runtimeBytes
                fits = peak <= hardware.usableMemoryBytes
                arithmetic = String(format: "%.1f GB weights + %.1f GB runtime ≈ %.1f GB, against %.1f GB the GPU can use",
                                    Double(weights) / 1_073_741_824, Double(ChatModelPicker.runtimeBytes) / 1_073_741_824,
                                    Double(peak) / 1_073_741_824, hardware.usableMemoryGB)
            }
            return ModelVerdict(runner: runner, grade: .notRunnable(fitsIfSupported: fits),
                                headline: "LocalLab can't run this yet", arithmetic: arithmetic, caveat: reason,
                                isEstimate: isEstimate, weightBytes: weights)
        }
    }

    /// Order for "Best fit": what runs well here first; within a grade, the likely better
    /// model — the curated score where there is one, else size (larger is usually better
    /// within a generation) — then popularity. Downloads alone put 0.5B models on top.
    public static func bestFitOrder(_ a: (CatalogEntry, ModelVerdict), _ b: (CatalogEntry, ModelVerdict)) -> Bool {
        if a.1.sortRank != b.1.sortRank { return a.1.sortRank < b.1.sortRank }
        let qa = quality(a.0), qb = quality(b.0)
        if qa != qb { return qa > qb }
        return a.0.downloads > b.0.downloads
    }

    static func quality(_ entry: CatalogEntry) -> Int {
        if let curated = ChatCatalog.spec(forRepo: entry.repo) { return curated.qualityScore }
        guard entry.parameterCount > 0 else { return 0 }
        return Int(min(80, 25 + 12 * log2(max(Double(entry.parameterCount) / 1e9, 0.5)))) - 5
    }
}
