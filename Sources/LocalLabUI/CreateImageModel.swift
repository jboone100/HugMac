import Foundation
import LocalLabCore
import Observation

/// The Create Image screen's state and actions (plan §7.1 #3).
///
/// A prompt, a size, and Smart Fit's verdict for that size on this Mac. Each image is a job on
/// the app's queue — one at a time with upscales and everything else — so it keeps going when
/// the window closes, and its result can go straight on to Upscale.
@MainActor
@Observable
public final class CreateImageModel {
    public enum ModelState: Equatable {
        case checking
        case notInstalled
        case installing(fraction: Double, detail: String)
        case installed
        case failed(String)
    }

    public let spec: ImageModelSpec
    public let queue: JobQueue
    private let store: ModelStore
    private let installer: ModelInstaller
    private let defaults: UserDefaults
    private let detectHardware: @MainActor () -> HardwareProfile
    private let sendToUpscale: @MainActor (URL) -> Void

    public var prompt = ""
    public var size: ImageSize {
        didSet {
            defaults.set(size.rawValue, forKey: Self.sizeKey)
            refit()
        }
    }
    /// Blank for a new random seed each time; a number to repeat an image exactly.
    public var seedText = ""

    public private(set) var modelState: ModelState = .checking
    public private(set) var fit: ImageFit?
    public private(set) var errorMessage: String?
    /// The job this screen started most recently, for its progress row.
    public private(set) var currentJobID: UUID?

    static let sizeKey = "LocalLab.createImageSize"

    public init(
        store: ModelStore, installer: ModelInstaller, queue: JobQueue,
        spec: ImageModelSpec = ImageModelCatalog.fluxSchnell4bit,
        defaults: UserDefaults = .standard,
        detectHardware: @escaping @MainActor () -> HardwareProfile = { HardwareProfile.detect() },
        sendToUpscale: @escaping @MainActor (URL) -> Void = { _ in }
    ) {
        self.store = store
        self.installer = installer
        self.queue = queue
        self.spec = spec
        self.defaults = defaults
        self.detectHardware = detectHardware
        self.sendToUpscale = sendToUpscale
        size = defaults.string(forKey: Self.sizeKey).flatMap(ImageSize.init(rawValue:)) ?? .square1024
        refit()
        // A finished job may have measured this Mac: the estimate improves.
        queue.onFinished { [weak self] _ in self?.refit() }
    }

    // MARK: - Derived

    public var currentJob: Job? { currentJobID.flatMap(queue.job) }

    /// Every image made here, newest first — the queue keeps them until Jobs is cleared.
    public var results: [Job] {
        queue.jobs
            .filter { if case .createImage = $0.kind { $0.state == .completed } else { false } }
            .sorted { ($0.finishedAt ?? $0.createdAt) > ($1.finishedAt ?? $1.createdAt) }
    }

    public var trimmedPrompt: String {
        prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public var seed: UInt64? {
        UInt64(seedText.trimmingCharacters(in: .whitespaces))
    }

    public var seedIsValid: Bool {
        seedText.trimmingCharacters(in: .whitespaces).isEmpty || seed != nil
    }

    /// Why Create is unavailable, or nil.
    public var startBlocker: String? {
        if modelState != .installed { return "Install the model first." }
        if trimmedPrompt.isEmpty { return "Describe the image you want." }
        if !seedIsValid { return "The seed must be a whole number, or blank for a random one." }
        if case .red(let because) = fit?.grade { return "Too large for this Mac — \(because)." }
        return nil
    }

    public var willQueue: Bool { queue.running != nil }

    // MARK: - Actions

    public func refit() {
        fit = ImageModelFitter(hardware: detectHardware(), calibration: queue.calibration).fit(spec, size: size)
    }

    public func refreshModelState() async {
        if case .installing = modelState { return }
        modelState = await installer.isInstalled(spec.repo) ? .installed : .notInstalled
    }

    public func installModel() async {
        errorMessage = nil
        modelState = .installing(fraction: 0, detail: "Resolving…")
        do {
            try await installer.install(spec.repo, manifest: spec.manifest) { [weak self] progress in
                Task { @MainActor [weak self] in
                    guard let self, case .installing = self.modelState else { return }
                    let detail = String(format: "%.1f of %.1f GB",
                                        Double(progress.bytesDone) / 1_073_741_824,
                                        Double(progress.bytesTotal) / 1_073_741_824)
                    self.modelState = .installing(fraction: progress.fraction, detail: detail)
                }
            }
            modelState = .installed
        } catch is CancellationError {
            modelState = .notInstalled
        } catch {
            modelState = .failed(String(describing: error))
        }
    }

    public func cancelInstall() async {
        await installer.cancel(spec.repo)
        modelState = .notInstalled
    }

    /// Queue an image. Pressing again while one waits queues another — a new seed each time
    /// unless one is set.
    public func start() {
        guard startBlocker == nil else { return }
        errorMessage = nil
        let spec = ImageJobSpec(
            model: self.spec.repo, prompt: trimmedPrompt, width: size.width, height: size.height,
            steps: self.spec.defaultSteps, seed: seed ?? UInt64.random(in: 0 ... 999_999_999),
            outputURL: outputURL(for: trimmedPrompt)
        )
        currentJobID = queue.enqueue(title: Self.title(for: trimmedPrompt), kind: .createImage(spec)).id
    }

    public func cancel() {
        if let currentJobID { queue.cancel(currentJobID) }
    }

    /// Put a result's prompt and seed back, to vary it or make it again.
    public func reuse(_ job: Job) {
        guard case .createImage(let spec) = job.kind else { return }
        prompt = spec.prompt
        seedText = String(spec.seed)
        if let match = ImageSize.allCases.first(where: { $0.width == spec.width && $0.height == spec.height }) {
            size = match
        }
    }

    public func upscale(_ job: Job) {
        guard let url = job.outcome?.outputURL else { return }
        sendToUpscale(url)
    }

    public func remove(_ job: Job) {
        queue.remove(job.id)
        if currentJobID == job.id { currentJobID = nil }
    }

    // MARK: - Naming

    /// "A red fox in the snow…" — the prompt's opening, for the Jobs list.
    static func title(for prompt: String) -> String {
        let words = prompt.split(whereSeparator: \.isWhitespace)
        let opening = words.prefix(8).joined(separator: " ")
        return words.count > 8 ? opening + "…" : opening
    }

    /// `<first words of the prompt>.png` in the outputs folder, never over an earlier file or
    /// one a waiting job has claimed.
    func outputURL(for prompt: String) -> URL {
        let allowed = CharacterSet.alphanumerics.union(.whitespaces)
        let cleaned = String(prompt.unicodeScalars.filter { allowed.contains($0) })
        let words = cleaned.split(whereSeparator: \.isWhitespace).prefix(6)
        let stem = words.isEmpty ? "image" : words.joined(separator: "-").lowercased()
        let claimed = Set(queue.jobs.compactMap { $0.state.isFinished ? nil : $0.kind.outputURL.path })
        return ModelStore.uniqueOutputURL(in: store.outputsDirectory, stem: stem, extension: "png", reserved: claimed)
    }
}
