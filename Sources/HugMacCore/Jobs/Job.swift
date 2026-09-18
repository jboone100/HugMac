import Foundation

/// What an upscale job was asked to do.
///
/// It holds the *request*, not a plan. A job queued behind another would otherwise be
/// planned against memory measured while that other job was running — and be given a cramped
/// plan, or refused, for a turn in which it will have the whole Mac. The executor plans when
/// the job actually starts, and saves that plan beside its checkpoints so a resume reuses the
/// exact chunking the checkpoints were made with.
public struct UpscaleJobSpec: Codable, Sendable, Equatable {
    public enum Source: Codable, Sendable, Equatable {
        case video(VideoMedia)
        case image(url: URL, width: Int, height: Int)

        public var url: URL {
            switch self {
            case .video(let video): video.url
            case .image(let url, _, _): url
            }
        }

        public var resolverSource: SeedVR2Resolver.Source {
            switch self {
            case .video(let video): .video(video)
            case .image(_, let width, let height): .image(width: width, height: height)
            }
        }
    }

    public let source: Source
    public let target: UpscaleTarget
    public let quality: QualityPreset
    public let variant: SeedVR2Variant
    public let outputURL: URL

    public init(
        source: Source, target: UpscaleTarget, quality: QualityPreset,
        variant: SeedVR2Variant, outputURL: URL
    ) {
        self.source = source
        self.target = target
        self.quality = quality
        self.variant = variant
        self.outputURL = outputURL
    }
}

/// What a finished job produced and measured.
public struct JobOutcome: Codable, Sendable, Equatable {
    public let outputURL: URL
    public let seconds: Double
    public let peakBytes: Int64
    public let samples: [CalibrationSample]

    public init(outputURL: URL, seconds: Double, peakBytes: Int64, samples: [CalibrationSample]) {
        self.outputURL = outputURL
        self.seconds = seconds
        self.peakBytes = peakBytes
        self.samples = samples
    }
}

/// A unit of long-running work, persisted to `jobs/<id>/job.json`. Plan §5.9.
public struct Job: Codable, Sendable, Equatable, Identifiable {
    public enum Kind: Codable, Sendable, Equatable {
        case upscale(UpscaleJobSpec)
    }

    public enum State: String, Codable, Sendable {
        /// Waiting its turn.
        case queued
        case running
        /// Stopped on purpose — by the user, or by the queue (the Mac got too hot, the app
        /// quit). Checkpoints kept; resumes where it stopped.
        case paused
        /// Was running when the app died. Checkpoints kept; resumes where it stopped.
        case interrupted
        case completed
        case failed
        /// Stopped for good; checkpoints deleted.
        case cancelled

        public var isFinished: Bool { self == .completed || self == .failed || self == .cancelled }
        public var isResumable: Bool { self == .paused || self == .interrupted || self == .failed }
    }

    public struct Progress: Codable, Sendable, Equatable {
        public var phase: String = ""
        public var unitsDone: Int = 0
        public var unitsTotal: Int = 0
        public var fraction: Double = 0
    }

    public let id: UUID
    public let createdAt: Date
    /// Queue position, assigned at enqueue. Order comes from this, not from `createdAt`:
    /// jobs queued in quick succession can share a timestamp.
    public var sequence = 0
    public var title: String
    public var kind: Kind
    public var state: State
    public var progress = Progress()
    /// Time actually spent running, summed across resumes.
    public var activeSeconds: Double = 0
    public var finishedAt: Date?
    public var outcome: JobOutcome?
    public var failure: String?
    /// Why a job is paused, when the queue paused it rather than the user.
    public var note: String?
    /// How many times this job has been started — more than one means it resumed.
    public var attempts = 0

    public init(id: UUID = UUID(), title: String, kind: Kind, createdAt: Date = Date()) {
        self.id = id
        self.createdAt = createdAt
        self.title = title
        self.kind = kind
        self.state = .queued
    }
}

/// Runs one kind of job. The queue owns ordering, persistence, keep-awake and cancellation;
/// an executor only does the work, resuming from anything already in `workDirectory`.
public protocol JobExecutor: Sendable {
    func run(
        _ job: Job,
        workDirectory: URL,
        progress: @Sendable @escaping (StageProgress) -> Void
    ) async throws -> JobOutcome
}

/// Holding the Mac awake while work runs. A protocol so tests can check it's released.
public protocol ActivityHolding: Sendable {
    func begin(reason: String) -> NSObjectProtocol
    func end(_ token: NSObjectProtocol)
}

public struct ProcessActivity: ActivityHolding {
    public init() {}

    public func begin(reason: String) -> NSObjectProtocol {
        ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .idleSystemSleepDisabled], reason: reason
        )
    }

    public func end(_ token: NSObjectProtocol) {
        ProcessInfo.processInfo.endActivity(token)
    }
}
