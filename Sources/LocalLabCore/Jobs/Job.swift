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
        /// Whatever another job produced — the second half of a chain such as
        /// text-to-video → upscale. Resolved when this job starts, from that job's outcome.
        case outputOf(jobID: UUID)

        public var url: URL? {
            switch self {
            case .video(let video): video.url
            case .image(let url, _, _): url
            case .outputOf: nil
            }
        }

        /// What the resolver plans against. `nil` until an `.outputOf` source is resolved.
        public var resolverSource: SeedVR2Resolver.Source? {
            switch self {
            case .video(let video): .video(video)
            case .image(_, let width, let height): .image(width: width, height: height)
            case .outputOf: nil
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

/// What a text-to-video (or image-to-video) job was asked to do.
///
/// Defined ahead of its engine so the queue, chaining and persistence are built for it now;
/// no executor is registered for it yet (plan §6.6 — Wan, then H3).
public struct TextToVideoJobSpec: Codable, Sendable, Equatable {
    public let model: String
    public let prompt: String
    /// First frame, for image-to-video.
    public let startImage: URL?
    public let seconds: Double
    public let shortSide: Int
    public let seed: UInt64
    public let outputURL: URL

    public init(
        model: String, prompt: String, startImage: URL? = nil, seconds: Double,
        shortSide: Int, seed: UInt64 = 42, outputURL: URL
    ) {
        self.model = model
        self.prompt = prompt
        self.startImage = startImage
        self.seconds = seconds
        self.shortSide = shortSide
        self.seed = seed
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
        case textToVideo(TextToVideoJobSpec)

        /// Which engine runs it. Every kind shares one line: only one job of *any* kind runs
        /// at a time.
        public var engineID: String {
            switch self {
            case .upscale: "upscale"
            case .textToVideo: "text-to-video"
            }
        }

        public var displayName: String {
            switch self {
            case .upscale(let spec):
                if case .image = spec.source { return "Image upscale" }
                return "Video upscale"
            case .textToVideo(let spec):
                return spec.startImage == nil ? "Text to video" : "Image to video"
            }
        }

        public var outputURL: URL {
            switch self {
            case .upscale(let spec): spec.outputURL
            case .textToVideo(let spec): spec.outputURL
            }
        }
    }

    public enum State: String, Codable, Sendable {
        /// Waiting its turn (or waiting for the job it depends on).
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
    /// Queue position, assigned at enqueue and changed by reordering. Order comes from this,
    /// not from `createdAt`: jobs queued in quick succession can share a timestamp.
    public var sequence = 0
    public var title: String
    public var kind: Kind
    /// A job that must complete first — and whose output this one may use as its input.
    public var dependsOn: UUID?
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
    /// Saved access to input files outside the library, by path: in the sandbox a dropped
    /// file is readable only until the app quits, so a job that waits or resumes needs these
    /// to reach its input again. Nil in jobs saved before the sandbox.
    public var inputBookmarks: [String: Data]?

    public init(
        id: UUID = UUID(), title: String, kind: Kind, dependsOn: UUID? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.createdAt = createdAt
        self.title = title
        self.kind = kind
        self.dependsOn = dependsOn
        self.state = .queued
    }
}

extension Job {
    /// Files the job reads that the user provided — not ones the app made.
    public var inputURLs: [URL] {
        switch kind {
        case .upscale(let spec): spec.source.url.map { [$0] } ?? []
        case .textToVideo(let spec): spec.startImage.map { [$0] } ?? []
        }
    }

    /// The same job with every file URL passed through `map` — used when the library moves,
    /// so saved inputs, outputs and results follow it.
    public func remappingURLs(_ map: (URL) -> URL) -> Job {
        var job = self
        switch kind {
        case .upscale(let spec):
            let source: UpscaleJobSpec.Source
            switch spec.source {
            case .video(let video):
                source = .video(VideoMedia(
                    url: map(video.url), width: video.width, height: video.height, fps: video.fps,
                    frameCount: video.frameCount, hasAudio: video.hasAudio, hasAlpha: video.hasAlpha
                ))
            case .image(let url, let width, let height):
                source = .image(url: map(url), width: width, height: height)
            case .outputOf(let id):
                source = .outputOf(jobID: id)
            }
            job.kind = .upscale(UpscaleJobSpec(
                source: source, target: spec.target, quality: spec.quality,
                variant: spec.variant, outputURL: map(spec.outputURL)
            ))
        case .textToVideo(let spec):
            job.kind = .textToVideo(TextToVideoJobSpec(
                model: spec.model, prompt: spec.prompt, startImage: spec.startImage.map(map),
                seconds: spec.seconds, shortSide: spec.shortSide, seed: spec.seed,
                outputURL: map(spec.outputURL)
            ))
        }
        if let outcome {
            job.outcome = JobOutcome(outputURL: map(outcome.outputURL), seconds: outcome.seconds,
                                     peakBytes: outcome.peakBytes, samples: outcome.samples)
        }
        return job
    }
}

/// What an executor gets besides the job itself.
public struct JobContext: Sendable {
    /// Where this job's checkpoints live; anything already here is resumed from.
    public let workDirectory: URL
    /// Outputs of the jobs this one depends on, by job id.
    public let dependencyOutputs: [UUID: URL]

    public init(workDirectory: URL, dependencyOutputs: [UUID: URL] = [:]) {
        self.workDirectory = workDirectory
        self.dependencyOutputs = dependencyOutputs
    }
}

/// Runs one kind of job. The queue owns ordering, persistence, keep-awake and cancellation;
/// an executor only does the work, resuming from anything already in its work directory.
public protocol JobExecutor: Sendable {
    func run(
        _ job: Job,
        context: JobContext,
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
