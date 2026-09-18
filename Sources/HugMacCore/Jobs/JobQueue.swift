import Foundation
import Observation

/// Long-running work that outlives the screen that started it. Plan §5.9.
///
/// - **Serial.** One job runs at a time: two video diffusion jobs don't fit together on
///   almost any Mac, and the resolver planned each one against the whole budget.
/// - **Persistent.** Every job is `jobs/<id>/job.json` on the storage root. A job that was
///   running when the app died comes back as *interrupted*, not lost.
/// - **Resumable.** Pause keeps the job's work directory; the executor resumes from the
///   checkpoints in it. Cancel deletes them.
/// - **Awake.** The Mac is held awake while a job runs — the ComfyUI baseline lost about
///   2.6 hours to what looks like the Mac sleeping mid-decode.
/// - **Heat-aware.** At a *critical* thermal state the running job pauses itself, and
///   resumes when the Mac cools to *fair*.
@MainActor
@Observable
public final class JobQueue {
    public private(set) var jobs: [Job] = []
    /// This machine's measurements, updated as jobs finish, so every plan made after a job
    /// is predicted from it.
    public private(set) var calibration: CalibrationStore
    /// Called when a job completes or fails: the app posts a notification, the Upscale
    /// screen re-plans with the measurements the job just recorded.
    @ObservationIgnored private var finishedHandlers: [(Job) -> Void] = []

    public func onFinished(_ handler: @escaping (Job) -> Void) {
        finishedHandlers.append(handler)
    }

    public let store: ModelStore
    private let executor: JobExecutor
    private let activity: ActivityHolding
    private let calibrationURL: URL?

    @ObservationIgnored private var runningID: UUID?
    @ObservationIgnored private var runningTask: Task<Void, Never>?
    @ObservationIgnored private var activityToken: NSObjectProtocol?
    /// Why the running task was stopped, if it was stopped on purpose.
    @ObservationIgnored private var stopIntent: [UUID: Job.State] = [:]
    @ObservationIgnored private var sessionStart: (at: Date, fraction: Double)?
    @ObservationIgnored private var lastSaved = Date.distantPast
    @ObservationIgnored private var thermalObserver: NSObjectProtocol?
    @ObservationIgnored private var pausedForHeat: UUID?

    public init(
        store: ModelStore,
        executor: JobExecutor,
        activity: ActivityHolding = ProcessActivity(),
        calibration: CalibrationStore = CalibrationStore(),
        calibrationURL: URL? = nil,
        observeThermalState: Bool = false
    ) {
        self.store = store
        self.executor = executor
        self.activity = activity
        self.calibration = calibration
        self.calibrationURL = calibrationURL
        jobs = Self.loadJobs(from: store)
        // Anything still marked running died with the last process.
        for index in jobs.indices where jobs[index].state == .running {
            jobs[index].state = .interrupted
            save(jobs[index])
        }
        if observeThermalState {
            thermalObserver = NotificationCenter.default.addObserver(
                forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main
            ) { [weak self] _ in
                let state = ProcessInfo.processInfo.thermalState
                MainActor.assumeIsolated { self?.thermalStateChanged(state) }
            }
        }
        scheduleNext()
    }

    // MARK: - Queries

    public func job(_ id: UUID) -> Job? {
        jobs.first { $0.id == id }
    }

    public var running: Job? {
        runningID.flatMap(job)
    }

    public var activeCount: Int {
        jobs.filter { $0.state == .queued || $0.state == .running }.count
    }

    public func workDirectory(for id: UUID) -> URL {
        jobDirectory(id).appendingPathComponent("work", isDirectory: true)
    }

    /// Time spent running, summed across resumes, including the current session.
    public func elapsedSeconds(for id: UUID, now: Date = Date()) -> Double {
        guard let job = job(id) else { return 0 }
        guard runningID == id, let start = sessionStart else { return job.activeSeconds }
        return job.activeSeconds + now.timeIntervalSince(start.at)
    }

    /// Seconds left on the running job, extrapolated from this session's own rate — a
    /// resumed job jumps forward past its checkpoints, so the rate from zero would lie.
    public func remainingSeconds(now: Date = Date()) -> Double? {
        guard let job = running, let start = sessionStart else { return nil }
        let done = job.progress.fraction - start.fraction
        let elapsed = now.timeIntervalSince(start.at)
        guard done > 0.02, elapsed > 5 else { return nil }
        return max((1 - job.progress.fraction) * elapsed / done, 0)
    }

    // MARK: - Actions

    @discardableResult
    public func enqueue(title: String, kind: Job.Kind) -> Job {
        var job = Job(title: title, kind: kind)
        job.sequence = (jobs.map(\.sequence).max() ?? 0) + 1
        jobs.append(job)
        save(job)
        scheduleNext()
        return job
    }

    /// Stop a job at its next checkpoint, keeping its work. A queued job just waits.
    public func pause(_ id: UUID, note: String? = nil) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        switch jobs[index].state {
        case .running:
            stopIntent[id] = .paused
            jobs[index].note = note
            runningTask?.cancel()
        case .queued:
            jobs[index].state = .paused
            jobs[index].note = note
            save(jobs[index])
        default:
            break
        }
    }

    /// Put a paused, interrupted or failed job back in line. It resumes from its
    /// checkpoints.
    public func resume(_ id: UUID) {
        guard let index = jobs.firstIndex(where: { $0.id == id }),
              jobs[index].state.isResumable else { return }
        jobs[index].state = .queued
        jobs[index].note = nil
        jobs[index].failure = nil
        save(jobs[index])
        scheduleNext()
    }

    /// Stop for good and delete the job's checkpoints. The record stays, marked cancelled.
    public func cancel(_ id: UUID) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        if jobs[index].state == .running {
            stopIntent[id] = .cancelled
            runningTask?.cancel()
            return
        }
        guard !jobs[index].state.isFinished else { return }
        jobs[index].state = .cancelled
        jobs[index].finishedAt = Date()
        save(jobs[index])
        try? FileManager.default.removeItem(at: workDirectory(for: id))
    }

    /// Forget a finished or stopped job. A running job has to be stopped first.
    public func remove(_ id: UUID) {
        guard let job = job(id), job.state != .running else { return }
        jobs.removeAll { $0.id == id }
        try? FileManager.default.removeItem(at: jobDirectory(id))
    }

    /// The app is quitting: record the running job as paused, so it reads as something the
    /// user can resume rather than something that crashed.
    public func suspendForQuit() {
        guard let id = runningID, let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        jobs[index].state = .paused
        jobs[index].note = "Paused when HugMac quit."
        save(jobs[index])
        stopIntent[id] = .paused
        runningTask?.cancel()
    }

    public func thermalStateChanged(_ state: ProcessInfo.ThermalState) {
        if state == .critical, let id = runningID {
            pausedForHeat = id
            pause(id, note: "Paused because the Mac is too hot. It resumes when it cools down.")
        } else if state == .nominal || state == .fair, let id = pausedForHeat {
            pausedForHeat = nil
            if job(id)?.state == .paused { resume(id) }
        }
    }

    // MARK: - Scheduling

    private func scheduleNext() {
        guard runningID == nil,
              let next = jobs.first(where: { $0.state == .queued }) else {
            releaseActivity()
            return
        }
        start(next.id)
    }

    private func start(_ id: UUID) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        jobs[index].state = .running
        jobs[index].attempts += 1
        jobs[index].note = nil
        save(jobs[index])
        runningID = id
        sessionStart = (Date(), jobs[index].progress.fraction)
        if activityToken == nil {
            activityToken = activity.begin(reason: "HugMac: \(jobs[index].title)")
        }

        let job = jobs[index]
        let executor = self.executor
        let work = workDirectory(for: id)
        let progress: @Sendable (StageProgress) -> Void = { update in
            Task { @MainActor [weak self] in self?.apply(update, to: id) }
        }

        runningTask = Task { [weak self] in
            let started = Date()
            let result: Result<JobOutcome, Error>
            do {
                try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
                // Detached: model evaluation blocks its thread, and it must not be the main
                // one. Detached tasks don't inherit cancellation, so it's forwarded by hand.
                let task = Task.detached(priority: .userInitiated) {
                    try await executor.run(job, workDirectory: work, progress: progress)
                }
                let outcome = try await withTaskCancellationHandler {
                    try await task.value
                } onCancel: {
                    task.cancel()
                }
                result = .success(outcome)
            } catch {
                result = .failure(error)
            }
            self?.finish(id, result: result, ranFor: Date().timeIntervalSince(started))
        }
    }

    private func apply(_ update: StageProgress, to id: UUID) {
        guard let index = jobs.firstIndex(where: { $0.id == id }),
              jobs[index].state == .running else { return }
        let phaseChanged = jobs[index].progress.phase != update.phase
        jobs[index].progress = Job.Progress(
            phase: update.phase, unitsDone: update.unitsDone,
            unitsTotal: update.unitsTotal, fraction: update.fraction
        )
        // Progress is persisted at phase changes and every few seconds — enough for a
        // relaunch to show where an interrupted job got to, without a write per frame.
        if phaseChanged || Date().timeIntervalSince(lastSaved) > 3 {
            save(jobs[index])
        }
    }

    private func finish(_ id: UUID, result: Result<JobOutcome, Error>, ranFor seconds: Double) {
        let intent = stopIntent.removeValue(forKey: id)
        runningID = nil
        runningTask = nil
        sessionStart = nil
        guard let index = jobs.firstIndex(where: { $0.id == id }) else {
            scheduleNext()
            return
        }
        jobs[index].activeSeconds += seconds

        switch result {
        case .success(let outcome):
            jobs[index].state = .completed
            jobs[index].outcome = outcome
            jobs[index].finishedAt = Date()
            jobs[index].progress.fraction = 1
            try? FileManager.default.removeItem(at: workDirectory(for: id))
            calibration.merge(outcome.samples)
            if let calibrationURL { try? calibration.save(to: calibrationURL) }
            for handler in finishedHandlers { handler(jobs[index]) }
        case .failure(let error):
            switch intent {
            case .paused:
                jobs[index].state = .paused
            case .cancelled:
                jobs[index].state = .cancelled
                jobs[index].finishedAt = Date()
                try? FileManager.default.removeItem(at: workDirectory(for: id))
            default:
                // A failure keeps its checkpoints: most failures (memory pressure, a full
                // disk) are worth retrying from where the job got to.
                jobs[index].state = .failed
                jobs[index].failure = error is CancellationError ? "Stopped." : String(describing: error)
                jobs[index].finishedAt = Date()
                for handler in finishedHandlers { handler(jobs[index]) }
            }
        }
        save(jobs[index])
        scheduleNext()
    }

    private func releaseActivity() {
        if let activityToken { activity.end(activityToken) }
        activityToken = nil
    }

    // MARK: - Persistence

    private func jobDirectory(_ id: UUID) -> URL {
        store.jobsDirectory.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    private func save(_ job: Job) {
        lastSaved = Date()
        let directory = jobDirectory(job.id)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Self.encoder.encode(job).write(
                to: directory.appendingPathComponent("job.json"), options: .atomic
            )
        } catch {
            // A job that can't be saved still runs; it just won't survive a relaunch.
        }
    }

    static func loadJobs(from store: ModelStore) -> [Job] {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: store.jobsDirectory, includingPropertiesForKeys: nil
        ) else { return [] }
        return entries
            .compactMap { try? Data(contentsOf: $0.appendingPathComponent("job.json")) }
            .compactMap { try? decoder.decode(Job.self, from: $0) }
            .sorted { ($0.sequence, $0.createdAt) < ($1.sequence, $1.createdAt) }
    }

    // ISO-8601 *with* fractional seconds: jobs are ordered by creation time, and two queued
    // in the same second must not swap places across a relaunch.
    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(date.formatted(.iso8601.year().month().day()
                .time(includingFractionalSeconds: true).timeZone(separator: .omitted)))
        }
        return encoder
    }()

    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let text = try decoder.singleValueContainer().decode(String.self)
            let strategy = Date.ISO8601FormatStyle(includingFractionalSeconds: true)
            if let date = try? strategy.parse(text) { return date }
            return try Date.ISO8601FormatStyle().parse(text)
        }
        return decoder
    }()
}
