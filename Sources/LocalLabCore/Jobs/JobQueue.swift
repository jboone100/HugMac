import Foundation
import Observation

/// Long-running work that outlives the screen that started it. Plan §5.9.
///
/// - **One line for everything.** Each kind of job has its own engine, but only one job of
///   *any* kind runs at a time: these jobs each want the whole machine, and each is planned
///   as if it has it.
/// - **Chains.** A job can depend on another and take its output as input — text-to-video,
///   then upscale. It waits until that job completes; if it fails, it waits for a retry; if
///   it's cancelled, this one fails with the reason.
/// - **Yours to arrange.** Reorder what's waiting, hold a job, or pause the whole queue after
///   the current job to get the Mac back without losing anything.
/// - **Persistent and resumable.** `jobs/<id>/job.json`; a job running when the app died
///   comes back *interrupted*; Pause keeps checkpoints, Resume skips finished work.
/// - **Awake and heat-aware.** Held awake while running; a *critical* thermal state pauses
///   the running job, cooling to *fair* resumes it.
///
/// Model downloads are deliberately not in this line: they use the network and disk, not
/// the GPU, so they run alongside.
@MainActor
@Observable
public final class JobQueue {
    public private(set) var jobs: [Job] = []
    /// When true, nothing new starts; the running job (if any) finishes normally.
    public private(set) var isPaused = false
    /// This machine's measurements, updated as jobs finish.
    public private(set) var calibration: CalibrationStore
    /// Memory free the last time the queue was idle. While a job runs, previews plan against
    /// this rather than the reduced figure the running job leaves — a queued job gets the
    /// whole Mac when its turn comes.
    public private(set) var idleAvailableBytes: Int64?

    public let store: ModelStore
    private let executors: [String: JobExecutor]
    private let activity: ActivityHolding
    private let calibrationURL: URL?
    private let sampleAvailableMemory: @Sendable () -> Int64

    @ObservationIgnored private var runningID: UUID?
    @ObservationIgnored private var runningTask: Task<Void, Never>?
    @ObservationIgnored private var activityToken: NSObjectProtocol?
    @ObservationIgnored private var stopIntent: [UUID: Job.State] = [:]
    @ObservationIgnored private var sessionStart: (at: Date, fraction: Double)?
    @ObservationIgnored private var lastSaved = Date.distantPast
    @ObservationIgnored private var thermalObserver: NSObjectProtocol?
    @ObservationIgnored private var pausedForHeat: UUID?
    @ObservationIgnored private var finishedHandlers: [(Job) -> Void] = []
    @ObservationIgnored private var startGates: [@MainActor () async -> Void] = []

    public init(
        store: ModelStore,
        executors: [String: JobExecutor],
        activity: ActivityHolding = ProcessActivity(),
        calibration: CalibrationStore = CalibrationStore(),
        calibrationURL: URL? = nil,
        observeThermalState: Bool = false,
        sampleAvailableMemory: @Sendable @escaping () -> Int64 = { HardwareProfile.detect().availableMemoryBytes }
    ) {
        self.store = store
        self.executors = executors
        self.activity = activity
        self.calibration = calibration
        self.calibrationURL = calibrationURL
        self.sampleAvailableMemory = sampleAvailableMemory
        jobs = Self.loadJobs(from: store)
        isPaused = Self.loadSettings(from: store).paused
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

    /// One executor for every kind — convenient where only one engine exists, and in tests.
    public convenience init(
        store: ModelStore,
        executor: JobExecutor,
        activity: ActivityHolding = ProcessActivity(),
        calibration: CalibrationStore = CalibrationStore(),
        calibrationURL: URL? = nil,
        observeThermalState: Bool = false,
        sampleAvailableMemory: @Sendable @escaping () -> Int64 = { HardwareProfile.detect().availableMemoryBytes }
    ) {
        self.init(
            store: store,
            executors: ["upscale": executor, "text-to-video": executor, "create-image": executor],
            activity: activity, calibration: calibration, calibrationURL: calibrationURL,
            observeThermalState: observeThermalState, sampleAvailableMemory: sampleAvailableMemory
        )
    }

    /// Called when a job completes or fails.
    public func onFinished(_ handler: @escaping (Job) -> Void) {
        finishedHandlers.append(handler)
    }

    // MARK: - Queries

    public func job(_ id: UUID) -> Job? {
        jobs.first { $0.id == id }
    }

    public var running: Job? {
        runningID.flatMap(job)
    }

    /// Queued or running.
    /// Run before each job loads anything — how Chat hands the machine over (plan §5.12):
    /// an idle chat model is ejected, and a reply still streaming finishes first.
    public func beforeEachJob(_ gate: @escaping @MainActor () async -> Void) {
        startGates.append(gate)
    }

    /// Add measurements taken outside the queue — a chat reply's speed — through the one
    /// owner of `calibration.json`, so two writers never overwrite each other.
    public func recordMeasurements(_ samples: [CalibrationSample]) {
        guard !samples.isEmpty else { return }
        calibration.merge(samples)
        if let calibrationURL { try? calibration.save(to: calibrationURL) }
    }

    /// New first-run probe results (plan §5.14). Previews scale reference Macs' timings by
    /// them; a job's own plan reads them from disk when it starts.
    public func updateProbes(_ probes: ProbeStore) {
        calibration.probes = probes
    }

    /// True while a job holds the machine — the probes wait, since they'd measure the job.
    public var isRunningJob: Bool {
        jobs.contains { $0.state == .running }
    }

    public var activeCount: Int {
        jobs.filter { $0.state == .queued || $0.state == .running }.count
    }

    public var hasFinished: Bool {
        jobs.contains { $0.state == .completed || $0.state == .cancelled }
    }

    /// Whether an engine exists for this kind in this build.
    public func canRun(_ kind: Job.Kind) -> Bool {
        executors[kind.engineID] != nil
    }

    /// The job this one is waiting on, if it's waiting on one.
    public func blocker(for id: UUID) -> Job? {
        guard let dependency = job(id)?.dependsOn.flatMap(job),
              dependency.state != .completed else { return nil }
        return dependency
    }

    public func workDirectory(for id: UUID) -> URL {
        jobDirectory(id).appendingPathComponent("work", isDirectory: true)
    }

    /// Memory a preview should plan against: what's free now, or — while a job is running —
    /// what was free when the queue was last idle, whichever is larger.
    public func plannableAvailableBytes(current: Int64) -> Int64 {
        guard runningID != nil, let idle = idleAvailableBytes else { return current }
        return max(current, idle)
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

    // MARK: - Adding and arranging

    @discardableResult
    public func enqueue(title: String, kind: Job.Kind, dependsOn: UUID? = nil) -> Job {
        var job = Job(title: title, kind: kind, dependsOn: dependsOn)
        job.sequence = (jobs.map(\.sequence).max() ?? 0) + 1
        // Inputs outside the library are readable now; keep a way back to them for when the
        // job runs, perhaps after a relaunch.
        let library = store.baseDirectory.standardizedFileURL.path
        var bookmarks: [String: Data] = [:]
        for url in job.inputURLs where !url.standardizedFileURL.path.hasPrefix(library + "/") {
            if let bookmark = FileAccess.bookmark(url) { bookmarks[url.path] = bookmark }
        }
        job.inputBookmarks = bookmarks.isEmpty ? nil : bookmarks
        jobs.append(job)
        save(job)
        scheduleNext()
        return job
    }

    /// Reorder, by the positions of `waitingJobs`. Only jobs that haven't started move;
    /// finished and running jobs keep their places.
    public func moveWaiting(fromOffsets source: IndexSet, toOffset destination: Int) {
        var waiting = waitingJobs
        // `move(fromOffsets:toOffset:)` lives in SwiftUI, which Core doesn't import.
        let moving = source.sorted().map { waiting[$0] }
        let insertAt = destination - source.filter { $0 < destination }.count
        for offset in source.sorted(by: >) { waiting.remove(at: offset) }
        waiting.insert(contentsOf: moving, at: max(0, min(insertAt, waiting.count)))
        let slots = waiting.map(\.sequence).sorted()
        for (job, sequence) in zip(waiting, slots) {
            guard let index = jobs.firstIndex(where: { $0.id == job.id }) else { continue }
            jobs[index].sequence = sequence
            save(jobs[index])
        }
        jobs.sort { $0.sequence < $1.sequence }
    }

    /// Jobs that haven't started or are held, in the order they'll run.
    public var waitingJobs: [Job] {
        jobs.filter { $0.state == .queued || $0.state == .paused || $0.state == .interrupted }
            .sorted { $0.sequence < $1.sequence }
    }

    /// Stop starting new jobs. The running job finishes; nothing is lost.
    public func pauseQueue() {
        isPaused = true
        saveSettings()
    }

    public func resumeQueue() {
        isPaused = false
        saveSettings()
        scheduleNext()
    }

    /// Forget completed and cancelled jobs. Failed ones stay — they can still be resumed.
    public func clearFinished() {
        for job in jobs where job.state == .completed || job.state == .cancelled {
            // A job that others still depend on stays, so their input stays resolvable.
            let needed = jobs.contains { $0.dependsOn == job.id && !$0.state.isFinished }
            guard !needed else { continue }
            jobs.removeAll { $0.id == job.id }
            try? FileManager.default.removeItem(at: jobDirectory(job.id))
        }
    }

    // MARK: - Controlling one job

    /// Stop a job at its next checkpoint, keeping its work. A queued job is held.
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

    /// Put a paused, interrupted or failed job back in line.
    public func resume(_ id: UUID) {
        guard let index = jobs.firstIndex(where: { $0.id == id }),
              jobs[index].state.isResumable else { return }
        jobs[index].state = .queued
        jobs[index].note = nil
        jobs[index].failure = nil
        save(jobs[index])
        scheduleNext()
    }

    /// Stop for good and delete the job's checkpoints. Anything depending on it fails, since
    /// its input will never exist.
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
        failDependents(of: id, because: "the job it needed (\(jobs[index].title)) was cancelled")
        scheduleNext()
    }

    /// Forget a job that isn't running.
    public func remove(_ id: UUID) {
        guard let job = job(id), job.state != .running else { return }
        jobs.removeAll { $0.id == id }
        try? FileManager.default.removeItem(at: jobDirectory(id))
        failDependents(of: id, because: "the job it needed (\(job.title)) was removed")
        scheduleNext()
    }

    /// The app is quitting: record the running job as paused, not crashed.
    public func suspendForQuit() {
        guard let id = runningID, let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        jobs[index].state = .paused
        jobs[index].note = "Paused when LocalLab quit."
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

    /// The first queued job, in order, whose dependency (if any) has completed.
    private var nextRunnable: Job? {
        jobs.sorted { $0.sequence < $1.sequence }.first { job in
            guard job.state == .queued else { return false }
            guard let dependency = job.dependsOn else { return true }
            return self.job(dependency)?.state == .completed
        }
    }

    private func scheduleNext() {
        guard runningID == nil else { return }
        guard !isPaused, let next = nextRunnable else {
            releaseActivity()
            idleAvailableBytes = sampleAvailableMemory()
            return
        }
        idleAvailableBytes = sampleAvailableMemory()
        start(next.id)
    }

    private func failDependents(of id: UUID, because reason: String) {
        for index in jobs.indices where jobs[index].dependsOn == id && !jobs[index].state.isFinished
            && jobs[index].state != .running {
            jobs[index].state = .failed
            jobs[index].failure = "Can't run: \(reason)."
            jobs[index].finishedAt = Date()
            save(jobs[index])
        }
    }

    private func start(_ id: UUID) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        let job = jobs[index]
        guard let executor = executors[job.kind.engineID] else {
            jobs[index].state = .failed
            jobs[index].failure = "This version of LocalLab can't run \(job.kind.displayName.lowercased()) jobs yet."
            jobs[index].finishedAt = Date()
            save(jobs[index])
            scheduleNext()
            return
        }

        // Reach its inputs again: after a relaunch the sandbox has forgotten them. A moved or
        // renamed file is followed, and the job updated to its new path.
        if let bookmarks = jobs[index].inputBookmarks {
            var moved: [String: URL] = [:]
            for (path, bookmark) in bookmarks {
                guard let url = FileAccess.resolve(bookmark) else { continue }
                // Compare real paths: a resolved bookmark may spell the same file differently
                // (`/private/var/…` for `/var/…`), which isn't a move.
                if url.resolvingSymlinksInPath().path != URL(fileURLWithPath: path).resolvingSymlinksInPath().path {
                    moved[path] = url
                }
            }
            if !moved.isEmpty {
                jobs[index] = jobs[index].remappingURLs { moved[$0.path] ?? $0 }
            }
        }

        jobs[index].state = .running
        jobs[index].attempts += 1
        jobs[index].note = nil
        save(jobs[index])
        runningID = id
        sessionStart = (Date(), jobs[index].progress.fraction)
        if activityToken == nil {
            activityToken = activity.begin(reason: "LocalLab: \(job.title)")
        }

        var dependencyOutputs: [UUID: URL] = [:]
        if let dependency = job.dependsOn, let output = self.job(dependency)?.outcome?.outputURL {
            dependencyOutputs[dependency] = output
        }
        let context = JobContext(workDirectory: workDirectory(for: id), dependencyOutputs: dependencyOutputs)
        let runnable = jobs[index]
        let progress: @Sendable (StageProgress) -> Void = { update in
            Task { @MainActor [weak self] in self?.apply(update, to: id) }
        }

        let gates = startGates
        runningTask = Task { [weak self] in
            for gate in gates { await gate() }
            let started = Date()
            let result: Result<JobOutcome, Error>
            do {
                try Task.checkCancellation()
                try FileManager.default.createDirectory(at: context.workDirectory, withIntermediateDirectories: true)
                // Detached: model evaluation blocks its thread, and it must not be the main
                // one. Detached tasks don't inherit cancellation, so it's forwarded by hand.
                let task = Task.detached(priority: .userInitiated) {
                    try await executor.run(runnable, context: context, progress: progress)
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
            save(jobs[index])
            for handler in finishedHandlers { handler(jobs[index]) }
        case .failure(let error):
            switch intent {
            case .paused:
                jobs[index].state = .paused
                save(jobs[index])
            case .cancelled:
                jobs[index].state = .cancelled
                jobs[index].finishedAt = Date()
                save(jobs[index])
                try? FileManager.default.removeItem(at: workDirectory(for: id))
                failDependents(of: id, because: "the job it needed (\(jobs[index].title)) was cancelled")
            default:
                // A failure keeps its checkpoints: most failures are worth retrying from
                // where the job got to. Dependents wait for that retry rather than failing.
                jobs[index].state = .failed
                jobs[index].failure = error is CancellationError ? "Stopped." : String(describing: error)
                jobs[index].finishedAt = Date()
                save(jobs[index])
                for handler in finishedHandlers { handler(jobs[index]) }
            }
        }
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

    struct Settings: Codable { var paused = false }

    private static func settingsURL(_ store: ModelStore) -> URL {
        store.jobsDirectory.appendingPathComponent("queue.json")
    }

    static func loadSettings(from store: ModelStore) -> Settings {
        guard let data = try? Data(contentsOf: settingsURL(store)),
              let settings = try? JSONDecoder().decode(Settings.self, from: data) else { return Settings() }
        return settings
    }

    private func saveSettings() {
        try? FileManager.default.createDirectory(at: store.jobsDirectory, withIntermediateDirectories: true)
        try? JSONEncoder().encode(Settings(paused: isPaused)).write(to: Self.settingsURL(store), options: .atomic)
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

    // ISO-8601 with fractional seconds. Built per use and nonisolated: the library mover
    // rewrites job records off the main actor.
    nonisolated static var encoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(date.formatted(.iso8601.year().month().day()
                .time(includingFractionalSeconds: true).timeZone(separator: .omitted)))
        }
        return encoder
    }

    nonisolated static var decoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let text = try decoder.singleValueContainer().decode(String.self)
            let strategy = Date.ISO8601FormatStyle(includingFractionalSeconds: true)
            if let date = try? strategy.parse(text) { return date }
            return try Date.ISO8601FormatStyle().parse(text)
        }
        return decoder
    }
}
