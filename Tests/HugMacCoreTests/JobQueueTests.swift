import CoreGraphics
import Foundation
import Testing
@testable import HugMacCore

// MARK: - Fakes

/// Does its "work" in units, checkpointing each one to the work directory exactly as the
/// real engine checkpoints chunks — so resume behaviour is exercised, not assumed.
final class FakeExecutor: JobExecutor, @unchecked Sendable {
    struct Attempt: Equatable {
        let jobID: UUID
        /// The first unit this attempt actually computed — 0 means it started from scratch.
        let firstComputedUnit: Int?
    }

    let units: Int
    let unitDelay: Duration
    /// Shared between executors, so concurrency is measured across kinds.
    let concurrency: ConcurrencyMeter
    private let lock = NSLock()
    private var _attempts: [Attempt] = []
    private var _inputs: [UUID: [UUID: URL]] = [:]
    private var failuresRemaining: [UUID: Int] = [:]

    init(units: Int = 5, unitDelay: Duration = .milliseconds(30), concurrency: ConcurrencyMeter = ConcurrencyMeter()) {
        self.units = units
        self.unitDelay = unitDelay
        self.concurrency = concurrency
    }

    var attempts: [Attempt] { lock.withLock { _attempts } }
    /// The dependency outputs each job was handed.
    func inputs(of id: UUID) -> [UUID: URL]? { lock.withLock { _inputs[id] } }

    func failNext(_ id: UUID) { lock.withLock { failuresRemaining[id, default: 0] += 1 } }

    func run(_ job: Job, context: JobContext,
             progress: @Sendable @escaping (StageProgress) -> Void) async throws -> JobOutcome {
        let workDirectory = context.workDirectory
        lock.withLock { _inputs[job.id] = context.dependencyOutputs }
        concurrency.enter()
        var firstComputed: Int?
        defer {
            concurrency.leave()
            lock.withLock { _attempts.append(Attempt(jobID: job.id, firstComputedUnit: firstComputed)) }
        }
        for unit in 0 ..< units {
            let marker = workDirectory.appendingPathComponent("unit-\(unit).done")
            if FileManager.default.fileExists(atPath: marker.path) { continue }
            try Task.checkCancellation()
            try await Task.sleep(for: unitDelay)
            let shouldFail = lock.withLock { () -> Bool in
                guard let remaining = failuresRemaining[job.id], remaining > 0, unit == 2 else { return false }
                failuresRemaining[job.id] = remaining - 1
                return true
            }
            if shouldFail { throw StageError.engineFailure(stage: "fake", detail: "simulated") }
            if firstComputed == nil { firstComputed = unit }
            FileManager.default.createFile(atPath: marker.path, contents: nil)
            progress(StageProgress(fraction: Double(unit + 1) / Double(units), phase: "vae-decode",
                                   unitsDone: unit + 1, unitsTotal: units))
        }
        let output = job.kind.outputURL
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("done".utf8).write(to: output)
        return JobOutcome(
            outputURL: output, seconds: 1, peakBytes: 1_000,
            samples: [CalibrationSample(engineID: SeedVR2Resolver.mlxEngineID, phase: "dit",
                                        workUnits: 10, seconds: 1, peakBytes: 1_000,
                                        chipName: "Apple M2 Max")]
        )
    }
}

/// The most jobs ever running at the same moment.
final class ConcurrencyMeter: @unchecked Sendable {
    private let lock = NSLock()
    private var current = 0
    private var _peak = 0
    var peak: Int { lock.withLock { _peak } }
    func enter() { lock.withLock { current += 1; _peak = max(_peak, current) } }
    func leave() { lock.withLock { current -= 1 } }
}

final class CountingActivity: ActivityHolding, @unchecked Sendable {
    private let lock = NSLock()
    private var held = 0
    private(set) var everHeld = false
    var isHeld: Bool { lock.withLock { held > 0 } }
    func begin(reason: String) -> NSObjectProtocol {
        lock.withLock { held += 1; everHeld = true }
        return NSObject()
    }
    func end(_ token: NSObjectProtocol) { lock.withLock { held -= 1 } }
}

@MainActor
func waitFor(_ condition: @MainActor () -> Bool, timeout: Double = 5) async {
    let deadline = Date().addingTimeInterval(timeout)
    while !condition(), Date() < deadline { try? await Task.sleep(for: .milliseconds(10)) }
}

func textToVideoKind(in store: ModelStore, name: String = "t2v") -> Job.Kind {
    .textToVideo(TextToVideoJobSpec(
        model: "Wan-AI/Wan2.1-T2V-1.3B", prompt: "a fox leaps over a log", seconds: 4,
        shortSide: 384, outputURL: store.outputsDirectory.appendingPathComponent("\(name).mp4")
    ))
}

func chainedUpscaleKind(in store: ModelStore, after jobID: UUID) -> Job.Kind {
    .upscale(UpscaleJobSpec(
        source: .outputOf(jobID: jobID), target: .scale(2), quality: .balanced, variant: .threeBInt8,
        outputURL: store.outputsDirectory.appendingPathComponent("chained-up.mp4")
    ))
}

func upscaleKind(in store: ModelStore, name: String = "clip") -> Job.Kind {
    let video = VideoMedia(url: URL(fileURLWithPath: "/tmp/\(name).mp4"), width: 64, height: 48,
                           fps: 24, frameCount: 9, hasAudio: false)
    return .upscale(UpscaleJobSpec(
        source: .video(video), target: .scale(2), quality: .balanced, variant: .threeBInt8,
        outputURL: store.outputsDirectory.appendingPathComponent("\(name)-out.mp4")
    ))
}

// MARK: - Tests

@Suite("Job queue")
@MainActor
struct JobQueueTests {

    func setUp() -> (ModelStore, () -> Void) {
        let store = temporaryStore()
        try? store.prepare()
        return (store, { try? FileManager.default.removeItem(at: store.baseDirectory) })
    }

    @Test("A job runs to completion, is saved, feeds calibration, and cleans up its work")
    func runsToCompletion() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let calibrationURL = store.baseDirectory.appendingPathComponent("calibration.json")
        let queue = JobQueue(store: store, executor: FakeExecutor(), calibrationURL: calibrationURL)
        let job = queue.enqueue(title: "clip", kind: upscaleKind(in: store))
        await waitFor { queue.job(job.id)?.state == .completed }

        let done = try #require(queue.job(job.id))
        #expect(done.state == .completed)
        #expect(done.progress.fraction == 1)
        #expect(done.outcome != nil)
        #expect(!FileManager.default.fileExists(atPath: queue.workDirectory(for: job.id).path))
        #expect(!queue.calibration.samples.isEmpty)
        #expect(FileManager.default.fileExists(atPath: calibrationURL.path))
        // It's on disk: a new queue over the same store sees it.
        #expect(JobQueue(store: store, executor: FakeExecutor()).job(job.id)?.state == .completed)
    }

    @Test("Jobs run one at a time, in the order they were queued")
    func serial() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let executor = FakeExecutor(units: 3)
        let queue = JobQueue(store: store, executor: executor)
        let first = queue.enqueue(title: "a", kind: upscaleKind(in: store, name: "a"))
        let second = queue.enqueue(title: "b", kind: upscaleKind(in: store, name: "b"))
        #expect(queue.job(first.id)?.state == .running)
        #expect(queue.job(second.id)?.state == .queued)
        await waitFor { queue.job(second.id)?.state == .completed }
        #expect(executor.attempts.map(\.jobID) == [first.id, second.id])
    }

    @Test("Pause keeps the work; resume skips what was already done")
    func pauseAndResume() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let executor = FakeExecutor(units: 6, unitDelay: .milliseconds(60))
        let queue = JobQueue(store: store, executor: executor)
        let job = queue.enqueue(title: "clip", kind: upscaleKind(in: store))
        await waitFor { (queue.job(job.id)?.progress.unitsDone ?? 0) >= 2 }
        queue.pause(job.id)
        await waitFor { queue.job(job.id)?.state == .paused }
        #expect(queue.job(job.id)?.state == .paused)
        #expect(FileManager.default.fileExists(atPath: queue.workDirectory(for: job.id).path))

        queue.resume(job.id)
        await waitFor { queue.job(job.id)?.state == .completed }
        #expect(queue.job(job.id)?.attempts == 2)
        let resumed = try #require(executor.attempts.last)
        #expect((resumed.firstComputedUnit ?? 0) >= 2, "the resumed attempt must skip checkpointed units")
    }

    @Test("Cancel discards the work and the next job starts")
    func cancel() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let queue = JobQueue(store: store, executor: FakeExecutor(units: 6, unitDelay: .milliseconds(60)))
        let first = queue.enqueue(title: "a", kind: upscaleKind(in: store, name: "a"))
        let second = queue.enqueue(title: "b", kind: upscaleKind(in: store, name: "b"))
        await waitFor { (queue.job(first.id)?.progress.unitsDone ?? 0) >= 1 }
        queue.cancel(first.id)
        await waitFor { queue.job(first.id)?.state == .cancelled }
        #expect(!FileManager.default.fileExists(atPath: queue.workDirectory(for: first.id).path))
        await waitFor { queue.job(second.id)?.state == .completed }
        #expect(queue.job(second.id)?.state == .completed)
    }

    @Test("A job running when the app died comes back interrupted, and resumes from its checkpoints")
    func survivesRelaunch() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let executor = FakeExecutor(units: 6, unitDelay: .milliseconds(60))
        var queue: JobQueue? = JobQueue(store: store, executor: executor)
        let job = try #require(queue?.enqueue(title: "clip", kind: upscaleKind(in: store)))
        await waitFor { (queue?.job(job.id)?.progress.unitsDone ?? 0) >= 3 }
        // Simulate a crash: job.json still says running. (Dropping the queue doesn't stop the
        // old work the way a real crash would, so the relaunch gets its own executor — its
        // attempts are then only the resumed run's, not interleaved with the abandoned one.)
        queue = nil

        let afterCrash = FakeExecutor(units: 6, unitDelay: .milliseconds(60))
        let relaunched = JobQueue(store: store, executor: afterCrash)
        #expect(relaunched.job(job.id)?.state == .interrupted)
        #expect((relaunched.job(job.id)?.progress.unitsDone ?? 0) >= 1, "progress survives the relaunch")
        relaunched.resume(job.id)
        await waitFor { relaunched.job(job.id)?.state == .completed }
        #expect(relaunched.job(job.id)?.state == .completed)
        let resumed = try #require(afterCrash.attempts.first)
        // Either it picked up at unit 3 or later, or every unit was already checkpointed —
        // never from the start.
        #expect(resumed.firstComputedUnit.map { $0 >= 3 } ?? true,
                "resumed at \(String(describing: resumed.firstComputedUnit)), expected ≥ 3")
    }

    @Test("A failure keeps its work so Resume continues, and the queue moves on")
    func failureIsResumable() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let executor = FakeExecutor(units: 5)
        let queue = JobQueue(store: store, executor: executor)
        let doomed = queue.enqueue(title: "a", kind: upscaleKind(in: store, name: "a"))
        executor.failNext(doomed.id)
        let next = queue.enqueue(title: "b", kind: upscaleKind(in: store, name: "b"))
        await waitFor { queue.job(next.id)?.state == .completed }
        #expect(queue.job(doomed.id)?.state == .failed)
        #expect(queue.job(doomed.id)?.failure?.contains("simulated") == true)

        queue.resume(doomed.id)
        await waitFor { queue.job(doomed.id)?.state == .completed }
        #expect(executor.attempts.last?.firstComputedUnit == 2, "resumed at the unit that failed")
    }

    @Test("The Mac is held awake only while a job runs")
    func keepAwake() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let activity = CountingActivity()
        let queue = JobQueue(store: store, executor: FakeExecutor(units: 3), activity: activity)
        #expect(!activity.isHeld)
        let job = queue.enqueue(title: "clip", kind: upscaleKind(in: store))
        #expect(activity.isHeld)
        await waitFor { queue.job(job.id)?.state == .completed }
        #expect(!activity.isHeld)
        #expect(activity.everHeld)
    }

    @Test("A critical thermal state pauses the running job; cooling down resumes it")
    func thermal() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let queue = JobQueue(store: store, executor: FakeExecutor(units: 8, unitDelay: .milliseconds(50)))
        let job = queue.enqueue(title: "clip", kind: upscaleKind(in: store))
        await waitFor { (queue.job(job.id)?.progress.unitsDone ?? 0) >= 1 }
        queue.thermalStateChanged(.critical)
        await waitFor { queue.job(job.id)?.state == .paused }
        #expect(queue.job(job.id)?.note?.contains("too hot") == true)
        queue.thermalStateChanged(.fair)
        await waitFor { queue.job(job.id)?.state == .completed }
        #expect(queue.job(job.id)?.state == .completed)
    }

    @Test("Quitting records the running job as paused, not crashed")
    func quit() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let queue = JobQueue(store: store, executor: FakeExecutor(units: 8, unitDelay: .milliseconds(50)))
        let job = queue.enqueue(title: "clip", kind: upscaleKind(in: store))
        await waitFor { (queue.job(job.id)?.progress.unitsDone ?? 0) >= 1 }
        queue.suspendForQuit()
        #expect(JobQueue.loadJobs(from: store).first { $0.id == job.id }?.state == .paused)
    }

    @Test("Remove forgets finished jobs and refuses running ones")
    func remove() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let queue = JobQueue(store: store, executor: FakeExecutor(units: 6, unitDelay: .milliseconds(50)))
        let job = queue.enqueue(title: "clip", kind: upscaleKind(in: store))
        queue.remove(job.id)
        #expect(queue.job(job.id) != nil, "a running job can't be removed")
        await waitFor { queue.job(job.id)?.state == .completed }
        queue.remove(job.id)
        #expect(queue.job(job.id) == nil)
        #expect(JobQueue.loadJobs(from: store).isEmpty)
    }

    @Test("Jobs queued in the same second keep their order across a relaunch")
    func orderSurvivesRelaunch() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let queue = JobQueue(store: store, executor: FakeExecutor(units: 50, unitDelay: .milliseconds(100)))
        let ids = (0 ..< 4).map { queue.enqueue(title: "\($0)", kind: upscaleKind(in: store, name: "\($0)")).id }
        queue.cancel(ids[0])
        let reloaded = JobQueue.loadJobs(from: store).map(\.id)
        #expect(reloaded == ids)
    }
}

// MARK: - Many kinds, one line

@Suite("Job queue: kinds, chains and arranging")
@MainActor
struct JobQueueArrangingTests {

    func setUp() -> (ModelStore, () -> Void) {
        let store = temporaryStore()
        try? store.prepare()
        return (store, { try? FileManager.default.removeItem(at: store.baseDirectory) })
    }

    @Test("Each kind goes to its own engine, but only one job of any kind runs at a time")
    func oneLineAcrossKinds() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let meter = ConcurrencyMeter()
        let upscaler = FakeExecutor(units: 3, concurrency: meter)
        let generator = FakeExecutor(units: 3, concurrency: meter)
        let queue = JobQueue(store: store, executors: ["upscale": upscaler, "text-to-video": generator])
        let a = queue.enqueue(title: "gen 1", kind: textToVideoKind(in: store, name: "g1"))
        let b = queue.enqueue(title: "up 1", kind: upscaleKind(in: store, name: "u1"))
        let c = queue.enqueue(title: "gen 2", kind: textToVideoKind(in: store, name: "g2"))
        await waitFor { [a, b, c].allSatisfy { queue.job($0.id)?.state == .completed } }

        #expect(generator.attempts.map(\.jobID) == [a.id, c.id])
        #expect(upscaler.attempts.map(\.jobID) == [b.id])
        #expect(meter.peak == 1, "two jobs ran at once")
    }

    @Test("A kind with no engine fails with a sentence, and the line moves on")
    func missingEngine() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let queue = JobQueue(store: store, executors: ["upscale": FakeExecutor(units: 2)])
        let orphan = queue.enqueue(title: "gen", kind: textToVideoKind(in: store))
        let next = queue.enqueue(title: "up", kind: upscaleKind(in: store))
        #expect(!queue.canRun(textToVideoKind(in: store)))
        await waitFor { queue.job(next.id)?.state == .completed }
        #expect(queue.job(orphan.id)?.state == .failed)
        #expect(queue.job(orphan.id)?.failure?.contains("can't run text to video") == true)
    }

    @Test("A chained upscale waits for the generation, then receives its output")
    func chain() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let executor = FakeExecutor(units: 3, unitDelay: .milliseconds(40))
        let queue = JobQueue(store: store, executor: executor)
        let generate = queue.enqueue(title: "gen", kind: textToVideoKind(in: store))
        let upscale = queue.enqueue(title: "up", kind: chainedUpscaleKind(in: store, after: generate.id),
                                    dependsOn: generate.id)
        #expect(queue.blocker(for: upscale.id)?.id == generate.id)
        await waitFor { queue.job(upscale.id)?.state == .completed }

        #expect(executor.attempts.map(\.jobID) == [generate.id, upscale.id])
        let handed = try #require(executor.inputs(of: upscale.id))
        #expect(handed[generate.id] == queue.job(generate.id)?.outcome?.outputURL)
        #expect(queue.blocker(for: upscale.id) == nil)
    }

    @Test("A blocked job doesn't hold up the jobs behind it")
    func noHeadOfLineBlocking() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let executor = FakeExecutor(units: 4, unitDelay: .milliseconds(40))
        let queue = JobQueue(store: store, executor: executor)
        let generate = queue.enqueue(title: "gen", kind: textToVideoKind(in: store))
        queue.pause(generate.id)   // hold it — never runs
        let upscale = queue.enqueue(title: "chained", kind: chainedUpscaleKind(in: store, after: generate.id),
                                    dependsOn: generate.id)
        let independent = queue.enqueue(title: "other", kind: upscaleKind(in: store, name: "other"))
        await waitFor { queue.job(independent.id)?.state == .completed }
        #expect(queue.job(independent.id)?.state == .completed)
        #expect(queue.job(upscale.id)?.state == .queued, "still waiting for the held generation")
    }

    @Test("If the generation fails, the chained job waits for a retry instead of failing")
    func chainSurvivesFailure() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let executor = FakeExecutor(units: 4)
        let queue = JobQueue(store: store, executor: executor)
        let generate = queue.enqueue(title: "gen", kind: textToVideoKind(in: store))
        executor.failNext(generate.id)
        let upscale = queue.enqueue(title: "up", kind: chainedUpscaleKind(in: store, after: generate.id),
                                    dependsOn: generate.id)
        await waitFor { queue.job(generate.id)?.state == .failed }
        try? await Task.sleep(for: .milliseconds(100))
        #expect(queue.job(upscale.id)?.state == .queued)
        #expect(queue.blocker(for: upscale.id)?.state == .failed)

        queue.resume(generate.id)
        await waitFor { queue.job(upscale.id)?.state == .completed }
        #expect(queue.job(upscale.id)?.state == .completed)
    }

    @Test("If the generation is cancelled, the chained job fails and says why")
    func chainCancelled() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let queue = JobQueue(store: store, executor: FakeExecutor(units: 8, unitDelay: .milliseconds(40)))
        let generate = queue.enqueue(title: "gen", kind: textToVideoKind(in: store))
        let upscale = queue.enqueue(title: "up", kind: chainedUpscaleKind(in: store, after: generate.id),
                                    dependsOn: generate.id)
        await waitFor { queue.job(generate.id)?.state == .running }
        queue.cancel(generate.id)
        await waitFor { queue.job(upscale.id)?.state == .failed }
        #expect(queue.job(upscale.id)?.failure?.contains("was cancelled") == true)
    }

    @Test("Reordering what's waiting changes what runs next")
    func reorder() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let executor = FakeExecutor(units: 3, unitDelay: .milliseconds(40))
        let queue = JobQueue(store: store, executor: executor)
        let first = queue.enqueue(title: "a", kind: upscaleKind(in: store, name: "a"))   // starts now
        let b = queue.enqueue(title: "b", kind: upscaleKind(in: store, name: "b"))
        let c = queue.enqueue(title: "c", kind: upscaleKind(in: store, name: "c"))
        // Move c ahead of b.
        queue.moveWaiting(fromOffsets: IndexSet(integer: 1), toOffset: 0)
        #expect(queue.waitingJobs.map(\.id) == [c.id, b.id])
        await waitFor { queue.job(b.id)?.state == .completed }
        #expect(executor.attempts.map(\.jobID) == [first.id, c.id, b.id])
        // The new order survives a relaunch.
        #expect(JobQueue.loadJobs(from: store).map(\.id) == [first.id, c.id, b.id])
    }

    @Test("Pausing the queue lets the current job finish and starts nothing new")
    func pauseQueue() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let queue = JobQueue(store: store, executor: FakeExecutor(units: 3, unitDelay: .milliseconds(40)))
        let current = queue.enqueue(title: "a", kind: upscaleKind(in: store, name: "a"))
        let next = queue.enqueue(title: "b", kind: upscaleKind(in: store, name: "b"))
        queue.pauseQueue()
        await waitFor { queue.job(current.id)?.state == .completed }
        try? await Task.sleep(for: .milliseconds(150))
        #expect(queue.job(current.id)?.state == .completed, "the running job isn't interrupted")
        #expect(queue.job(next.id)?.state == .queued, "nothing new starts while paused")

        // Paused survives a relaunch.
        let relaunched = JobQueue(store: store, executor: FakeExecutor(units: 3))
        #expect(relaunched.isPaused)
        #expect(relaunched.job(next.id)?.state == .queued)
        relaunched.resumeQueue()
        await waitFor { relaunched.job(next.id)?.state == .completed }
        #expect(relaunched.job(next.id)?.state == .completed)
    }

    @Test("Clear finished keeps failed jobs, and any output a waiting job still needs")
    func clearFinished() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let executor = FakeExecutor(units: 4)   // failNext triggers at unit 2
        let queue = JobQueue(store: store, executor: executor)
        let done = queue.enqueue(title: "done", kind: upscaleKind(in: store, name: "done"))
        let broken = queue.enqueue(title: "broken", kind: upscaleKind(in: store, name: "broken"))
        executor.failNext(broken.id)
        await waitFor { queue.job(broken.id)?.state == .failed }
        queue.pauseQueue()
        let needed = queue.enqueue(title: "gen", kind: textToVideoKind(in: store))
        queue.resumeQueue()
        await waitFor { queue.job(needed.id)?.state == .completed }
        queue.pauseQueue()
        let waiting = queue.enqueue(title: "up", kind: chainedUpscaleKind(in: store, after: needed.id),
                                    dependsOn: needed.id)

        queue.clearFinished()
        #expect(queue.job(done.id) == nil)
        #expect(queue.job(broken.id) != nil, "failed jobs can still be resumed")
        #expect(queue.job(needed.id) != nil, "a waiting job still needs its output")
        #expect(queue.job(waiting.id) != nil)
    }

    @Test("While a job runs, previews plan against the memory free when the Mac was idle")
    func previewMemoryWhileBusy() async throws {
        let (store, cleanUp) = setUp(); defer { cleanUp() }
        let idle: Int64 = 20 << 30
        let queue = JobQueue(store: store, executor: FakeExecutor(units: 6, unitDelay: .milliseconds(50)),
                             sampleAvailableMemory: { idle })
        #expect(queue.plannableAvailableBytes(current: 6 << 30) == 6 << 30, "idle: what's free now")
        let job = queue.enqueue(title: "a", kind: upscaleKind(in: store))
        #expect(queue.running?.id == job.id)
        #expect(queue.plannableAvailableBytes(current: 6 << 30) == idle,
                "busy: the running job's memory will be back when this one starts")
        queue.cancel(job.id)
    }
}

// MARK: - Segment joining

@Suite("Video segments")
struct SegmentTests {

    @Test("Segments join end to end without losing or duplicating frames")
    func concatenate() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("hugmac-segments-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        var segments: [URL] = []
        for (index, frames) in [8, 8, 5].enumerated() {
            let url = directory.appendingPathComponent("segment-\(index).mp4")
            let writer = try VideoIO.Writer(url: url, width: 64, height: 48, fps: 24)
            for _ in 0 ..< frames {
                let context = CGContext(data: nil, width: 64, height: 48, bitsPerComponent: 8, bytesPerRow: 0,
                                        space: CGColorSpaceCreateDeviceRGB(),
                                        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
                context?.setFillColor(CGColor(red: CGFloat(index) / 3, green: 0.5, blue: 0.5, alpha: 1))
                context?.fill(CGRect(x: 0, y: 0, width: 64, height: 48))
                if let image = context?.makeImage() { try writer.append(image) }
            }
            try await writer.finish()
            segments.append(url)
        }
        let joined = directory.appendingPathComponent("joined.mp4")
        try await VideoIO.concatenate(segments, to: joined)
        let probed = try await VideoIO.probe(joined)
        #expect(probed.frameCount == 21)
        let frames = try await VideoIO.readFrames(from: joined, startIndex: 0, count: 30, fps: probed.fps)
        #expect(frames.count == 21)
    }
}
