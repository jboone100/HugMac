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
    private let lock = NSLock()
    private var _attempts: [Attempt] = []
    private var failuresRemaining: [UUID: Int] = [:]

    init(units: Int = 5, unitDelay: Duration = .milliseconds(30)) {
        self.units = units
        self.unitDelay = unitDelay
    }

    var attempts: [Attempt] { lock.withLock { _attempts } }

    func failNext(_ id: UUID) { lock.withLock { failuresRemaining[id, default: 0] += 1 } }

    func run(_ job: Job, workDirectory: URL,
             progress: @Sendable @escaping (StageProgress) -> Void) async throws -> JobOutcome {
        var firstComputed: Int?
        defer { lock.withLock { _attempts.append(Attempt(jobID: job.id, firstComputedUnit: firstComputed)) } }
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
        guard case .upscale(let spec) = job.kind else { throw StageError.cancelled }
        try Data("done".utf8).write(to: spec.outputURL)
        return JobOutcome(
            outputURL: spec.outputURL, seconds: 1, peakBytes: 1_000,
            samples: [CalibrationSample(engineID: SeedVR2Resolver.mlxEngineID, phase: "dit",
                                        workUnits: 10, seconds: 1, peakBytes: 1_000,
                                        chipName: "Apple M2 Max")]
        )
    }
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
        // Simulate a crash: the process is gone, job.json still says running.
        queue = nil

        let relaunched = JobQueue(store: store, executor: executor)
        #expect(relaunched.job(job.id)?.state == .interrupted)
        #expect((relaunched.job(job.id)?.progress.unitsDone ?? 0) >= 1, "progress survives the relaunch")
        relaunched.resume(job.id)
        await waitFor { relaunched.job(job.id)?.state == .completed }
        #expect(relaunched.job(job.id)?.state == .completed)
        #expect((executor.attempts.last?.firstComputedUnit ?? 0) >= 3)
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
