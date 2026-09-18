import Foundation
import HugMacCore
import HugMacMLX
import Observation
import UserNotifications

/// App-lifetime state. Owned by the `App`, not by a window, so a job keeps running when its
/// window closes and a reopened window picks it back up.
@MainActor
@Observable
public final class AppModel {
    public let queue: JobQueue
    public let upscale: UpscaleModel
    public let machine: MachineModel

    public init(queue: JobQueue, upscale: UpscaleModel, machine: MachineModel) {
        self.queue = queue
        self.upscale = upscale
        self.machine = machine
    }

    /// The production wiring: the default library, the real installer and engine, this
    /// Mac's measurements, thermal observation, and completion notifications.
    public static func live() -> AppModel {
        let store = ModelStore()
        let calibrationURL = CalibrationStore.defaultURL()
        // One engine per kind of job, all sharing one line. Text-to-video has a job type but
        // no engine yet, so it isn't registered — the queue reports that rather than crash.
        let queue = JobQueue(
            store: store,
            executors: ["upscale": SeedVR2JobExecutor(store: store, calibrationURL: calibrationURL)],
            calibration: CalibrationStore.load(from: calibrationURL),
            calibrationURL: calibrationURL,
            observeThermalState: true
        )
        queue.onFinished { job in Notifier.jobFinished(job) }
        let upscale = UpscaleModel(store: store, installer: ModelInstaller(store: store), queue: queue)
        let machine = MachineModel(store: store, queue: queue, calibrationURL: calibrationURL)
        // A finished job adds measurements and may have installed a model: both change
        // what the profile says.
        queue.onFinished { _ in machine.refresh() }
        return AppModel(queue: queue, upscale: upscale, machine: machine)
    }
}

/// A local notification when a long job ends — you shouldn't have to keep checking.
@MainActor
enum Notifier {
    static func jobFinished(_ job: Job) {
        let title = job.state == .completed ? "Upscale finished" : "Upscale stopped"
        let body = job.title + (job.state == .completed
            ? (job.outcome.map { " — " + Format.duration($0.seconds) } ?? "")
            : (job.failure.map { " — " + $0 } ?? ""))
        let identifier = job.id.uuidString
        Task { @MainActor in
            let center = UNUserNotificationCenter.current()
            guard (try? await center.requestAuthorization(options: [.alert, .sound])) == true else { return }
            let content = UNMutableNotificationContent()
            content.title = title
            content.body = body
            content.sound = .default
            try? await center.add(UNNotificationRequest(identifier: identifier, content: content, trigger: nil))
        }
    }
}
