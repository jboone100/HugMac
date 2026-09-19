import Foundation
import LocalLabCore
import LocalLabMLX
import Observation
import UserNotifications

/// App-lifetime state. Owned by the `App`, not by a window, so a job keeps running when its
/// window closes and a reopened window picks it back up.
/// The sidebar's destinations — app state, so one screen can send you to another (Browse's
/// "Open in Chat").
public enum AppSection: Hashable, Sendable {
    case browse, chat, createImage, machine, upscale, jobs
}

@MainActor
@Observable
public final class AppModel {
    public var section: AppSection?
    public let queue: JobQueue
    public let upscale: UpscaleModel
    public let machine: MachineModel
    public let chat: ChatModel
    public let storage: StorageModel
    public let browse: BrowseModel
    public let createImage: CreateImageModel

    public init(queue: JobQueue, upscale: UpscaleModel, machine: MachineModel, chat: ChatModel,
                storage: StorageModel, browse: BrowseModel, createImage: CreateImageModel) {
        self.queue = queue
        self.upscale = upscale
        self.machine = machine
        self.chat = chat
        self.storage = storage
        self.browse = browse
        self.createImage = createImage
    }

    /// The production wiring: the default library, the real installer and engine, this
    /// Mac's measurements, thermal observation, and completion notifications.
    public static func live() -> AppModel {
        // Before anything resolves a path: the library may still be under the old name.
        LegacyMigration.moveApplicationSupportIfNeeded()
        LegacyMigration.copySettingsIfNeeded()
        // The library the user chose — or, if its drive isn't connected, the default one,
        // with the missing one named rather than shown as empty.
        let library = LibraryLocation.resolve()
        let store = library.store
        let calibrationURL = CalibrationStore.defaultURL()
        // One engine per kind of job, all sharing one line. Text-to-video has a job type but
        // no engine yet, so it isn't registered — the queue reports that rather than crash.
        let queue = JobQueue(
            store: store,
            executors: [
                "upscale": SeedVR2JobExecutor(store: store, calibrationURL: calibrationURL),
                "create-image": FluxJobExecutor(store: store),
            ],
            calibration: CalibrationStore.load(from: calibrationURL),
            calibrationURL: calibrationURL,
            observeThermalState: true
        )
        queue.onFinished { job in Notifier.jobFinished(job) }
        let installer = ModelInstaller(store: store)
        let upscale = UpscaleModel(store: store, installer: installer, queue: queue)
        let machine = MachineModel(store: store, queue: queue, calibrationURL: calibrationURL)
        // A finished job adds measurements and may have installed a model: both change
        // what the profile says.
        queue.onFinished { _ in machine.refresh() }
        let chat = ChatModel(store: store, installer: installer, queue: queue, backend: MLXChatEngine())
        let router = AppRouter()
        let createImage = CreateImageModel(
            store: store, installer: installer, queue: queue,
            sendToUpscale: { url in
                router.app?.section = .upscale
                Task { await upscale.load(url) }
            }
        )
        let storage = StorageModel(
            resolution: library, installer: installer, queue: queue, chat: chat,
            busyReason: {
                if case .installing = upscale.modelState { return "The upscaler is downloading. Let it finish or pause it first." }
                if case .installing = createImage.modelState { return "The image model is downloading. Let it finish or stop it first." }
                return nil
            },
            libraryChanged: {
                chat.refresh()
                machine.refresh()
                await upscale.refreshModelState()
                await createImage.refreshModelState()
            }
        )
        let browse = BrowseModel(
            store: store, installer: installer, queue: queue,
            libraryChanged: {
                chat.refresh()
                machine.refresh()
                await upscale.refreshModelState()
                await createImage.refreshModelState()
                await storage.refresh()
            },
            openInChat: { repo in
                chat.select(.manual(repo: repo))
                router.app?.section = .chat
            },
            openCreateImage: {
                router.app?.section = .createImage
            }
        )
        let app = AppModel(queue: queue, upscale: upscale, machine: machine, chat: chat, storage: storage,
                           browse: browse, createImage: createImage)
        router.app = app
        return app
    }
}

/// Lets a screen built before the app model send you to another screen.
@MainActor
final class AppRouter {
    weak var app: AppModel?
}

/// A local notification when a long job ends — you shouldn't have to keep checking.
@MainActor
enum Notifier {
    static func jobFinished(_ job: Job) {
        let title = job.kind.displayName + (job.state == .completed ? " finished" : " stopped")
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
