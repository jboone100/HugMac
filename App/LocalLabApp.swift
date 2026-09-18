import AppKit
import LocalLabUI
import SwiftUI

@main
struct LocalLabApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    /// App-lifetime, not window-lifetime: closing the window leaves jobs running, and a
    /// reopened window shows them where they are.
    @State private var app = AppModel.live()

    var body: some Scene {
        WindowGroup("LocalLab") {
            RootView(app: app, initialFile: DebugLaunch.initialFile)
                .frame(minWidth: 900, minHeight: 640)
                .background(DebugLaunch.Snapshotter())
                .task { delegate.app = app }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    @MainActor var app: AppModel?

    /// Quitting mid-job records it as paused (not crashed); its checkpoints are kept and it
    /// resumes from them next time.
    @MainActor
    func applicationWillTerminate(_ notification: Notification) {
        app?.queue.suspendForQuit()
    }
}
