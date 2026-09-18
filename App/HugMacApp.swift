import HugMacUI
import SwiftUI

@main
struct HugMacApp: App {
    var body: some Scene {
        WindowGroup("HugMac") {
            RootView(initialFile: DebugLaunch.initialFile)
                .frame(minWidth: 900, minHeight: 640)
                .background(DebugLaunch.Snapshotter())
        }
    }
}
