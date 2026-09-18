import AppKit
import SwiftUI

/// Debug-only launch hooks, so the real window can be checked without Screen Recording
/// permission or a human at the keyboard:
///
///     HUGMAC_OPEN=clip.mp4            pre-load a file into Upscale
///     HUGMAC_SNAPSHOT=out.png         render the window to a PNG, then quit
///     HUGMAC_SNAPSHOT_SIZE=1100x1500  window size for the snapshot
///     HUGMAC_SNAPSHOT_DELAY=4         seconds to wait for the plan to settle
///     HUGMAC_EXPAND_REASONS=1         open "Why these settings"
///
/// The app writes its window id to `<snapshot>.windowid` and stays open for
/// `HUGMAC_SNAPSHOT_HOLD` seconds; `scripts/snapshot.sh` captures it with `screencapture -l`.
enum DebugLaunch {
    static var initialFile: URL? {
        #if DEBUG
        ProcessInfo.processInfo.environment["HUGMAC_OPEN"].map { URL(fileURLWithPath: $0) }
        #else
        nil
        #endif
    }

    struct Snapshotter: NSViewRepresentable {
        func makeNSView(context: Context) -> NSView {
            let view = NSView()
            #if DEBUG
            let environment = ProcessInfo.processInfo.environment
            if let path = environment["HUGMAC_SNAPSHOT"] {
                let delay = Double(environment["HUGMAC_SNAPSHOT_DELAY"] ?? "") ?? 4
                let size = environment["HUGMAC_SNAPSHOT_SIZE"].flatMap(Self.parseSize)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    guard let window = view.window else { return }
                    // An occluded or never-shown window is never rendered, so bring it
                    // on screen before waiting for the content to settle.
                    NSApp.setActivationPolicy(.regular)
                    NSApp.activate(ignoringOtherApps: true)
                    window.makeKeyAndOrderFront(nil)
                    if let size { window.setContentSize(size) }
                    DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                        // On current macOS, SwiftUI draws through a path that neither
                        // `cacheDisplay` nor `CALayer.render(in:)` can read back — both give
                        // a blank window. So publish the window id and let the window server
                        // capture it (`screencapture -l`), then quit once that's done.
                        let idFile = URL(fileURLWithPath: path).appendingPathExtension("windowid")
                        try? String(window.windowNumber).write(to: idFile, atomically: true, encoding: .utf8)
                        let hold = Double(environment["HUGMAC_SNAPSHOT_HOLD"] ?? "") ?? 6
                        DispatchQueue.main.asyncAfter(deadline: .now() + hold) {
                            NSApp.terminate(nil)
                        }
                    }
                }
            }
            #endif
            return view
        }

        func updateNSView(_ nsView: NSView, context: Context) {}

        static func parseSize(_ text: String) -> NSSize? {
            let parts = text.split(separator: "x").compactMap { Double($0) }
            return parts.count == 2 ? NSSize(width: parts[0], height: parts[1]) : nil
        }
    }
}
