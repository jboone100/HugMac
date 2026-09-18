import SwiftUI

/// The window: a sidebar of tasks, and the task on the right. One task so far.
public struct RootView: View {
    @State private var upscale = UpscaleModel.live()
    private let initialFile: URL?

    /// `initialFile` pre-loads the Upscale screen — used by debug snapshots, and later by
    /// "Open With".
    public init(initialFile: URL? = nil) {
        self.initialFile = initialFile
    }

    public var body: some View {
        NavigationSplitView {
            List {
                Section("Tasks") {
                    Label("Upscale", systemImage: "arrow.up.left.and.arrow.down.right")
                }
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 200)
        } detail: {
            UpscaleView(model: upscale)
        }
        .task {
            if let initialFile { await upscale.load(initialFile) }
            #if DEBUG
            await DebugAutoRun.runIfRequested(upscale)
            #endif
        }
    }
}

#if DEBUG
/// `HUGMAC_AUTOSTART=1 HUGMAC_RESULT=out.txt` — start the loaded upscale as soon as it can,
/// write a one-line result, and quit. Exercises the real engine inside the app bundle, where
/// MLX has to find its Metal kernels in the bundle rather than beside a SwiftPM binary.
@MainActor
enum DebugAutoRun {
    static func runIfRequested(_ model: UpscaleModel) async {
        let environment = ProcessInfo.processInfo.environment
        guard environment["HUGMAC_AUTOSTART"] == "1" else { return }
        let resultURL = environment["HUGMAC_RESULT"].map { URL(fileURLWithPath: $0) }
        await model.refreshModelState()
        guard model.canStart else {
            write("not-started: plan=\(model.plan != nil) model=\(model.modelState) refusal=\(model.refusal ?? "-")", to: resultURL)
            exit(2)
        }
        model.start()
        while model.isRunning { try? await Task.sleep(for: .milliseconds(200)) }
        if let outcome = model.outcome {
            write(String(format: "ok %@ %.1fs peak=%lld predicted=%lld",
                         outcome.outputURL.path, outcome.seconds, outcome.peakBytes,
                         model.plan?.peakBytes ?? 0), to: resultURL)
            exit(0)
        }
        write("failed: \(model.errorMessage ?? "unknown")", to: resultURL)
        exit(1)
    }

    static func write(_ line: String, to url: URL?) {
        print(line)
        if let url { try? (line + "\n").write(to: url, atomically: true, encoding: .utf8) }
    }
}
#endif
