import SwiftUI

/// The window: a sidebar of tasks and jobs, and the selection on the right.
public struct RootView: View {
    let app: AppModel
    private let initialFile: URL?
    @State private var selection: Section? = .upscale

    enum Section: Hashable { case upscale, jobs }

    /// `initialFile` pre-loads the Upscale screen — used by debug runs, and later by
    /// "Open With".
    public init(app: AppModel, initialFile: URL? = nil) {
        self.app = app
        self.initialFile = initialFile
    }

    public var body: some View {
        NavigationSplitView {
            List(selection: $selection) {
                SwiftUI.Section("Tasks") {
                    Label("Upscale", systemImage: "arrow.up.left.and.arrow.down.right")
                        .tag(Section.upscale)
                }
                SwiftUI.Section("Activity") {
                    Label("Jobs", systemImage: "list.bullet.rectangle")
                        .badge(app.queue.activeCount)
                        .tag(Section.jobs)
                }
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 200)
        } detail: {
            switch selection {
            case .jobs: JobsView(queue: app.queue)
            default: UpscaleView(model: app.upscale)
            }
        }
        .task {
            if let initialFile { await app.upscale.load(initialFile) }
            #if DEBUG
            await DebugAutoRun.runIfRequested(app)
            #endif
        }
    }
}

#if DEBUG
/// `HUGMAC_AUTOSTART=1 HUGMAC_RESULT=out.txt` — start the loaded upscale as soon as it can,
/// write a one-line result when its job ends, and quit. `HUGMAC_RESUME_JOBS=1` resumes every
/// interrupted or paused job instead. Exercises the real engine inside the app bundle.
@MainActor
enum DebugAutoRun {
    static func runIfRequested(_ app: AppModel) async {
        let environment = ProcessInfo.processInfo.environment
        let resultURL = environment["HUGMAC_RESULT"].map { URL(fileURLWithPath: $0) }

        if environment["HUGMAC_RESUME_JOBS"] == "1" {
            for job in app.queue.jobs where job.state.isResumable { app.queue.resume(job.id) }
            while app.queue.activeCount > 0 { try? await Task.sleep(for: .milliseconds(250)) }
            let lines = app.queue.jobs.map { "\($0.state.rawValue) attempts=\($0.attempts) \($0.outcome?.outputURL.path ?? $0.failure ?? "")" }
            write(lines.joined(separator: "\n"), to: resultURL)
            exit(0)
        }

        guard environment["HUGMAC_AUTOSTART"] == "1" else { return }
        let model = app.upscale
        await model.refreshModelState()
        guard model.canStart else {
            write("not-started: plan=\(model.plan != nil) model=\(model.modelState) refusal=\(model.refusal ?? "-")", to: resultURL)
            exit(2)
        }
        model.start()
        while model.isRunning { try? await Task.sleep(for: .milliseconds(200)) }
        if let outcome = model.outcome {
            // Let the Done panel draw its player before reporting — the step that crashed
            // with SwiftUI's VideoPlayer.
            let hold = Double(environment["HUGMAC_HOLD_SECONDS"] ?? "") ?? 0
            if hold > 0 { try? await Task.sleep(for: .seconds(hold)) }
            write(String(format: "ok %@ %.1fs peak=%lld", outcome.outputURL.path, outcome.seconds, outcome.peakBytes), to: resultURL)
            exit(0)
        }
        write("failed: \(model.errorMessage ?? "unknown")", to: resultURL)
        exit(1)
    }

    static func write(_ text: String, to url: URL?) {
        print(text)
        if let url { try? (text + "\n").write(to: url, atomically: true, encoding: .utf8) }
    }
}
#endif
