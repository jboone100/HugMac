import LocalLabCore
import SwiftUI

/// The window: a sidebar of tasks and jobs, and the selection on the right.
public struct RootView: View {
    let app: AppModel
    private let initialFile: URL?
    private typealias Section = AppSection

    /// `initialFile` pre-loads the Upscale screen — used by debug runs, and later by
    /// "Open With".
    public init(app: AppModel, initialFile: URL? = nil) {
        self.app = app
        self.initialFile = initialFile
        // First launch opens on This Mac, where the first-run measurement shows its work.
        if app.section == nil {
            app.section = initialFile == nil && app.machine.needsMeasuring ? .machine : .upscale
        }
    }

    public var body: some View {
        NavigationSplitView {
            List(selection: Bindable(app).section) {
                SwiftUI.Section("Models") {
                    Label("Browse", systemImage: "square.grid.2x2")
                        .tag(Section.browse)
                }
                SwiftUI.Section("Tasks") {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right")
                        .tag(Section.chat)
                    Label("Upscale", systemImage: "arrow.up.left.and.arrow.down.right")
                        .tag(Section.upscale)
                }
                SwiftUI.Section("Mac") {
                    Label("This Mac", systemImage: "cpu")
                        .tag(Section.machine)
                }
                SwiftUI.Section("Activity") {
                    Label("Jobs", systemImage: "list.bullet.rectangle")
                        .badge(app.queue.activeCount)
                        .tag(Section.jobs)
                }
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 200)
        } detail: {
            VStack(spacing: 0) {
                if let missing = app.storage.unavailable {
                    HStack(spacing: 8) {
                        Image(systemName: "externaldrive.badge.exclamationmark")
                        Text("The library on “\(LibraryLocation.displayName(missing))” isn't connected — using the library on this Mac until it is.")
                        Spacer()
                        SettingsLink { Text("Storage…") }
                    }
                    .font(.callout)
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(Color.orange.opacity(0.15))
                }
                detail
            }
        }
        .task {
            if let initialFile { await app.upscale.load(initialFile) }
            #if DEBUG
            if !DebugAutoRun.isRequested { app.machine.measureIfNeeded() }
            await DebugAutoRun.runIfRequested(app)
            #else
            app.machine.measureIfNeeded()
            #endif
        }
    }

    @ViewBuilder private var detail: some View {
        switch app.section {
        case .jobs: JobsView(queue: app.queue)
        case .machine: MachineView(model: app.machine)
        case .chat: ChatView(model: app.chat)
        case .browse: BrowseView(model: app.browse)
        default: UpscaleView(model: app.upscale)
        }
    }
}

#if DEBUG
/// `LOCALLAB_AUTOSTART=1 LOCALLAB_RESULT=out.txt` — start the loaded upscale as soon as it can,
/// write a one-line result when its job ends, and quit. `LOCALLAB_RESUME_JOBS=1` resumes every
/// interrupted or paused job instead. Exercises the real engine inside the app bundle.
@MainActor
enum DebugAutoRun {
    /// Headless runs measure the engine, not the Mac — they skip the first-run probes.
    static var isRequested: Bool {
        let environment = ProcessInfo.processInfo.environment
        return environment["LOCALLAB_AUTOSTART"] == "1" || environment["LOCALLAB_RESUME_JOBS"] == "1"
    }

    static func runIfRequested(_ app: AppModel) async {
        let environment = ProcessInfo.processInfo.environment
        let resultURL = environment["LOCALLAB_RESULT"].map { URL(fileURLWithPath: $0) }

        if environment["LOCALLAB_RESUME_JOBS"] == "1" {
            for job in app.queue.jobs where job.state.isResumable { app.queue.resume(job.id) }
            while app.queue.activeCount > 0 { try? await Task.sleep(for: .milliseconds(250)) }
            let lines = app.queue.jobs.map { "\($0.state.rawValue) attempts=\($0.attempts) \($0.outcome?.outputURL.path ?? $0.failure ?? "")" }
            write(lines.joined(separator: "\n"), to: resultURL)
            exit(0)
        }

        guard environment["LOCALLAB_AUTOSTART"] == "1" else { return }
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
            let hold = Double(environment["LOCALLAB_HOLD_SECONDS"] ?? "") ?? 0
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
