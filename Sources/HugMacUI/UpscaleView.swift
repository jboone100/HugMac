import AVKit
import HugMacCore
import SwiftUI
import UniformTypeIdentifiers

/// Upscale: a file, an output size, a quality — and a plan that shows its working.
public struct UpscaleView: View {
    @Bindable var model: UpscaleModel
    @State private var isTargeted = false
    @State private var isImporting = false

    public init(model: UpscaleModel) {
        self.model = model
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                inputSection
                if model.isBatch {
                    optionsSection
                    BatchList(model: model)
                } else if model.input != nil {
                    optionsSection
                    PlanCard(model: model)
                    runSection
                }
                if let message = model.batchMessage {
                    Label(message, systemImage: "tray.and.arrow.down.fill").foregroundStyle(.secondary)
                }
                if let outcome = model.outcome {
                    ResultPanel(outcome: outcome, plan: model.plan, isVideo: model.input?.isVideo ?? true)
                }
                if let message = model.errorMessage {
                    Label(message, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                }
            }
            .padding(24)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .fileImporter(isPresented: $isImporting, allowedContentTypes: [.movie, .image],
                      allowsMultipleSelection: true) { result in
            if case .success(let urls) = result {
                Task { await model.load(urls) }
            }
        }
        .task { await model.refreshModelState() }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Upscale").font(.largeTitle.weight(.semibold))
            Text("SeedVR2 — runs entirely on this Mac. Every setting is chosen for this machine; open *Why these settings* to see the working.")
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Input

    @ViewBuilder private var inputSection: some View {
        if model.isBatch {
            HStack {
                Text("\(model.batch.count) files").font(.headline)
                Spacer()
                Button("Replace…") { isImporting = true }
                Button("Clear") { model.clearInput() }
            }
            .controlSize(.small)
        } else if let input = model.input {
            HStack(alignment: .center, spacing: 16) {
                Thumbnail(image: model.thumbnail)
                    .frame(width: 176, height: 99)
                VStack(alignment: .leading, spacing: 4) {
                    Text(input.url.lastPathComponent).font(.headline).lineLimit(1)
                    Text(describe(input)).foregroundStyle(.secondary).font(.callout)
                    HStack {
                        Button("Replace…") { isImporting = true }
                        Button("Clear") { model.clearInput() }
                    }
                    .disabled(model.isRunning)
                    .controlSize(.small)
                    .padding(.top, 4)
                }
                Spacer()
            }
        } else {
            dropZone
        }
    }

    private var dropZone: some View {
        VStack(spacing: 10) {
            Image(systemName: "film.stack").font(.system(size: 34)).foregroundStyle(.secondary)
            Text("Drop videos or images here").font(.headline)
            Text("Several at once queue one job each; they run one at a time.")
                .font(.caption).foregroundStyle(.secondary)
            Button("Choose…") { isImporting = true }
            if model.isLoadingInput { ProgressView().controlSize(.small) }
        }
        .frame(maxWidth: .infinity, minHeight: 190)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [6, 5]))
                .foregroundStyle(isTargeted ? Color.accentColor : Color.secondary.opacity(0.5))
        )
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(isTargeted ? Color.accentColor.opacity(0.08) : Color.clear)
        )
        .dropDestination(for: URL.self) { urls, _ in
            guard !urls.isEmpty else { return false }
            Task { await model.load(urls) }
            return true
        } isTargeted: { isTargeted = $0 }
    }

    private func describe(_ input: UpscaleModel.Input) -> String {
        switch input {
        case .video(let video):
            var parts = ["\(video.width)×\(video.height)",
                         String(format: "%.1f s", video.durationSeconds),
                         String(format: "%.0f fps", video.fps),
                         "\(video.frameCount) frames"]
            parts.append(video.hasAudio ? "audio" : "no audio")
            if video.hasAlpha { parts.append("alpha") }
            return parts.joined(separator: " · ")
        case .image(_, let width, let height):
            return "\(width)×\(height) · image"
        }
    }

    // MARK: - Options

    private var optionsSection: some View {
        Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 16, verticalSpacing: 14) {
            GridRow {
                Text("Output").gridColumnAlignment(.trailing)
                HStack(spacing: 12) {
                    Picker("Output", selection: $model.outputSize) {
                        ForEach(model.availableOutputSizes) { Text($0.label).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                    if let plan = model.plan {
                        Text("→ \(plan.outputWidth)×\(plan.outputHeight)")
                            .monospacedDigit().foregroundStyle(.secondary)
                    }
                }
            }
            GridRow {
                Text("Quality")
                VStack(alignment: .leading, spacing: 4) {
                    Picker("Quality", selection: $model.quality) {
                        Text("Fast").tag(QualityPreset.fast)
                        Text("Balanced").tag(QualityPreset.balanced)
                        Text("Best").tag(QualityPreset.best)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                    if model.input?.isVideo == true {
                        Text(qualityNote).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .disabled(model.isRunning)
    }

    private var qualityNote: String {
        switch model.quality {
        case .fast: "No overlap between chunks — quickest; seams can show on fast motion."
        case .balanced: "One shared frame between chunks, cross-faded."
        case .best: "Two shared frames between chunks — smoothest seams, a little slower."
        }
    }

    // MARK: - Run

    @ViewBuilder private var runSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            jobStatus
            startButton
        }
    }

    @ViewBuilder private var jobStatus: some View {
        if let job = model.currentJob, job.state == .queued || job.state == .running || job.state == .paused || job.state == .interrupted {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                VStack(alignment: .leading, spacing: 8) {
                    ProgressView(value: job.progress.fraction)
                    HStack {
                        Text(runLabel(job)).font(.callout.weight(.medium))
                        Spacer()
                        Text(timing(job, now: context.date))
                            .font(.callout).monospacedDigit().foregroundStyle(.secondary)
                        switch job.state {
                        case .running:
                            Button("Pause") { model.pause() }
                            Button("Cancel", role: .cancel) { model.cancel() }
                        case .queued:
                            Button("Cancel", role: .cancel) { model.cancel() }
                        default:
                            Button("Resume") { model.resume() }
                            Button("Cancel", role: .cancel) { model.cancel() }
                        }
                    }
                    Text(footnote(job)).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    private var startButton: some View {
            HStack {
                Button {
                    model.start()
                } label: {
                    Label(model.willQueue || model.isRunning ? "Add to queue" : "Upscale",
                          systemImage: "arrow.up.left.and.arrow.down.right")
                        .frame(minWidth: 120)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!model.canStart)
                if let blocked = startBlockedReason {
                    Text(blocked).font(.callout).foregroundStyle(.secondary)
                } else if model.willQueue {
                    Text("Jobs run one at a time — this one waits its turn, and is planned for the memory free when it starts.")
                        .font(.callout).foregroundStyle(.secondary)
                }
            }
    }

    private var startBlockedReason: String? {
        if model.modelState != .installed { return "Install the model first." }
        if model.plan == nil { return "Nothing fits right now — see the plan above." }
        return nil
    }

    private func runLabel(_ job: Job) -> String {
        switch job.state {
        case .queued:
            if let blocker = model.queue.blocker(for: job.id) { return "Waiting for \(blocker.title)" }
            return model.queue.isPaused ? "Queued — the queue is paused" : "Queued — waiting its turn"
        case .paused: return "Paused" + (job.note.map { " — \($0)" } ?? "")
        case .interrupted: return "Interrupted — resumes from its last checkpoint"
        default: break
        }
        let phase = UpscaleModel.phaseLabel(job.progress.phase)
        guard job.progress.unitsTotal > 1 else { return phase }
        let chunk = min(job.progress.unitsDone + 1, job.progress.unitsTotal)
        return "\(phase) · chunk \(chunk) of \(job.progress.unitsTotal)"
    }

    private func timing(_ job: Job, now: Date) -> String {
        let elapsed = model.queue.elapsedSeconds(for: job.id, now: now)
        guard job.state == .running else { return elapsed > 0 ? "\(Format.clock(elapsed)) so far" : "" }
        if let remaining = model.remainingSeconds(now: now) {
            return "\(Format.clock(elapsed)) elapsed · about \(Format.duration(remaining)) left"
        }
        return "\(Format.clock(elapsed)) elapsed"
    }

    private func footnote(_ job: Job) -> String {
        switch job.state {
        case .running:
            return "Runs in the background — closing this window doesn't stop it. The Mac is kept awake while it runs."
        case .paused, .interrupted:
            return "Work done so far is kept; resuming skips it."
        default:
            return "Progress also shows under Jobs."
        }
    }
}

// MARK: - Thumbnail

struct Thumbnail: View {
    let image: CGImage?

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8).fill(.quaternary)
            if let image {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                Image(systemName: "film").foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Plan card

/// The preflight: what will run, what it will cost, and why — before anything loads.
struct PlanCard: View {
    let model: UpscaleModel
    @State private var showReasons = PlanCard.reasonsInitiallyExpanded

    #if DEBUG
    static let reasonsInitiallyExpanded = ProcessInfo.processInfo.environment["HUGMAC_EXPAND_REASONS"] == "1"
    #else
    static let reasonsInitiallyExpanded = false
    #endif

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                modelRow
                Divider()
                if let refusal = model.refusal {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("This won't fit right now").font(.headline)
                            Text(refusal).font(.callout)
                        }
                    } icon: {
                        Image(systemName: "xmark.octagon.fill")
                    }
                    .foregroundStyle(.red)
                } else if let plan = model.plan {
                    if model.planIsProvisional {
                        Label("Another job is running, so this is planned for the memory free when the Mac is idle. The job is planned for real when it starts.",
                              systemImage: "clock")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                    phases(plan)
                    MemoryBar(peak: plan.peakBytes,
                              budget: model.hardware.plannableMemoryBytes(),
                              total: model.hardware.totalMemoryBytes)
                    summary(plan)
                    DisclosureGroup("Why these settings", isExpanded: $showReasons) {
                        ReasonsList(reasons: plan.reasons).padding(.top, 6)
                    }
                }
            }
            .padding(6)
        } label: {
            Label("Plan for this Mac", systemImage: "cpu")
        }
    }

    @ViewBuilder private var modelRow: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text("SeedVR2 3B · int8").font(.headline)
                Text(model.hardware.chipName + (model.hardware.gpuCoreCount.map { ", \($0) GPU cores" } ?? ""))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            switch model.modelState {
            case .checking:
                ProgressView().controlSize(.small)
            case .installed:
                Label("Installed", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.callout)
            case .notInstalled:
                Button("Install · \(Format.bytes(model.variant.downloadBytes))") {
                    Task { await model.installModel() }
                }
            case .installing(let fraction, let detail):
                VStack(alignment: .trailing, spacing: 4) {
                    ProgressView(value: fraction).frame(width: 180)
                    HStack(spacing: 6) {
                        Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                        Button("Pause") { Task { await model.cancelInstall() } }
                            .controlSize(.mini)
                    }
                }
            case .failed(let message):
                VStack(alignment: .trailing, spacing: 4) {
                    Button("Retry install") { Task { await model.installModel() } }
                    Text(message).font(.caption).foregroundStyle(.red)
                        .multilineTextAlignment(.trailing).frame(maxWidth: 320, alignment: .trailing)
                }
            }
        }
    }

    private func phases(_ plan: SeedVR2Plan) -> some View {
        Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 6) {
            GridRow {
                Text("Step").foregroundStyle(.secondary)
                Text("Peak memory").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                Text("Time").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                Text("")
            }
            .font(.caption)
            ForEach(plan.phases, id: \.phase) { phase in
                GridRow {
                    Text(Self.phaseName(phase.phase))
                    Text(Format.bytes(phase.peakBytes)).monospacedDigit()
                    Text(phase.time.seconds.map { "≈ " + Format.duration($0) } ?? "—").monospacedDigit()
                    Text(phase.peakIsMeasured ? "measured here" : "estimated")
                        .font(.caption2)
                        .padding(.horizontal, 6).padding(.vertical, 1)
                        .background(Capsule().fill(phase.peakIsMeasured
                                                   ? Color.green.opacity(0.15) : Color.orange.opacity(0.15)))
                }
            }
        }
    }

    private func summary(_ plan: SeedVR2Plan) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if let total = plan.totalTime.seconds {
                Text("About \(Format.duration(total))").font(.title3.weight(.semibold))
                    + Text("  " + Self.timeSource(plan.totalTime)).font(.caption).foregroundColor(.secondary)
            } else {
                Text("Time unknown until the first run on this Mac").font(.callout.weight(.medium))
            }
            let chunkNote = plan.isSingleImage
                ? "Single image"
                : "\(plan.chunks.count) chunks of up to \(plan.chunks.map(\.length).max() ?? 0) frames"
            Text(chunkNote + " · VAE " + tiling(plan))
                .font(.callout).foregroundStyle(.secondary)
            Text(diskLine).font(.callout).foregroundStyle(.secondary)
        }
    }

    static func timeSource(_ time: TimeEstimate) -> String {
        switch time {
        case .extrapolated(_, let chip, .probes): "estimated from an \(chip), scaled by this Mac's measured speed"
        case .extrapolated(_, let chip, .specs): "estimated from an \(chip), scaled by spec sheet"
        default: "predicted from runs on this Mac"
        }
    }

    private func tiling(_ plan: SeedVR2Plan) -> String {
        func describe(_ tiling: VAETiling?) -> String {
            tiling.map { "\($0.tileSize) px tiles" } ?? "whole frames"
        }
        if plan.encodeTiling == plan.decodeTiling { return describe(plan.decodeTiling) }
        return "encode \(describe(plan.encodeTiling)), decode \(describe(plan.decodeTiling))"
    }

    private var diskLine: String {
        var parts: [String] = []
        if let output = model.estimatedOutputBytes { parts.append("output ≈ \(Format.bytes(output))") }
        if let scratch = model.estimatedScratchBytes { parts.append("working files ≈ \(Format.bytes(scratch))") }
        parts.append("\(Format.bytes(model.freeDiskBytes)) free")
        return "Disk: " + parts.joined(separator: " · ")
    }

    static func phaseName(_ phase: String) -> String {
        switch phase {
        case "vae-encode": "Encode frames"
        case "dit": "Upscale (transformer)"
        case "vae-decode": "Decode frames"
        default: phase
        }
    }
}

/// Peak memory against what the plan was allowed to use.
struct MemoryBar: View {
    let peak: Int64
    let budget: Int64
    let total: Int64

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            GeometryReader { geometry in
                let width = geometry.size.width
                let scale = width / CGFloat(max(total, 1))
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary)
                    Capsule().fill(Color.accentColor.opacity(0.18))
                        .frame(width: CGFloat(budget) * scale)
                    Capsule().fill(Color.accentColor)
                        .frame(width: max(CGFloat(peak) * scale, 4))
                }
            }
            .frame(height: 8)
            Text("Peak \(Format.bytes(peak)) of \(Format.bytes(budget)) available now · \(Format.bytes(total)) in this Mac")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

/// Every resolved setting, with who decided it and why.
struct ReasonsList: View {
    let reasons: [SettingReason]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(reasons.enumerated()), id: \.offset) { _, reason in
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 8) {
                        Text(reason.setting).font(.callout.weight(.medium))
                        Text(reason.value).font(.callout.monospaced())
                        ProvenanceBadge(provenance: reason.provenance)
                    }
                    Text(reason.because).font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

struct ProvenanceBadge: View {
    let provenance: SettingProvenance

    var body: some View {
        Text(label)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 6).padding(.vertical, 1)
            .background(Capsule().fill(color.opacity(0.15)))
            .foregroundStyle(color)
    }

    private var label: String {
        switch provenance {
        case .intent: "your choice"
        case .autoMachine: "auto · this Mac"
        case .autoInput: "auto · from file"
        case .advanced: "default"
        case .manual: "manual"
        case .eliminated: "n/a"
        }
    }

    private var color: Color {
        switch provenance {
        case .intent: .accentColor
        case .autoMachine, .autoInput: .blue
        case .advanced, .eliminated: .secondary
        case .manual: .orange
        }
    }
}

// MARK: - Result

struct ResultPanel: View {
    let outcome: JobOutcome
    let plan: SeedVR2Plan?
    let isVideo: Bool

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                preview
                    .frame(maxWidth: .infinity)
                    .frame(height: 360)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(outcome.outputURL.lastPathComponent).font(.headline)
                        Text(stats).font(.callout).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Show in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([outcome.outputURL])
                    }
                    Button("Open") { NSWorkspace.shared.open(outcome.outputURL) }
                }
            }
            .padding(6)
        } label: {
            Label("Done", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        }
    }

    @ViewBuilder private var preview: some View {
        if isVideo {
            PlayerView(url: outcome.outputURL)
        } else if let image = NSImage(contentsOf: outcome.outputURL) {
            Image(nsImage: image).resizable().aspectRatio(contentMode: .fit)
        }
    }

    private var stats: String {
        var parts = [Format.duration(outcome.seconds), "peak \(Format.bytes(outcome.peakBytes))"]
        if let predicted = plan?.peakBytes {
            parts[1] += " (predicted \(Format.bytes(predicted)))"
        }
        return parts.joined(separator: " · ")
    }
}

/// AppKit's `AVPlayerView`, wrapped for SwiftUI.
///
/// Not SwiftUI's `VideoPlayer`: built with the macOS 27 SDK and run on macOS 26, it aborted
/// the app the first time it was shown (a fatal error setting up its type metadata inside
/// AVKit's SwiftUI overlay). `AVPlayerView` is long-standing AppKit API with no such bridge.
struct PlayerView: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .inline
        view.player = AVPlayer(url: url)
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if (view.player?.currentItem?.asset as? AVURLAsset)?.url != url {
            view.player = AVPlayer(url: url)
        }
    }

    static func dismantleNSView(_ view: AVPlayerView, coordinator: ()) {
        view.player?.pause()
    }
}

// MARK: - Batch

/// Several files sharing one Output and Quality: what each becomes, and which won't be queued.
struct BatchList: View {
    let model: UpscaleModel

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(model.batch) { item in
                    HStack(alignment: .firstTextBaseline) {
                        Image(systemName: item.input.isVideo ? "film" : "photo")
                            .foregroundStyle(.secondary)
                        Text(item.input.url.lastPathComponent).lineLimit(1)
                        Spacer()
                        if let plan = item.plan, item.problem == nil {
                            Text("\(item.input.width)×\(item.input.height) → \(plan.outputWidth)×\(plan.outputHeight)")
                                .monospacedDigit().foregroundStyle(.secondary)
                            Text("peak \(Format.bytes(plan.peakBytes))")
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            Text(item.problem ?? "—").font(.caption).foregroundStyle(.orange)
                                .lineLimit(2).multilineTextAlignment(.trailing)
                                .frame(maxWidth: 320, alignment: .trailing)
                        }
                    }
                    .font(.callout)
                }
                Divider()
                HStack {
                    Button {
                        model.addBatchToQueue()
                    } label: {
                        Label("Add \(model.batchQueueableCount) to queue", systemImage: "tray.and.arrow.down")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.canQueueBatch)
                    if model.modelState != .installed {
                        Text("Install the model first.").font(.callout).foregroundStyle(.secondary)
                    } else {
                        Text("They run one at a time, each planned for the memory free when it starts.")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                }
            }
            .padding(6)
        } label: {
            Label("Batch", systemImage: "square.stack")
        }
    }
}
