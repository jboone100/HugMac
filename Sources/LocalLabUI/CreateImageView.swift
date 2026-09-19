import AppKit
import ImageIO
import LocalLabCore
import SwiftUI

/// Create Image: describe it, pick a size, and see what that costs on this Mac before
/// anything loads.
public struct CreateImageView: View {
    @Bindable var model: CreateImageModel
    @State private var selectedResult: UUID?

    public init(model: CreateImageModel) {
        self.model = model
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                if model.modelState != .installed {
                    ModelCard(model: model)
                }
                promptSection
                optionsSection
                runSection
                if let message = model.errorMessage {
                    Label(message, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red)
                }
                ResultsSection(model: model, selected: $selectedResult)
            }
            .padding(24)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .task {
            await model.refreshModelState()
            model.refit()
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Create Image").font(.largeTitle.weight(.semibold))
            Text("\(model.spec.displayName) — runs entirely on this Mac. Describe what you want to see; the more specific, the better.")
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Prompt

    private var promptSection: some View {
        ZStack(alignment: .topLeading) {
            TextEditor(text: $model.prompt)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(8)
                .frame(minHeight: 90, maxHeight: 160)
            if model.prompt.isEmpty {
                Text("A lighthouse on a rocky cliff at dusk, waves breaking below, oil painting")
                    .foregroundStyle(.tertiary)
                    .padding(.horizontal, 13).padding(.vertical, 8)
                    .allowsHitTesting(false)
            }
        }
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .textBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.quaternary))
    }

    // MARK: - Options

    private var optionsSection: some View {
        Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 16, verticalSpacing: 12) {
            GridRow {
                Text("Size").gridColumnAlignment(.trailing)
                HStack(spacing: 10) {
                    Picker("Size", selection: $model.size) {
                        ForEach(ImageSize.allCases) { size in
                            Text("\(size.label) — \(size.dimensions)").tag(size)
                        }
                    }
                    .labelsHidden()
                    .fixedSize()
                    if let fit = model.fit {
                        FitSummary(fit: fit)
                    }
                }
            }
            GridRow {
                Text("Seed")
                HStack(spacing: 10) {
                    TextField("Random", text: $model.seedText)
                        .frame(width: 130)
                        .monospacedDigit()
                    Text(model.seedIsValid
                         ? "The same seed, prompt and size make the same image."
                         : "Use a whole number, or leave it blank.")
                        .font(.caption)
                        .foregroundStyle(model.seedIsValid ? Color.secondary : Color.red)
                }
            }
        }
    }

    // MARK: - Run

    @ViewBuilder private var runSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let job = model.currentJob, !job.state.isFinished {
                JobProgress(job: job, model: model)
            } else if let job = model.currentJob, job.state == .failed {
                Label(job.failure ?? "The image couldn't be made.", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }
            HStack {
                Button {
                    model.start()
                } label: {
                    Label(model.willQueue ? "Add to queue" : "Create", systemImage: "wand.and.stars")
                        .frame(minWidth: 110)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(model.startBlocker != nil)
                if let blocker = model.startBlocker {
                    Text(blocker).font(.callout).foregroundStyle(.secondary)
                } else if model.willQueue {
                    Text("Jobs run one at a time — this one waits its turn.")
                        .font(.callout).foregroundStyle(.secondary)
                } else {
                    Text("⌘↩").font(.callout).foregroundStyle(.tertiary)
                }
            }
        }
    }
}

// MARK: - Smart Fit

private struct FitSummary: View {
    let fit: ImageFit
    @State private var showWorking = false

    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 9, height: 9)
            Text(summary).font(.callout).foregroundStyle(.secondary)
            Button {
                showWorking.toggle()
            } label: {
                Image(systemName: "info.circle")
            }
            .buttonStyle(.borderless)
            .help("How this was worked out")
            .popover(isPresented: $showWorking, arrowEdge: .bottom) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Smart Fit").font(.headline)
                    Text(fit.arithmetic).font(.callout).fixedSize(horizontal: false, vertical: true)
                    if case .extrapolated = fit.time {
                        Text("The time is scaled from another Mac's runs. After the first image, it's measured on this one.")
                            .font(.caption).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(14)
                .frame(width: 340)
            }
        }
    }

    private var color: Color {
        switch fit.grade {
        case .green: .green
        case .yellow: .yellow
        case .red: .red
        }
    }

    private var summary: String {
        let time = fit.time.seconds.map { "About \(ImageModelFitter.duration($0)) an image" } ?? "Time not known yet"
        switch fit.grade {
        case .green: return time
        // Yellow is either memory (close some apps) or time (slow here).
        case .yellow(let because): return fit.fitsNow ? time + " — slow on this Mac" : because.prefix(1).uppercased() + because.dropFirst()
        case .red(let because): return "Too large — " + because
        }
    }
}

// MARK: - Model card

private struct ModelCard: View {
    let model: CreateImageModel

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(model.spec.displayName).font(.headline)
                        Text("\(Format.bytes(model.spec.downloadBytes)) download · \(LicenseInfo.displayName(model.spec.license)) licence · \(model.spec.repo)")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    action
                }
                if case .installing(let fraction, let detail) = model.modelState {
                    ProgressView(value: fraction)
                    Text(detail).font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
                if case .failed(let message) = model.modelState {
                    Label(message, systemImage: "exclamationmark.triangle.fill")
                        .font(.callout).foregroundStyle(.red)
                }
                Text("Downloaded once from Hugging Face. After that it works offline, and nothing you type leaves this Mac.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(6)
        }
    }

    @ViewBuilder private var action: some View {
        switch model.modelState {
        case .checking:
            ProgressView().controlSize(.small)
        case .notInstalled, .failed:
            Button("Install") { Task { await model.installModel() } }
                .buttonStyle(.borderedProminent)
        case .installing:
            Button("Stop") { Task { await model.cancelInstall() } }
        case .installed:
            EmptyView()
        }
    }
}

// MARK: - Progress

private struct JobProgress: View {
    let job: Job
    let model: CreateImageModel

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            VStack(alignment: .leading, spacing: 6) {
                ProgressView(value: job.progress.fraction)
                HStack {
                    Text(label).font(.callout.weight(.medium))
                    Spacer()
                    let elapsed = model.queue.elapsedSeconds(for: job.id, now: context.date)
                    if job.state == .running, elapsed > 0 {
                        Text("\(Format.clock(elapsed)) elapsed")
                            .font(.callout).monospacedDigit().foregroundStyle(.secondary)
                    }
                    Button("Cancel", role: .cancel) { model.cancel() }
                }
            }
        }
    }

    private var label: String {
        switch job.state {
        case .queued:
            if let blocker = model.queue.blocker(for: job.id) { return "Waiting for \(blocker.title)" }
            return model.queue.isPaused ? "Queued — the queue is paused" : "Queued — waiting its turn"
        case .paused, .interrupted:
            return "Stopped — resume it under Jobs"
        default:
            switch job.progress.phase {
            case "text-encode": return "Reading the prompt"
            case "transformer":
                return job.progress.unitsTotal > 0
                    ? "Drawing · step \(min(job.progress.unitsDone + 1, job.progress.unitsTotal)) of \(job.progress.unitsTotal)"
                    : "Drawing"
            case "vae-decode": return "Finishing"
            default: return "Starting"
            }
        }
    }
}

// MARK: - Results

private struct ResultsSection: View {
    let model: CreateImageModel
    @Binding var selected: UUID?

    var body: some View {
        let results = model.results
        if !results.isEmpty {
            let current = results.first { $0.id == selected } ?? results[0]
            VStack(alignment: .leading, spacing: 12) {
                Divider()
                ResultDetail(job: current, model: model)
                if results.count > 1 {
                    ScrollView(.horizontal) {
                        HStack(spacing: 8) {
                            ForEach(results) { job in
                                Button {
                                    selected = job.id
                                } label: {
                                    StoredImage(url: job.outcome?.outputURL, maxPixels: 160)
                                        .frame(width: 80, height: 80)
                                        .clipShape(RoundedRectangle(cornerRadius: 6))
                                        .overlay(RoundedRectangle(cornerRadius: 6)
                                            .strokeBorder(job.id == current.id ? Color.accentColor : .clear, lineWidth: 2))
                                }
                                .buttonStyle(.plain)
                                .help(prompt(of: job))
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
            }
        }
    }

    private func prompt(of job: Job) -> String {
        if case .createImage(let spec) = job.kind { return spec.prompt }
        return job.title
    }
}

private struct ResultDetail: View {
    let job: Job
    let model: CreateImageModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            let url = job.outcome?.outputURL
            if let url {
                // Drag the picture out to Finder, Mail, Photos…
                StoredImage(url: url, maxPixels: 1600)
                    .frame(maxWidth: .infinity, maxHeight: 520)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .draggable(url) {
                        StoredImage(url: url, maxPixels: 160).frame(width: 80, height: 80)
                    }
            }
            if case .createImage(let spec) = job.kind {
                Text(spec.prompt).font(.callout).textSelection(.enabled)
                Text("\(spec.width) × \(spec.height) · seed \(spec.seed)"
                     + (job.outcome.map { " · made in \(Format.duration($0.seconds))" } ?? ""))
                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            HStack {
                Button {
                    model.upscale(job)
                } label: {
                    Label("Upscale…", systemImage: "arrow.up.left.and.arrow.down.right")
                }
                .help("Open this image in Upscale")
                Button("Use this prompt") { model.reuse(job) }
                    .help("Put the prompt and seed back to make it again or vary it")
                Spacer()
                if let url {
                    Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                    Button("Open") { NSWorkspace.shared.open(url) }
                }
            }
            .controlSize(.small)
        }
    }
}

/// A PNG from disk, decoded off the main thread at no more than `maxPixels` on its long side.
private struct StoredImage: View {
    let url: URL?
    let maxPixels: Int
    @State private var image: CGImage?

    var body: some View {
        ZStack {
            if let image {
                Image(decorative: image, scale: 1).resizable().aspectRatio(contentMode: .fit)
            } else {
                RoundedRectangle(cornerRadius: 6).fill(.quaternary)
            }
        }
        .task(id: url) {
            guard let url else { image = nil; return }
            let limit = maxPixels
            image = await Task.detached { Self.load(url, maxPixels: limit) }.value
        }
    }

    nonisolated static func load(_ url: URL, maxPixels: Int) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixels,
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }
}
