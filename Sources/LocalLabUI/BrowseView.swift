import LocalLabCore
import SwiftUI

/// Browse: MLX models from Hugging Face, graded by Smart Fit for this Mac.
public struct BrowseView: View {
    @Bindable var model: BrowseModel

    public init(model: BrowseModel) {
        self.model = model
    }

    public var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                controls
                Divider()
                list
            }
            .frame(width: 430)
            Divider()
            Group {
                if let row = model.selectedRow {
                    ModelDetail(model: model, row: row)
                } else {
                    Text("Choose a model to see how it runs on this Mac.")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .navigationTitle("Browse")
        .task { if model.rows.isEmpty { await model.search() } }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Search MLX models", text: $model.query.text)
                .textFieldStyle(.roundedBorder)
            HStack {
                Picker("Task", selection: $model.query.task) {
                    Text("All tasks").tag(CatalogTask?.none)
                    ForEach(CatalogTask.allCases, id: \.self) { Text($0.title).tag(CatalogTask?.some($0)) }
                }
                .labelsHidden()
                .fixedSize()
                Picker("Sort", selection: $model.query.sort) {
                    Text("Best fit").tag(CatalogSort.bestFit)
                    Text("Most downloaded").tag(CatalogSort.downloads)
                    Text("Most liked").tag(CatalogSort.likes)
                    Text("Recently updated").tag(CatalogSort.recent)
                }
                .labelsHidden()
                .fixedSize()
                Spacer()
            }
            HStack(spacing: 8) {
                Text("From").foregroundStyle(.secondary)
                Picker("From", selection: $model.query.publisher) {
                    Text("mlx-community").tag(CatalogPublisher.mlxCommunity)
                    Text("All publishers").tag(CatalogPublisher.everyone)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                .help("mlx-community does most MLX conversions of well-known models. All publishers adds everyone else publishing MLX weights — other converters and individuals, including republished and modified versions.")
                Spacer()
            }
            .font(.callout)
            if let since = model.offlineSince {
                Label("Offline — showing results from \(since.formatted(date: .abbreviated, time: .shortened)).", systemImage: "wifi.slash")
                    .font(.caption).foregroundStyle(.orange)
            }
        }
        .padding(10)
    }

    private var list: some View {
        List(selection: Binding(get: { model.selected }, set: { model.select($0) })) {
            ForEach(model.rows) { row in
                BrowseRow(row: row, installed: model.installed.contains(row.entry.repo),
                          installing: model.installs[row.entry.repo])
                    .tag(row.entry.repo)
            }
            if model.canLoadMore {
                Button("Load more") { Task { await model.loadMore() } }
                    .buttonStyle(.borderless)
                    .frame(maxWidth: .infinity)
            }
        }
        .overlay {
            if model.isLoading && model.rows.isEmpty {
                ProgressView()
            } else if let error = model.errorMessage, model.rows.isEmpty {
                VStack(spacing: 8) {
                    Text(error).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    Button("Try again") { Task { await model.search() } }
                }
                .padding()
            } else if model.rows.isEmpty && !model.isLoading {
                Text("No MLX models match.").foregroundStyle(.secondary)
            }
        }
    }
}

private struct FitDot: View {
    let grade: ModelVerdict.Grade

    var body: some View {
        Group {
            switch grade {
            case .green: Circle().fill(.green)
            case .yellow: Circle().fill(.yellow)
            case .red: Circle().fill(.red)
            case .notRunnable: Circle().strokeBorder(.secondary, lineWidth: 1.5)
            }
        }
        .frame(width: 10, height: 10)
    }
}

private struct BrowseRow: View {
    let row: BrowseModel.Row
    let installed: Bool
    let installing: Double?

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            FitDot(grade: row.verdict.grade).padding(.top, 4)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(row.entry.name).font(.body.weight(.medium)).lineLimit(1)
                    if installed {
                        Text("Installed").font(.caption2)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Capsule().fill(Color.accentColor.opacity(0.15)))
                    }
                }
                Text(subtitle).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                Text(row.verdict.headline).font(.caption)
                    .foregroundStyle(row.verdict.runner.isRunnable ? .primary : .secondary)
                if let installing {
                    ProgressView(value: installing).controlSize(.small)
                }
            }
            Spacer(minLength: 0)
            VStack(alignment: .trailing, spacing: 2) {
                if let bytes = row.verdict.weightBytes {
                    Text((row.verdict.isEstimate ? "~" : "") + Format.bytes(bytes)).font(.caption).monospacedDigit()
                }
                Text(downloads).font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 3)
    }

    private var subtitle: String {
        var parts = [row.entry.publisher, row.verdict.runner.label == "Not yet" ? taskName : row.verdict.runner.label]
        if let bits = row.entry.bits { parts.append("\(bits)-bit") }
        return parts.joined(separator: " · ")
    }

    private var taskName: String {
        CatalogTask.allCases.first { $0.pipelineTagForDisplay == row.entry.task }?.title ?? (row.entry.task ?? "model")
    }

    private var downloads: String {
        let count = row.entry.downloads
        if count >= 1_000_000 { return String(format: "%.1fM ↓", Double(count) / 1_000_000) }
        if count >= 1_000 { return String(format: "%.0fk ↓", Double(count) / 1_000) }
        return "\(count) ↓"
    }
}

private struct ModelDetail: View {
    let model: BrowseModel
    let row: BrowseModel.Row

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(row.entry.name).font(.title2.weight(.semibold)).textSelection(.enabled)
                    HStack(spacing: 8) {
                        Text(row.entry.publisher).foregroundStyle(.secondary)
                        if let url = model.huggingFaceURL(row.entry.repo) {
                            Link("View on Hugging Face", destination: url).font(.callout)
                        }
                    }
                }
                verdict
                install
                facts
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var verdict: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    FitDot(grade: row.verdict.grade)
                    Text(row.verdict.headline).font(.headline)
                }
                if !row.verdict.arithmetic.isEmpty {
                    Text(row.verdict.arithmetic).font(.callout).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let caveat = row.verdict.caveat {
                    Text(caveat).font(.callout)
                        .foregroundStyle(row.verdict.runner.isRunnable ? Color.orange : Color.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if case .notRunnable(let fits?) = row.verdict.grade {
                    Text(fits ? "It would fit in memory here once LocalLab can run it." : "It wouldn't fit in memory here even then.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if row.verdict.isEstimate {
                    Text(model.details[row.entry.repo] == nil ? "Estimated from the search listing — refining…" : "Estimated: this model's figures aren't curated yet.")
                        .font(.caption).foregroundStyle(.tertiary)
                }
                if let error = model.detailsError {
                    Text("Couldn't load its file list: \(error)").font(.caption).foregroundStyle(.orange)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Smart Fit for this Mac", systemImage: "cpu")
        }
    }

    @ViewBuilder private var install: some View {
        let repo = row.entry.repo
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                if model.installed.contains(repo) {
                    HStack {
                        Label("Installed", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                        Spacer()
                        if case .chat = row.verdict.runner {
                            Button("Open in Chat") { model.chat(with: repo) }
                        }
                        Button("Delete", role: .destructive) { Task { await model.uninstall(repo) } }
                    }
                } else if let fraction = model.installs[repo] {
                    HStack {
                        ProgressView(value: fraction)
                        Button("Pause") { Task { await model.cancelInstall(repo) } }
                    }
                } else {
                    if LicenseInfo.needsAcknowledgement(row.entry.license) {
                        Toggle(isOn: Binding(get: { model.isAcknowledged(repo) }, set: { model.acknowledge(repo, $0) })) {
                            Text("I've read the licence (\(LicenseInfo.displayName(row.entry.license))) and will use the model within its terms.")
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .toggleStyle(.checkbox)
                    }
                    HStack {
                        Text(downloadSize).foregroundStyle(.secondary)
                        Spacer()
                        Button("Install") { Task { await model.install(row) } }
                            .disabled(model.installBlocker(row) != nil)
                    }
                    if case .notYet = row.verdict.runner {
                        Text("You can install it now; LocalLab won't be able to run it until its engine arrives.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                if let error = model.installErrors[repo] {
                    Text(error).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Library", systemImage: "square.and.arrow.down")
        }
    }

    private var downloadSize: String {
        if let details = model.details[row.entry.repo] { return Format.bytes(details.downloadBytes) + " to download" }
        if let bytes = row.verdict.weightBytes { return "About " + Format.bytes(bytes) + " to download" }
        return "Size unknown"
    }

    private var facts: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
            fact("Runs in LocalLab", row.verdict.runner.isRunnable ? row.verdict.runner.label : "Not yet")
            fact("Task", row.entry.task ?? "—")
            fact("Architecture", row.entry.modelType ?? "—")
            fact("Quantization", row.entry.bits.map { "\($0)-bit" } ?? "—")
            fact("Licence", LicenseInfo.displayName(row.entry.license))
            fact("Based on", row.entry.baseModel ?? "—")
            fact("Downloads", row.entry.downloads.formatted())
            fact("Likes", row.entry.likes.formatted())
            if let updated = row.entry.lastModified {
                fact("Updated", updated.formatted(date: .abbreviated, time: .omitted))
            }
            if let config = model.details[row.entry.repo]?.config {
                fact("Layers", "\(config.layers)" + (config.cacheLayers != config.layers ? " (\(config.cacheLayers) with a growing cache)" : ""))
                fact("Context", "\(config.maxContext.formatted()) tokens")
                if let experts = config.experts, let active = config.expertsPerToken {
                    fact("Experts", "\(active) of \(experts) per token")
                }
            }
        }
        .font(.callout)
    }

    private func fact(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(value).textSelection(.enabled)
        }
    }
}

extension CatalogTask {
    /// The pipeline tag this task filters on, for labelling rows.
    var pipelineTagForDisplay: String {
        switch self {
        case .chat: "text-generation"
        case .vision: "image-text-to-text"
        case .speechToText: "automatic-speech-recognition"
        case .textToSpeech: "text-to-speech"
        case .image: "text-to-image"
        case .video: "text-to-video"
        case .upscale: "image-to-image"
        case .embeddings: "feature-extraction"
        }
    }
}
