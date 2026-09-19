import LocalLabCore
import SwiftUI
import UniformTypeIdentifiers

/// The Settings window. Storage is its first pane.
public struct SettingsView: View {
    let app: AppModel

    public init(app: AppModel) {
        self.app = app
    }

    public var body: some View {
        TabView {
            StorageSettingsView(model: app.storage)
                .tabItem { Label("Storage", systemImage: "externaldrive") }
            ChatSettingsView(model: app.chat)
                .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
        }
        .frame(width: 680, height: 600)
    }
}

/// Settings → Storage (plan §5.4).
public struct StorageSettingsView: View {
    let model: StorageModel
    @State private var choosingFolder = false
    @State private var confirmDelete: StorageModel.ModelRow?

    public init(model: StorageModel) {
        self.model = model
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                if let missing = model.unavailable {
                    Label("The library on “\(LibraryLocation.displayName(missing))” isn't connected. LocalLab is using the library on this Mac until it is — reconnect it and restart LocalLab. Nothing was deleted.",
                          systemImage: "externaldrive.badge.exclamationmark")
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
                location
                moveStatus
                contents
            }
            .padding(20)
        }
        .task { await model.refresh() }
        .fileImporter(isPresented: $choosingFolder, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { model.choose(url) }
        }
        .confirmationDialog(
            "Delete \(confirmDelete?.name ?? "")?",
            isPresented: Binding(get: { confirmDelete != nil }, set: { if !$0 { confirmDelete = nil } }),
            presenting: confirmDelete
        ) { row in
            Button("Delete \(Format.bytes(row.bytes))", role: .destructive) {
                Task { await model.delete(row.repo) }
            }
        } message: { _ in
            Text("Its files are removed from the library. You can install it again later.")
        }
    }

    // MARK: - Location

    private var location: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline) {
                    Image(systemName: "internaldrive").foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(model.displayLocation).font(.callout.monospaced()).textSelection(.enabled)
                        Text("\(model.volumeName) · \(Format.bytes(model.freeBytes)) free · library \(model.isMeasuring ? "measuring…" : Format.bytes(model.totalBytes))")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                }
                HStack {
                    Button("Change…") { choosingFolder = true }
                        .disabled(model.moveBlocker != nil)
                    Button("Show in Finder") { model.reveal() }
                    if model.isCustom {
                        Button("Move back to this Mac") { model.moveToDefaultLocation() }
                            .disabled(model.moveBlocker != nil)
                    }
                    Spacer()
                }
                if let blocker = model.moveBlocker, !isMoving {
                    Text(blocker).font(.caption).foregroundStyle(.secondary)
                }
                Text("Your measurements, speed tests and conversations stay on this Mac wherever the library is — they describe this Mac, not the library.")
                    .font(.caption).foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Library location", systemImage: "folder")
        }
    }

    private var isMoving: Bool {
        if case .moving = model.moveState { return true }
        return false
    }

    @ViewBuilder private var moveStatus: some View {
        if let pending = model.pendingMove, !isMoving {
            GroupBox {
                HStack {
                    Text("A move to “\(LibraryLocation.displayName(pending))” was stopped part way. What's been copied and verified is kept; your library is still here.")
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer()
                    Button("Resume") { model.resumePendingMove() }
                    Button("Discard") { model.discardPendingMove() }
                }
                .padding(6)
            }
        }
        switch model.moveState {
        case .idle:
            EmptyView()
        case .confirmMove(let plan):
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Move the library to “\(LibraryLocation.displayName(plan.destination))”?").font(.headline)
                    Text(plan.destination.path).font(.caption.monospaced()).foregroundStyle(.secondary)
                    Text(moveDescription(plan)).fixedSize(horizontal: false, vertical: true)
                    HStack {
                        Spacer()
                        Button("Cancel") { model.dismiss() }
                        Button("Move \(Format.bytes(plan.totalBytes))") { model.startMove(plan) }
                            .keyboardShortcut(.defaultAction)
                    }
                }
                .padding(6)
            }
        case .confirmUse(let root):
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Text("That folder already holds a LocalLab library.").font(.headline)
                    Text("Use it instead of this one? Nothing is moved or deleted — this library stays where it is. LocalLab restarts to switch.")
                        .fixedSize(horizontal: false, vertical: true)
                    HStack {
                        Spacer()
                        Button("Cancel") { model.dismiss() }
                        Button("Use “\(LibraryLocation.displayName(root))”") { model.useLibrary(at: root) }
                            .keyboardShortcut(.defaultAction)
                    }
                }
                .padding(6)
            }
        case .moving(let progress):
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    ProgressView(value: progress.fraction)
                    HStack {
                        Text(phaseText(progress)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                        Spacer()
                        Text("\(Format.bytes(progress.bytesDone)) of \(Format.bytes(progress.bytesTotal))")
                            .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                        Button("Stop") { model.stopMove() }
                    }
                }
                .padding(6)
            }
        case .failed(let message):
            HStack {
                Label(message, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer()
                Button("OK") { model.dismiss() }
            }
        case .finished(let root):
            Label("The library is now on “\(LibraryLocation.displayName(root))”. LocalLab is restarting…", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        }
    }

    private func moveDescription(_ plan: LibraryMover.Plan) -> String {
        if plan.sameVolume {
            return "It's on the same disk, so the move is instant. LocalLab restarts when it's done."
        }
        return "\(plan.fileCount) files go to another disk. Each is copied, then read back and checked before anything here is deleted. You can stop and resume. LocalLab restarts when it's done."
    }

    private func phaseText(_ progress: LibraryMover.Progress) -> String {
        switch progress.phase {
        case .copying: "Copying \(progress.file)"
        case .verifying: "Checking \(progress.file)"
        case .finishing: "Finishing…"
        }
    }

    // MARK: - Contents

    private var contents: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                if model.models.isEmpty && !model.isMeasuring {
                    Text("No models installed.").foregroundStyle(.secondary)
                }
                ForEach(model.models) { row in
                    HStack {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(row.name)
                            Text("\(row.kind) · \(row.repo)").font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(Format.bytes(row.bytes)).monospacedDigit().foregroundStyle(.secondary)
                        Button {
                            model.reveal(model.url(forRepo: row.repo))
                        } label: {
                            Image(systemName: "magnifyingglass")
                        }
                        .buttonStyle(.borderless)
                        .help("Show in Finder")
                        Button(role: .destructive) {
                            confirmDelete = row
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                        .disabled(model.deleteBlocker(row.repo) != nil || isMoving)
                        .help(model.deleteBlocker(row.repo) ?? "Delete")
                    }
                    Divider()
                }
                if let error = model.deleteError {
                    Text(error).font(.caption).foregroundStyle(.orange)
                }
                HStack {
                    Text("Outputs")
                    Spacer()
                    Text(Format.bytes(model.outputsBytes)).monospacedDigit().foregroundStyle(.secondary)
                    Button("Show") { model.reveal(model.outputsURL) }.controlSize(.small)
                }
                HStack {
                    Text("Partial downloads")
                    Spacer()
                    Text(Format.bytes(model.downloadsBytes)).monospacedDigit().foregroundStyle(.secondary)
                    if model.downloadsBytes > 0 {
                        Button("Discard") { Task { await model.discardDownloads() } }.controlSize(.small)
                    }
                }
                HStack {
                    Text("Job checkpoints")
                    Spacer()
                    Text(Format.bytes(model.jobsBytes)).monospacedDigit().foregroundStyle(.secondary)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("In the library", systemImage: "shippingbox")
        }
    }
}
