import AppKit
import LocalLabCore
import SwiftUI
import UniformTypeIdentifiers

/// Chat (plan §5.12): opens on the best model this Mac runs well, says why, and names a
/// better one when there is one — without downloading it.
public struct ChatView: View {
    @Bindable var model: ChatModel

    public init(model: ChatModel) {
        self.model = model
    }

    public var body: some View {
        HStack(spacing: 0) {
            ConversationList(model: model)
                .frame(width: 220)
            Divider()
            VStack(spacing: 0) {
                ModelBar(model: model)
                Divider()
                if model.installed.isEmpty {
                    Recommendations(model: model)
                } else {
                    Transcript(model: model)
                        // Drop images onto the conversation to ask about them.
                        .dropDestination(for: URL.self) { urls, _ in
                            guard model.canAttachImages, !model.isGenerating else { return false }
                            model.attach(urls)
                            return true
                        }
                    Divider()
                    Composer(model: model)
                }
            }
        }
        .navigationTitle(model.active?.title ?? "Chat")
        .onAppear { model.refresh() }
    }
}

// MARK: - Conversations

private struct ConversationList: View {
    let model: ChatModel

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Conversations").font(.headline)
                Spacer()
                Button {
                    model.newConversation()
                } label: {
                    Image(systemName: "square.and.pencil")
                }
                .buttonStyle(.borderless)
                .help("New chat")
                .disabled(model.isGenerating)
            }
            .padding(10)
            List(selection: Binding(get: { model.activeID }, set: { if let id = $0 { model.open(id) } })) {
                ForEach(model.conversations) { conversation in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(conversation.title).lineLimit(1)
                        Text(conversation.updatedAt.formatted(date: .abbreviated, time: .shortened))
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    .tag(conversation.id)
                    .contextMenu {
                        Button("Delete", role: .destructive) { model.delete(conversation.id) }
                    }
                }
            }
            .listStyle(.sidebar)
        }
    }
}

// MARK: - Model bar

private struct ModelBar: View {
    let model: ChatModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // One row when there's room; otherwise the options drop to a second row rather
            // than being squeezed until their labels wrap a letter per line.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) {
                    modelControls
                    options
                    Spacer(minLength: 12)
                    engineStatus
                }
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 10) {
                        modelControls
                        Spacer(minLength: 12)
                        engineStatus
                    }
                    HStack(spacing: 10) { options }
                }
            }
            if let fit = model.current {
                Text(fit.arithmetic + " · " + speedSource(fit.speed) + (model.needsVision ? " · with images" : ""))
                    .font(.caption).foregroundStyle(.secondary)
                if case .yellow(let because) = fit.grade {
                    Label(because.prefix(1).uppercased() + because.dropFirst(), systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
            if let reason = model.unavailableReason, !model.installed.isEmpty {
                Label(reason, systemImage: "xmark.octagon").font(.caption).foregroundStyle(.red)
            }
            if let better = model.suggestion {
                HStack(spacing: 6) {
                    Image(systemName: "sparkles").foregroundStyle(.secondary)
                    Text("\(better.spec.displayName) would run well here too, and is better — \(Format.bytes(better.spec.weightBytes)) to download.")
                        .font(.caption).foregroundStyle(.secondary)
                    InstallButton(model: model, spec: better.spec, compact: true)
                }
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    @ViewBuilder private var modelControls: some View {
        Menu {
            Button {
                model.select(.smartFit)
            } label: {
                Text((model.selection == .smartFit ? "✓ " : "") + "Smart Fit — best for this Mac")
            }
            Divider()
            ForEach(model.choices, id: \.spec.repo) { fit in
                Button {
                    model.select(.manual(repo: fit.spec.repo))
                } label: {
                    Text(menuLabel(fit))
                }
                .disabled(!model.installed.contains(fit.spec.repo))
            }
        } label: {
            Text(title).lineLimit(1)
        }
        .fixedSize()
        if let fit = model.current {
            GradeDot(grade: fit.grade)
        }
    }

    @ViewBuilder private var options: some View {
        if let fit = model.current {
            Menu("Context \(ChatModelPicker.contextLabel(fit.context))") {
                Button("Smart Fit") { model.setContext(nil) }
                ForEach(ChatModelPicker.contextLadder.reversed(), id: \.self) { tokens in
                    Button(ChatModelPicker.contextLabel(tokens)) { model.setContext(tokens) }
                }
            }
            .fixedSize()
            if fit.spec.thinks {
                Toggle("Think first", isOn: Bindable(model).thinking)
                    .toggleStyle(.checkbox)
                    .fixedSize()
                    .help("Let the model reason before it answers — slower, sometimes better. The reasoning is shown folded.")
            }
        }
    }

    private var title: String {
        let name = model.current?.spec.displayName ?? "No model"
        return model.selection == .smartFit ? "Smart Fit — \(name)" : name
    }

    private func menuLabel(_ fit: ChatFit) -> String {
        let installed = model.installed.contains(fit.spec.repo)
        let mark: String = switch fit.grade {
        case .green: "🟢"
        case .yellow: "🟡"
        case .red: "🔴"
        }
        return "\(mark) \(fit.spec.displayName) — \(String(format: "%.1f GB · ~%.0f tok/s", fit.peakGB, fit.tokensPerSecond))\(installed ? "" : " · not installed")"
    }

    private func speedSource(_ estimate: TimeEstimate) -> String {
        switch estimate {
        case .measured: "speed measured on this Mac"
        case .extrapolated(_, _, .probes): "speed estimated from this Mac's measured bandwidth"
        case .extrapolated(_, _, .specs):
            model.needsVision ? "vision speed estimated at under half the text speed" : "speed estimated from the spec sheet"
        case .unknown: "speed unknown"
        }
    }

    @ViewBuilder private var engineStatus: some View {
        switch model.engineState {
        case .unloaded:
            Text("Not loaded").font(.caption).foregroundStyle(.secondary).fixedSize()
        case .loading:
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Loading…").font(.caption).foregroundStyle(.secondary).fixedSize()
            }
        case .loaded:
            Button {
                Task { await model.eject() }
            } label: {
                Label("Eject", systemImage: "eject")
            }
            .disabled(model.isGenerating)
            .help("Unload the model and give its memory back to the system.")
        case .failed(let message):
            Label("Couldn't load", systemImage: "exclamationmark.triangle")
                .font(.caption).foregroundStyle(.red).fixedSize().help(message)
        }
    }
}

private struct GradeDot: View {
    let grade: ChatFit.Grade

    var body: some View {
        Circle().fill(color).frame(width: 10, height: 10).help(help)
    }

    private var color: Color {
        switch grade {
        case .green: .green
        case .yellow: .yellow
        case .red: .red
        }
    }

    private var help: String {
        switch grade {
        case .green: "Runs well on this Mac"
        case .yellow(let because), .red(let because): because
        }
    }
}

private struct InstallButton: View {
    let model: ChatModel
    let spec: ChatModelSpec
    var compact = false

    var body: some View {
        if let fraction = model.installs[spec.repo] {
            HStack(spacing: 6) {
                ProgressView(value: fraction).frame(width: compact ? 90 : 160)
                Button("Pause") { Task { await model.cancelInstall(spec) } }.controlSize(.mini)
            }
        } else {
            Button(compact ? "Install" : "Install · \(Format.bytes(spec.weightBytes))") {
                Task { await model.install(spec) }
            }
            .controlSize(compact ? .small : .regular)
        }
    }
}

// MARK: - No model yet

private struct Recommendations: View {
    let model: ChatModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Choose a chat model for this Mac").font(.title3.weight(.semibold))
                Text("These run well here, best first. Nothing downloads until you choose.")
                    .foregroundStyle(.secondary)
                ForEach(model.recommendations, id: \.spec.repo) { fit in
                    GroupBox {
                        HStack(alignment: .top) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(fit.spec.displayName).font(.headline)
                                Text(fit.arithmetic).font(.caption).foregroundStyle(.secondary)
                                Text("License: \(fit.spec.license)").font(.caption).foregroundStyle(.tertiary)
                                if let error = model.installErrors[fit.spec.repo] {
                                    Text(error).font(.caption).foregroundStyle(.red)
                                }
                            }
                            Spacer()
                            InstallButton(model: model, spec: fit.spec)
                        }
                        .padding(4)
                    }
                }
                if model.recommendations.isEmpty {
                    Text("No chat model in LocalLab's list runs comfortably on this Mac. The smallest ones are in the model menu above, graded.")
                        .foregroundStyle(.secondary)
                }
            }
            .padding(20)
            .frame(maxWidth: 640, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
    }
}

// MARK: - Transcript

private struct Transcript: View {
    let model: ChatModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    ForEach(model.messages) { message in
                        MessageRow(model: model, message: message,
                                   streaming: model.isGenerating && message.id == model.messages.last?.id)
                            .id(message.id)
                    }
                    if let error = model.errorMessage {
                        Label(error, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(16)
                .frame(maxWidth: 820, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
            .onChange(of: model.messages.last?.text) { proxy.scrollTo("bottom", anchor: .bottom) }
            .onChange(of: model.activeID) { proxy.scrollTo("bottom", anchor: .bottom) }
        }
        .overlay {
            if model.messages.isEmpty {
                Text(model.current.map { "Ask \($0.spec.displayName) anything." } ?? "")
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct MessageRow: View {
    let model: ChatModel
    let message: ChatMessage
    let streaming: Bool

    var body: some View {
        if message.role == .user {
            HStack {
                Spacer(minLength: 80)
                VStack(alignment: .trailing, spacing: 6) {
                    if let attachments = message.attachments, !attachments.isEmpty {
                        HStack(spacing: 6) {
                            ForEach(attachments, id: \.self) { name in
                                AttachmentThumbnail(url: model.attachmentURL(name), size: 140)
                            }
                        }
                    }
                    Text(message.text).textSelection(.enabled)
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(RoundedRectangle(cornerRadius: 10).fill(Color.accentColor.opacity(0.15)))
                }
            }
        } else {
            VStack(alignment: .leading, spacing: 6) {
                if message.text.isEmpty && streaming {
                    ProgressView().controlSize(.small)
                } else {
                    MessageText(text: message.text, mode: model.renderMode(for: message), streaming: streaming)
                }
                HStack(spacing: 10) {
                    if let stats = message.stats {
                        Text("\(stats.generatedTokens) tokens · \(String(format: "%.0f", stats.tokensPerSecond)) tok/s")
                        if let cached = stats.cachedTokens, cached > 0 {
                            Text("continued — \(cached.formatted()) tokens already read")
                                .help("The model kept its reading of this conversation, so only the new message was read.")
                        }
                    }
                    if message.stopped { Text("stopped") }
                    Spacer()
                    Menu("Show as \(model.renderMode(for: message).rawValue.capitalized)") {
                        ForEach(RenderMode.allCases, id: \.self) { mode in
                            Button(mode.rawValue.capitalized) { model.setRenderMode(mode, for: message.id) }
                        }
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(message.text, forType: .string)
                    } label: {
                        Image(systemName: "doc.on.doc")
                    }
                    .buttonStyle(.borderless)
                    .help("Copy the reply as written")
                }
                .font(.caption).foregroundStyle(.secondary)
                .opacity(streaming ? 0 : 1)
            }
        }
    }
}

// MARK: - Composer

private struct Composer: View {
    @Bindable var model: ChatModel
    @State private var choosing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.pendingAttachments.isEmpty {
                HStack(spacing: 8) {
                    ForEach(model.pendingAttachments, id: \.self) { name in
                        ZStack(alignment: .topTrailing) {
                            AttachmentThumbnail(url: model.attachmentURL(name), size: 64)
                            Button {
                                model.removeAttachment(name)
                            } label: {
                                Image(systemName: "xmark.circle.fill").symbolRenderingMode(.hierarchical)
                            }
                            .buttonStyle(.borderless)
                            .offset(x: 6, y: -6)
                            .help("Remove")
                        }
                    }
                }
            }
            if let error = model.attachError {
                Text(error).font(.caption).foregroundStyle(.orange)
            }
            HStack(alignment: .bottom, spacing: 10) {
                Button {
                    choosing = true
                } label: {
                    Image(systemName: "paperclip")
                }
                .buttonStyle(.borderless)
                .disabled(!model.canAttachImages || model.isGenerating)
                .help(model.canAttachImages
                      ? "Attach an image to ask about — or drop one onto the conversation."
                      : "None of your installed chat models can see images. Browse has ones that can.")
                TextField(model.pendingAttachments.isEmpty ? "Message" : "Ask about the image — or just send to have it described",
                          text: $model.draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1 ... 8)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                    .onSubmit { model.send() }
                    .disabled(model.current == nil && model.pendingAttachments.isEmpty)
                if model.isGenerating {
                    Button("Stop") { model.stop() }
                        .keyboardShortcut(".", modifiers: .command)
                } else {
                    Button("Send") { model.send() }
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(!model.canSend)
                }
            }
        }
        .padding(12)
        .fileImporter(isPresented: $choosing, allowedContentTypes: [.image], allowsMultipleSelection: true) { result in
            if case .success(let urls) = result { model.attach(urls) }
        }
    }
}

/// An attached image, scaled to fit a square.
private struct AttachmentThumbnail: View {
    let url: URL
    let size: CGFloat

    var body: some View {
        Group {
            if let image = NSImage(contentsOf: url) {
                Image(nsImage: image).resizable().scaledToFit()
            } else {
                Image(systemName: "photo").foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: size, maxHeight: size)
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}
