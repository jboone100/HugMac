import AppKit
import HugMacCore
import SwiftUI

/// Renders a reply the way it reads best (plan §5.5): markdown as markdown, code as code with
/// a Copy button, JSON pretty-printed, and a reasoning model's thinking folded away.
struct MessageText: View {
    let text: String
    let mode: RenderMode
    let streaming: Bool

    var body: some View {
        switch mode {
        case .raw:
            Text(text).textSelection(.enabled).font(.body.monospaced())
        case .markdown:
            BlocksView(blocks: MarkdownParser.parse(text), streaming: streaming)
        case .auto:
            switch OutputClassifier.classify(text) {
            case .json(let pretty):
                CodeBlockView(language: "json", text: pretty, closed: true)
            case .markdown:
                BlocksView(blocks: MarkdownParser.parse(text), streaming: streaming)
            }
        }
    }
}

private struct BlocksView: View {
    let blocks: [MarkdownBlock]
    let streaming: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                view(for: block)
            }
        }
        .textSelection(.enabled)
    }

    @ViewBuilder
    private func view(for block: MarkdownBlock) -> some View {
        switch block {
        case .heading(let level, let text):
            Inline(text).font(level == 1 ? .title2.bold() : level == 2 ? .title3.bold() : .headline)
        case .paragraph(let text):
            Inline(text)
        case .code(let language, let text, let closed):
            CodeBlockView(language: language, text: text, closed: closed)
        case .list(let ordered, let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(ordered ? "\(index + 1)." : "•").foregroundStyle(.secondary).monospacedDigit()
                        Inline(item)
                    }
                }
            }
        case .quote(let text):
            HStack(spacing: 8) {
                RoundedRectangle(cornerRadius: 1).fill(.tertiary).frame(width: 3)
                Inline(text).foregroundStyle(.secondary)
            }
        case .table(let header, let rows):
            TableBlock(header: header, rows: rows)
        case .rule:
            Divider()
        case .thinking(let text, let closed):
            ThinkingBlock(text: text, closed: closed)
        }
    }
}

/// Inline markdown — emphasis, code spans, links — via Foundation's parser. Falls back to the
/// literal text if it doesn't parse, which half-streamed markdown sometimes won't.
private struct Inline: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        if let attributed = try? AttributedString(
            markdown: text,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace, failurePolicy: .returnPartiallyParsedIfPossible)
        ) {
            Text(attributed).fixedSize(horizontal: false, vertical: true)
        } else {
            Text(text).fixedSize(horizontal: false, vertical: true)
        }
    }
}

struct CodeBlockView: View {
    let language: String?
    let text: String
    let closed: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language ?? "code").font(.caption).foregroundStyle(.secondary)
                if !closed {
                    ProgressView().controlSize(.mini)
                }
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                } label: {
                    Label("Copy", systemImage: "doc.on.doc").font(.caption)
                }
                .buttonStyle(.borderless)
            }
            .padding(.horizontal, 10).padding(.vertical, 5)
            Divider()
            ScrollView(.horizontal) {
                Text(text).font(.callout.monospaced()).textSelection(.enabled)
                    .padding(10)
                    .fixedSize(horizontal: true, vertical: true)
            }
        }
        .background(RoundedRectangle(cornerRadius: 6).fill(Color(nsColor: .textBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
    }
}

private struct TableBlock: View {
    let header: [String]
    let rows: [[String]]

    var body: some View {
        ScrollView(.horizontal) {
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                GridRow {
                    ForEach(Array(header.enumerated()), id: \.offset) { _, cell in
                        Inline(cell).bold()
                    }
                }
                Divider()
                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    GridRow {
                        ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                            Inline(cell)
                        }
                    }
                }
            }
            .padding(8)
        }
        .background(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
    }
}

private struct ThinkingBlock: View {
    let text: String
    let closed: Bool
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(text).font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                if !closed { ProgressView().controlSize(.mini) }
                Text(closed ? "Thought for \(wordCount) words" : "Thinking…")
                    .font(.callout).foregroundStyle(.secondary)
            }
        }
    }

    private var wordCount: Int {
        text.split(whereSeparator: \.isWhitespace).count
    }
}
