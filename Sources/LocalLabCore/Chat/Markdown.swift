import Foundation

/// A model reply split into blocks, so each renders the way it should — markdown as
/// markdown, code as code — instead of MLXUI's literal asterisks and pound signs (plan §5.5).
///
/// **Streaming-safe by construction.** An unterminated fence is a code block marked open,
/// never a flicker between plain text and code; an unterminated `<think>` is thinking in
/// progress. Parsing is a single pass over lines, cheap enough to rerun on every update of
/// the trailing message.
public enum MarkdownBlock: Sendable, Equatable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case code(language: String?, text: String, closed: Bool)
    case list(ordered: Bool, items: [String])
    case quote(String)
    case table(header: [String], rows: [[String]])
    case rule
    /// A reasoning model's `<think>` section — shown folded.
    case thinking(String, closed: Bool)
}

public enum MarkdownParser {
    public static func parse(_ text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var remaining = Substring(text)

        // A leading <think> … </think> section, as Qwen3.5 emits it.
        let trimmedStart = remaining.drop { $0.isWhitespace }
        if trimmedStart.hasPrefix("<think>") {
            let body = trimmedStart.dropFirst("<think>".count)
            if let end = body.range(of: "</think>") {
                blocks.append(.thinking(body[..<end.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines), closed: true))
                remaining = body[end.upperBound...]
            } else {
                return [.thinking(body.trimmingCharacters(in: .whitespacesAndNewlines), closed: false)]
            }
        } else if let end = remaining.range(of: "</think>") {
            // The opening tag was in the prompt, not the reply: everything before the close
            // was thinking.
            blocks.append(.thinking(remaining[..<end.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines), closed: true))
            remaining = remaining[end.upperBound...]
        }

        let lines = remaining.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var index = 0
        var paragraph: [String] = []

        func flushParagraph() {
            let text = paragraph.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { blocks.append(.paragraph(text)) }
            paragraph = []
        }

        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Fenced code.
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                flushParagraph()
                let fence = String(trimmed.prefix(3))
                let language = trimmed.dropFirst(3).trimmingCharacters(in: .whitespaces)
                var body: [String] = []
                index += 1
                var closed = false
                while index < lines.count {
                    if lines[index].trimmingCharacters(in: .whitespaces).hasPrefix(fence) {
                        closed = true
                        index += 1
                        break
                    }
                    body.append(lines[index])
                    index += 1
                }
                blocks.append(.code(language: language.isEmpty ? nil : language,
                                    text: body.joined(separator: "\n"), closed: closed))
                continue
            }

            if trimmed.isEmpty {
                flushParagraph()
                index += 1
                continue
            }

            // ATX heading.
            if let level = headingLevel(trimmed) {
                flushParagraph()
                blocks.append(.heading(level: level,
                                       text: trimmed.drop { $0 == "#" }.trimmingCharacters(in: .whitespaces)))
                index += 1
                continue
            }

            // Horizontal rule.
            if isRule(trimmed) {
                flushParagraph()
                blocks.append(.rule)
                index += 1
                continue
            }

            // Table: a header row followed by a |---| separator.
            if trimmed.hasPrefix("|"), index + 1 < lines.count, isTableSeparator(lines[index + 1]) {
                flushParagraph()
                let header = cells(trimmed)
                var rows: [[String]] = []
                index += 2
                while index < lines.count, lines[index].trimmingCharacters(in: .whitespaces).hasPrefix("|") {
                    rows.append(cells(lines[index].trimmingCharacters(in: .whitespaces)))
                    index += 1
                }
                blocks.append(.table(header: header, rows: rows))
                continue
            }

            // Block quote.
            if trimmed.hasPrefix(">") {
                flushParagraph()
                var quoted: [String] = []
                while index < lines.count {
                    let current = lines[index].trimmingCharacters(in: .whitespaces)
                    guard current.hasPrefix(">") else { break }
                    quoted.append(current.dropFirst().trimmingCharacters(in: .whitespaces))
                    index += 1
                }
                blocks.append(.quote(quoted.joined(separator: "\n")))
                continue
            }

            // Lists: consecutive items of one kind; indented lines continue the item.
            if let ordered = listMarker(trimmed) {
                flushParagraph()
                var items: [String] = []
                while index < lines.count {
                    let current = lines[index]
                    let currentTrimmed = current.trimmingCharacters(in: .whitespaces)
                    if listMarker(currentTrimmed) == ordered, !current.hasPrefix("    ") {
                        items.append(stripMarker(currentTrimmed))
                    } else if !currentTrimmed.isEmpty, current.hasPrefix(" "), !items.isEmpty {
                        // An indented bullet is a sub-point, not text starting with "*".
                        let line = listMarker(currentTrimmed) == false
                            ? "◦ " + stripMarker(currentTrimmed)
                            : currentTrimmed
                        items[items.count - 1] += "\n" + line
                    } else {
                        break
                    }
                    index += 1
                }
                blocks.append(.list(ordered: ordered, items: items))
                continue
            }

            paragraph.append(line)
            index += 1
        }
        flushParagraph()
        return blocks
    }

    static func headingLevel(_ line: String) -> Int? {
        let hashes = line.prefix { $0 == "#" }.count
        guard (1 ... 6).contains(hashes) else { return nil }
        let rest = line.dropFirst(hashes)
        return rest.isEmpty || rest.first == " " ? hashes : nil
    }

    static func isRule(_ line: String) -> Bool {
        let compact = line.replacingOccurrences(of: " ", with: "")
        guard compact.count >= 3, let first = compact.first, "-*_".contains(first) else { return false }
        return compact.allSatisfy { $0 == first }
    }

    static func isTableSeparator(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("|"), trimmed.contains("-") else { return false }
        return trimmed.allSatisfy { "|-: ".contains($0) }
    }

    static func cells(_ row: String) -> [String] {
        var parts = row.split(separator: "|", omittingEmptySubsequences: false).map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        if parts.first == "" { parts.removeFirst() }
        if parts.last == "" { parts.removeLast() }
        return parts
    }

    /// true for "1." / "1)", false for "-" / "*" / "+", nil for not a list item.
    static func listMarker(_ line: String) -> Bool? {
        if line.hasPrefix("- ") || line.hasPrefix("* ") || line.hasPrefix("+ ") { return false }
        let digits = line.prefix { $0.isNumber }
        guard !digits.isEmpty, digits.count <= 3 else { return nil }
        let rest = line.dropFirst(digits.count)
        return rest.hasPrefix(". ") || rest.hasPrefix(") ") ? true : nil
    }

    static func stripMarker(_ line: String) -> String {
        if listMarker(line) == false { return String(line.dropFirst(2)) }
        let digits = line.prefix { $0.isNumber }
        return String(line.dropFirst(digits.count + 2))
    }
}

/// What `auto` rendering decides a reply is.
public enum OutputKind: Sendable, Equatable {
    case markdown
    /// The whole reply is JSON — shown pretty-printed.
    case json(pretty: String)
}

public enum OutputClassifier {
    public static func classify(_ text: String) -> OutputKind {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let first = trimmed.first, first == "{" || first == "[",
           let data = trimmed.data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]),
           let pretty = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]),
           let string = String(data: pretty, encoding: .utf8) {
            return .json(pretty: string)
        }
        // Everything else renders as markdown: plain prose parses to paragraphs, so it is
        // shown exactly as written.
        return .markdown
    }
}
