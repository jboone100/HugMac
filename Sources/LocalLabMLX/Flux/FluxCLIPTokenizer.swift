//
// Ported from MLXUI (same author, MIT License); MLXUI's port follows
// ml-explore/mlx-examples/flux/flux/tokenizers.py (MIT), itself after Hugging Face's CLIPTokenizer.
//

import Foundation

/// FLUX's CLIP tokenizer: byte-pair encoding over the repo's `tokenizer/vocab.json` and
/// `tokenizer/merges.txt`, in CLIP's 77-token window.
///
/// The rank table is keyed by an ordered *pair* of subwords, as the reference builds it from
/// `tuple(line.split())` — keying by a joined `"a b"` string would silently never merge.
struct FluxCLIPTokenizer: Sendable {
    static let maxLength = 77

    struct Bigram: Hashable, Sendable {
        let a: String
        let b: String
        init(_ a: String, _ b: String) { self.a = a; self.b = b }
    }

    let ranks: [Bigram: Int]
    let vocabulary: [String: Int]

    private static let bos = "<|startoftext|>"
    private static let eos = "<|endoftext|>"

    init(ranks: [Bigram: Int], vocabulary: [String: Int]) {
        self.ranks = ranks
        self.vocabulary = vocabulary
    }

    /// Reads `vocab.json` and `merges.txt` from `directory`.
    init(directory: URL) throws {
        let vocabData = try Data(contentsOf: directory.appendingPathComponent("vocab.json"))
        let vocabulary = try JSONDecoder().decode([String: Int].self, from: vocabData)
        let merges = try String(contentsOf: directory.appendingPathComponent("merges.txt"), encoding: .utf8)
        var ranks: [Bigram: Int] = [:]
        // The first line is a "#version" header.
        for (rank, line) in merges.split(separator: "\n").dropFirst().enumerated() {
            let parts = line.split(separator: " ")
            guard parts.count == 2 else { continue }
            ranks[Bigram(String(parts[0]), String(parts[1]))] = rank
        }
        self.init(ranks: ranks, vocabulary: vocabulary)
    }

    /// Letters, numbers, punctuation runs and the two markers, case-insensitive — the
    /// reference's split.
    private static let pattern: NSRegularExpression? = try? NSRegularExpression(
        pattern: #"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#,
        options: [.caseInsensitive]
    )

    /// Lowercase → split → BPE → BOS/EOS → truncate to 77, keeping EOS last.
    func tokenize(_ text: String) -> [Int] {
        let clean = text.lowercased()
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        let range = NSRange(clean.startIndex..., in: clean)
        let words = (Self.pattern?.matches(in: clean, range: range) ?? []).compactMap { match in
            Range(match.range, in: clean).map { String(clean[$0]) }
        }
        var ids = words.flatMap(bpe).compactMap { vocabulary[$0] }
        if let bos = vocabulary[Self.bos] { ids.insert(bos, at: 0) }
        if let eos = vocabulary[Self.eos] {
            ids.append(eos)
            if ids.count > Self.maxLength {
                ids = Array(ids.prefix(Self.maxLength))
                ids[ids.count - 1] = eos
            }
        }
        return ids
    }

    /// Repeatedly merge the lowest-ranked adjacent pair.
    func bpe(_ word: String) -> [String] {
        guard let last = word.last else { return [] }
        var parts = word.dropLast().map { String($0) } + ["\(last)</w>"]
        while parts.count > 1 {
            var best: (pair: Bigram, rank: Int)?
            for (a, b) in zip(parts, parts.dropFirst()) {
                if let rank = ranks[Bigram(a, b)], rank < best?.rank ?? .max {
                    best = (Bigram(a, b), rank)
                }
            }
            guard let pair = best?.pair else { break }
            var merged: [String] = []
            var index = 0
            while index < parts.count {
                if index + 1 < parts.count, parts[index] == pair.a, parts[index + 1] == pair.b {
                    merged.append(pair.a + pair.b)
                    index += 2
                } else {
                    merged.append(parts[index])
                    index += 1
                }
            }
            parts = merged
        }
        return parts
    }
}

/// Both of FLUX's tokenizers, read from an installed model's `tokenizer/` and `tokenizer_2/`.
struct FluxTokenizers: Sendable {
    let clip: FluxCLIPTokenizer
    let t5: FluxSentencePiece.Model

    init(directory: URL) throws {
        clip = try FluxCLIPTokenizer(directory: directory.appendingPathComponent("tokenizer"))
        t5 = try FluxSentencePiece.parse(from: directory.appendingPathComponent("tokenizer_2/spiece.model"))
    }

    func t5Tokens(_ prompt: String, maxLength: Int) -> [Int] {
        FluxT5Tokenizer.tokenize(prompt, model: t5, maxLength: maxLength)
    }

    func clipTokens(_ prompt: String) -> [Int] {
        clip.tokenize(prompt)
    }
}
