import Foundation

/// Which files of a repo an install needs, and what must be inside them. Plan §6.2.
///
/// Most repos are handled by `ModelFileSelector`'s heuristics. A manifest exists for the
/// cases where heuristics are either wasteful or unsafe:
///
///  - **Waste.** `MiniMaxAI/MiniMax-H3` is 498 GB because it ships three duplicated
///    pipelines; one mode needs a fraction of that. An explicit `include` list downloads
///    only the components a mode uses.
///  - **Safety.** A file can be present, the right size, and still be the wrong file. The
///    owner's ComfyUI logs record `preview TAE is missing decoder tensors`, discovered only
///    when a run started. `requiredTensors` is checked at install time, by name, from the
///    safetensors header — before the model is marked installed.
public struct ComponentManifest: Sendable, Equatable {
    /// Glob patterns (repo-relative, `fnmatch` syntax). `nil` means "use the heuristics".
    public var include: [String]?
    /// Patterns removed after `include` or the heuristics have run.
    public var exclude: [String]
    /// Tensor names each safetensors file must contain, keyed by repo-relative path.
    public var requiredTensors: [String: [String]]
    /// Other repos whose files are bundled into this install, under a subdirectory.
    public var companions: [Companion]

    public struct Companion: Sendable, Equatable {
        public let repo: String
        public let subdirectory: String

        public init(repo: String, subdirectory: String) {
            self.repo = repo
            self.subdirectory = subdirectory
        }
    }

    public init(
        include: [String]? = nil,
        exclude: [String] = [],
        requiredTensors: [String: [String]] = [:],
        companions: [Companion] = []
    ) {
        self.include = include
        self.exclude = exclude
        self.requiredTensors = requiredTensors
        self.companions = companions
    }

    /// Heuristic selection, plus any companion `ModelFileSelector` already knows about.
    public static func heuristic(for repo: String) -> ComponentManifest {
        let companions = ModelFileSelector.companionRepo(for: repo).map {
            [Companion(repo: $0, subdirectory: "encodec")]
        } ?? []
        return ComponentManifest(companions: companions)
    }

    /// The subset of `paths` this manifest selects.
    public func select(from paths: [String]) -> [String] {
        let chosen: [String]
        if let include {
            chosen = paths.filter { path in include.contains { Self.matches(path, pattern: $0) } }
        } else {
            chosen = ModelFileSelector.filesToDownload(siblings: paths)
        }
        return chosen.filter { path in !exclude.contains { Self.matches(path, pattern: $0) } }
    }

    static func matches(_ path: String, pattern: String) -> Bool {
        fnmatch(pattern, path, 0) == 0
    }
}
