import Foundation

/// One named input of a stage. Replaces MLXUI's single `accepts: MediaKind`, which forced
/// every second input through `StageConfig` (the VLM question rode in `StageConfig.prompt`)
/// and made "image + prompt → video" or "image + mask → image" awkward to express.
///
/// A Run UI builds its form from these: an image well per image slot, a text box per text
/// slot, a file picker per video slot.
public struct InputSlot: Sendable, Hashable {
    public let name: String
    public let kind: MediaKind
    public let required: Bool

    public init(name: String, kind: MediaKind, required: Bool = true) {
        self.name = name
        self.kind = kind
        self.required = required
    }

    /// The conventional name for a single-input stage, so ported one-slot stages stay trivial.
    public static func input(_ kind: MediaKind) -> InputSlot {
        InputSlot(name: "input", kind: kind)
    }
}

/// Progress from a running stage. Richer than MLXUI's bare `Double` because a video job
/// needs to be resumable: `checkpoint` is where the work so far was persisted, so a crash
/// at chunk 38 of 61 resumes at 38 rather than 0.
public struct StageProgress: Sendable {
    public let fraction: Double
    public let phase: String
    public let unitsDone: Int
    public let unitsTotal: Int
    public let checkpoint: URL?

    public init(
        fraction: Double,
        phase: String = "",
        unitsDone: Int = 0,
        unitsTotal: Int = 0,
        checkpoint: URL? = nil
    ) {
        self.fraction = fraction
        self.phase = phase
        self.unitsDone = unitsDone
        self.unitsTotal = unitsTotal
        self.checkpoint = checkpoint
    }
}

/// One node of work: declares the media it consumes by slot name and what it produces.
public protocol PipelineStage: Sendable {
    var id: String { get }
    var name: String { get }
    var inputs: [InputSlot] { get }
    var produces: MediaKind { get }

    func run(
        _ inputs: [String: Media],
        progress: @Sendable @escaping (StageProgress) -> Void
    ) async throws -> Media
}

public extension PipelineStage {
    /// Shared guard so every `run` starts the same way: every required slot present, and
    /// every supplied value of the declared kind. A mismatch is caught here — before any
    /// model loads — rather than part way into a multi-hour job.
    func validate(_ supplied: [String: Media]) throws {
        for slot in inputs {
            guard let value = supplied[slot.name] else {
                if slot.required {
                    throw StageError.missingInput(slot: slot.name, kind: slot.kind)
                }
                continue
            }
            guard value.kind == slot.kind else {
                throw StageError.kindMismatch(slot: slot.name, expected: slot.kind, got: value.kind)
            }
        }
    }

    func require(_ supplied: [String: Media], _ slot: String, _ kind: MediaKind) throws -> Media {
        guard let value = supplied[slot] else {
            throw StageError.missingInput(slot: slot, kind: kind)
        }
        guard value.kind == kind else {
            throw StageError.kindMismatch(slot: slot, expected: kind, got: value.kind)
        }
        return value
    }
}

/// Errors a stage (or its planner) can throw. `CustomStringConvertible` so
/// `String(describing:)` yields a plain sentence rather than an enum dump.
public enum StageError: Error, CustomStringConvertible, Equatable {
    case missingInput(slot: String, kind: MediaKind)
    case kindMismatch(slot: String, expected: MediaKind, got: MediaKind)
    case modelNotInstalled(id: String)
    case componentIncomplete(model: String, detail: String)
    case insufficientMemory(requiredGB: Double, availableGB: Double)
    case insufficientDisk(requiredGB: Double, availableGB: Double)
    case unsupportedSetting(String)
    case engineFailure(stage: String, detail: String)
    case cancelled

    public var description: String {
        switch self {
        case .missingInput(let slot, let kind):
            return "This step needs a \(kind.rawValue) for '\(slot)' — add one, then run again."
        case .kindMismatch(let slot, let expected, let got):
            return "This step expected \(expected.rawValue) for '\(slot)' but got \(got.rawValue) — check what feeds it."
        case .modelNotInstalled(let id):
            return "The model '\(id)' isn't installed yet — install it, then run again."
        case .componentIncomplete(let model, let detail):
            return "'\(model)' is installed but incomplete — \(detail). Reinstall it."
        case .insufficientMemory(let required, let available):
            return String(format: "This step needs about %.1f GB but only %.1f GB is available — close some apps or pick a lighter setting.", required, available)
        case .insufficientDisk(let required, let available):
            return String(format: "This step needs about %.1f GB of disk but only %.1f GB is free where models are stored.", required, available)
        case .unsupportedSetting(let setting):
            return "This step's \(setting) setting isn't supported by this engine yet."
        case .engineFailure(let stage, let detail):
            return "The \(stage) engine failed — \(detail)."
        case .cancelled:
            return "Cancelled."
        }
    }
}
