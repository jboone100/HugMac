import Foundation

/// Who decides a setting. The point of the app: a user sees `intent` settings, and nothing
/// else unless they open Advanced.
public enum SettingProvenance: String, Sendable, Codable {
    /// The architecture makes it meaningless — e.g. `device`, or the offload settings, which
    /// exist for discrete-GPU machines. On Apple Silicon the CPU and GPU share one memory
    /// pool, so "offload to CPU" frees nothing.
    case eliminated
    /// Derived from the machine: available memory, GPU cores, calibration history.
    case autoMachine
    /// Read from the input media.
    case autoInput
    /// The user's choice, on the task screen.
    case intent
    /// A fixed, sensible default. Visible in Advanced, rarely touched.
    case advanced
    /// The user overrode an auto value in Advanced.
    case manual
}

/// A resolved setting with the reason it holds that value, so Advanced can explain itself
/// instead of presenting bare numbers.
public struct SettingReason: Sendable, Equatable, Codable {
    public let setting: String
    public let value: String
    public let provenance: SettingProvenance
    public let because: String

    public init(_ setting: String, _ value: String, _ provenance: SettingProvenance, because: String) {
        self.setting = setting
        self.value = value
        self.provenance = provenance
        self.because = because
    }
}

/// How much the user cares about quality versus time. The only knob besides output size.
public enum QualityPreset: String, Sendable, Codable, CaseIterable {
    case fast, balanced, best

    /// Frames shared between neighbouring chunks. More overlap costs time and buys
    /// smoother seams.
    public var temporalOverlap: Int {
        switch self {
        case .fast: 0
        case .balanced: 1
        case .best: 2
        }
    }
}

/// What the user asked for, in their terms.
public enum UpscaleTarget: Sendable, Equatable, Codable {
    /// Multiply both dimensions.
    case scale(Int)
    /// Set the short side, preserving aspect ratio — what the ComfyUI node's `resolution`
    /// meant (the owner's run used 768).
    case shortSide(Int)
    /// An exact frame size.
    case exact(width: Int, height: Int)
}
