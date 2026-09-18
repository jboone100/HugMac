import Foundation
import MLX

/// Loads model components and — the part MLXUI never did — actually gives the memory back.
///
/// MLXUI released a stage's Swift objects but left `Memory.cacheLimit` at 60% of RAM, so
/// freed MLX buffers stayed resident and Activity Monitor never dropped. Dropping the
/// reference is not enough: the cache has to be *lowered* before it is cleared, then
/// restored, or the allocator simply holds the buffers.
public actor SeedVR2Residency {
    /// A phase's worth of weights, held only while that phase runs.
    public enum Component: String, Sendable {
        case vae, transformer
    }

    private var loaded: Set<Component> = []
    private let normalCacheLimitBytes: Int

    public init(cacheLimitBytes: Int? = nil) {
        // A modest cache: enough to avoid thrashing inside a phase, not so much that a
        // finished phase keeps gigabytes parked.
        self.normalCacheLimitBytes = cacheLimitBytes ?? 1 << 30
        Memory.cacheLimit = self.normalCacheLimitBytes
    }

    public var resident: Set<Component> { loaded }

    public func markLoaded(_ component: Component) {
        loaded.insert(component)
    }

    /// Release everything and return the memory to the system.
    public func evictAll() {
        loaded.removeAll()
        Self.releaseBuffers(restoringLimitTo: normalCacheLimitBytes)
    }

    public func evict(_ component: Component) {
        loaded.remove(component)
        Self.releaseBuffers(restoringLimitTo: normalCacheLimitBytes)
    }

    /// The sequence that actually frees memory: clamp the cache to nothing, clear it, then
    /// restore the working limit.
    nonisolated static func releaseBuffers(restoringLimitTo limit: Int) {
        Memory.cacheLimit = 0
        Memory.clearCache()
        Memory.cacheLimit = limit
    }

    /// Peak resident bytes MLX has seen, for `CalibrationSample`.
    public nonisolated static func peakMemoryBytes() -> Int64 {
        Int64(Memory.peakMemory)
    }

    public nonisolated static func resetPeak() {
        // mlx-swift exposes the reset through the setter, which ignores the value it is
        // given and calls `mlx_reset_peak_memory()`.
        Memory.peakMemory = 0
    }
}
