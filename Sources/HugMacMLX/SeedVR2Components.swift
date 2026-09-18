import Foundation
import HugMacCore

/// An installed SeedVR2 checkpoint on disk, verified before anything loads.
///
/// Node-graph tools show what the alternative costs: runs that fail on a model folder
/// that isn't where the loader looks, on a missing `processor` directory,
/// and — worst — on a *present but incomplete* file (`preview TAE is missing decoder
/// tensors`). So completeness is checked **by tensor name**, not by file existence, and it is
/// checked before a multi-hour job starts rather than partway in.
public struct SeedVR2Components: Sendable {
    public let directory: URL
    public let transformerURL: URL
    public let vaeURL: URL
    public let positionEmbeddingURL: URL
    public let configURL: URL?

    public init(directory: URL) {
        self.directory = directory
        transformerURL = directory.appendingPathComponent("transformer.safetensors")
        vaeURL = directory.appendingPathComponent("vae.safetensors")
        positionEmbeddingURL = directory.appendingPathComponent("pos_emb.safetensors")
        let config = directory.appendingPathComponent("config.json")
        configURL = FileManager.default.fileExists(atPath: config.path) ? config : nil
    }

    /// Throws `StageError.componentIncomplete` naming what is missing or damaged.
    ///
    /// Reads safetensors *headers* only — structure and tensor names — using the same check
    /// the installer runs. The earlier version called `MLX.loadArrays` on each file, which for
    /// the transformer meant mapping 4.2 GB just to read a list of names.
    public func verify() throws {
        let fileManager = FileManager.default
        for url in [transformerURL, vaeURL, positionEmbeddingURL] {
            guard fileManager.fileExists(atPath: url.path) else {
                throw StageError.componentIncomplete(
                    model: directory.lastPathComponent,
                    detail: "\(url.lastPathComponent) is missing"
                )
            }
        }
        for url in [transformerURL, vaeURL, positionEmbeddingURL] {
            do {
                let header = try SafetensorsHeader.read(url)
                if let required = SeedVR2Variant.manifest.requiredTensors[url.lastPathComponent] {
                    try header.require(required)
                }
            } catch {
                throw StageError.componentIncomplete(
                    model: directory.lastPathComponent,
                    detail: "\(url.lastPathComponent): \(error)"
                )
            }
        }
    }
}
