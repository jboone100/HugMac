// swift-tools-version: 6.0
import PackageDescription
import Foundation

// The MLX target needs Xcode's Metal Toolchain component to compile mlx-swift's shaders
// (`xcodebuild -downloadComponent MetalToolchain`). Set HUGMAC_SKIP_MLX=1 to build and test
// the pure-Swift core on a machine that doesn't have it yet.
let skipMLX = ProcessInfo.processInfo.environment["HUGMAC_SKIP_MLX"] == "1"

let mlxProducts: [Product] = skipMLX ? [] : [
    .library(name: "HugMacMLX", targets: ["HugMacMLX"]),
    .library(name: "HugMacUI", targets: ["HugMacUI"]),
]
let mlxDependencies: [Package.Dependency] = skipMLX ? [] : [
    // Pinned exactly: mlx-swift has shipped breaking API changes in patch releases.
    .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.4"),
    // Chat (plan §5.12). Owner-approved 2026-09-18, pinned exactly to MLXUI's versions:
    // the LLM runtime, and the real tokenizers + Jinja chat templates. A hand-rolled
    // tokenizer is not an option — MLXUI's produced garbage tokens (its journal 2026-37).
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "3.31.3"),
    .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
]
let mlxTargets: [Target] = skipMLX ? [] : [
    // The benchmark runner: resolves a plan, runs the engine, reports per-phase peak and
    // time against the ComfyUI baseline.
    .executableTarget(name: "hugmac-bench", dependencies: ["HugMacCore", "HugMacMLX"]),
    // The SwiftUI screens and their models. The app target (App/, via project.yml) is a thin
    // shell around this, so the same code runs in the Xcode app and under `swift test`.
    .target(name: "HugMacUI", dependencies: ["HugMacCore", "HugMacMLX"]),
    .testTarget(name: "HugMacUITests", dependencies: ["HugMacUI", "HugMacCore"]),
    // Everything that touches MLX weights and the GPU.
    .target(
        name: "HugMacMLX",
        dependencies: [
            "HugMacCore",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXRandom", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"),
            .product(name: "MLXLLM", package: "mlx-swift-lm"),
            .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            .product(name: "Tokenizers", package: "swift-transformers"),
        ]
    ),
]

let package = Package(
    name: "HugMac",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "HugMacCore", targets: ["HugMacCore"]),
    ] + mlxProducts,
    dependencies: mlxDependencies,
    targets: [
        // Pure Swift + Apple frameworks. No MLX, so it builds and tests in seconds.
        .target(name: "HugMacCore"),
        .testTarget(name: "HugMacCoreTests", dependencies: ["HugMacCore"]),
    ] + mlxTargets
)
