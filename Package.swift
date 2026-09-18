// swift-tools-version: 6.0
import PackageDescription
import Foundation

// The MLX target needs Xcode's Metal Toolchain component to compile mlx-swift's shaders
// (`xcodebuild -downloadComponent MetalToolchain`). Set HUGMAC_SKIP_MLX=1 to build and test
// the pure-Swift core on a machine that doesn't have it yet.
let skipMLX = ProcessInfo.processInfo.environment["HUGMAC_SKIP_MLX"] == "1"

let mlxProducts: [Product] = skipMLX ? [] : [.library(name: "HugMacMLX", targets: ["HugMacMLX"])]
let mlxDependencies: [Package.Dependency] = skipMLX ? [] : [
    // Pinned exactly: mlx-swift has shipped breaking API changes in patch releases.
    .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.4"),
]
let mlxTargets: [Target] = skipMLX ? [] : [
    // The benchmark runner: resolves a plan, runs the engine, reports per-phase peak and
    // time against the ComfyUI baseline.
    .executableTarget(name: "hugmac-bench", dependencies: ["HugMacCore", "HugMacMLX"]),
    // Everything that touches MLX weights and the GPU.
    .target(
        name: "HugMacMLX",
        dependencies: [
            "HugMacCore",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXRandom", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"),
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
