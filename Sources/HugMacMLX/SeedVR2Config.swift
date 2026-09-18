//
// SeedVR2Config.swift — ported from MLXUI (same author, MIT License) with no architectural change.
//
// The VAE is a 3D causal autoencoder and the transformer does 3-D windowed attention over
// (t, h, w), so both are already generic over the temporal dimension. MLXUI only ever fed
// them T=1; the video work is chunking and frame IO around them, not new layers.
//

import Foundation

/// SeedVR2 3B transformer configuration.
/// Decoded from `config.json` in the installed model directory.
struct SeedVR2Config: Codable, Sendable {
    var vidInChannels:  Int   = 33
    var vidOutChannels: Int   = 16
    var vidDim:         Int   = 2560
    var txtInDim:       Int   = 5120
    var heads:          Int   = 20
    var headDim:        Int   = 128
    var expandRatio:    Int   = 4
    var ropeOnText:     Bool  = true
    var normEps:        Float = 1e-5
    var patchSize:      [Int] = [1, 2, 2]
    var numLayers:      Int   = 32
    var mmLayers:       Int   = 10
    var ropeDim:        Int   = 128
    var window:         [Int] = [4, 3, 3]

    enum CodingKeys: String, CodingKey {
        case vidInChannels  = "vid_in_channels"
        case vidOutChannels = "vid_out_channels"
        case vidDim         = "vid_dim"
        case txtInDim       = "txt_in_dim"
        case heads
        case headDim        = "head_dim"
        case expandRatio    = "expand_ratio"
        case ropeOnText     = "rope_on_text"
        case normEps        = "norm_eps"
        case patchSize      = "patch_size"
        case numLayers      = "num_layers"
        case mmLayers       = "mm_layers"
        case ropeDim        = "rope_dim"
        case window
    }
}

/// VAE architecture parameters (fixed — same for all SeedVR2 checkpoints).
struct SeedVR2VAEArchConfig: Sendable {
    var baseCh:        Int   = 128
    var chMult:        [Int] = [1, 2, 4, 4]
    var numResBlocks:  Int   = 2
    var zChannels:     Int   = 16
    var inChannels:    Int   = 3
    var outChannels:   Int   = 3
    var scalingFactor: Float = 0.9152
    var numGroups:     Int   = 32
}
