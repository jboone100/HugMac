# HugMac

Browse, grade, install and run MLX models on Apple Silicon. See `Design/HugMac-plan.md`.

## Current state

First increment: **the SeedVR2 upscaler**, image and video (plan §6.6 steps 1–2).

| Target | What's in it | Status |
|---|---|---|
| `HugMacCore` | `Media` (+video), named-slot `PipelineStage`, `HardwareProfile`, `SettingsResolver` for SeedVR2, `VideoIO`, `CalibrationStore` | **builds; 22 tests pass** |
| `HugMacMLX` | SeedVR2 VAE + transformer (ported from MLXUI), the temporal engine, residency manager, component verification, the stage | **written, not yet compiled** — see below |

## Building

```bash
swift test                      # everything
HUGMAC_SKIP_MLX=1 swift test    # core only, no Metal toolchain needed
```

`HugMacMLX` cannot compile on this machine yet: Xcode 27 ships the Metal compiler as a
separate component, and mlx-swift builds its own Metal kernels.

```
error: cannot execute tool 'metal' due to missing Metal Toolchain;
       use: xcodebuild -downloadComponent MetalToolchain
```

Install it once, then `swift build` covers both targets:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Design notes that the code depends on

- **Frame counts are 4n+1.** The VAE has two temporal downsample levels (4×, causal), so
  latent frames are `1 + (frames − 1) / 4`. The reference pipeline's logged latent shape for a
  5-frame batch (`[2, 96, 168, 16]` at 1344×768) confirms it.
- **Frame sizes pad to a multiple of 16** — VAE stride 8 × transformer patch 2.
- **Phase-major, not chunk-major.** Encode every chunk, release the VAE, denoise every chunk,
  release the transformer, then decode and write. Latents are ~1 MB per chunk, so holding them
  costs megabytes and saves reloading weights 61 times.
- **Tiling is a last resort.** In the reference run on this Mac the two VAE phases were 97% of
  6 h 33 m, and tiling a 1344×768 frame at 512 px with 128 px overlap doubles their pixel work.
  `SeedVR2Resolver` ranks least-tiled plans first and only shortens chunks after that.
- **Memory is actually returned.** `SeedVR2Residency` lowers `Memory.cacheLimit`, clears the
  cache, then restores it. Dropping the reference alone leaves the buffers resident.

## Acceptance benchmark

`Diner_0.mp4` (243 frames, 672×384) → 1344×768 on an M2 Max / 32 GB, every setting chosen by
the resolver, audio intact. The bar, from the owner's ComfyUI run: **6 h 33 m** total (of which
~2.7 h was steady-state VAE decode) at a **7.57 GB** peak.
