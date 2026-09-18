# HugMac

Browse, grade, install and run MLX models on Apple Silicon. See `Design/HugMac-plan.md`.

## Current state

Done so far: **the SeedVR2 upscaler**, image and video (plan §6.6 steps 1–3), and **the model
installer**, ported from MLXUI's `InstallManager`.

| Target | What's in it | Status |
|---|---|---|
| `HugMacCore` | `Media` (+video), named-slot `PipelineStage`, `HardwareProfile`, `SettingsResolver` for SeedVR2, `VideoIO`, `CalibrationStore`, **the model installer** | builds; 68 tests pass |
| `HugMacMLX` | SeedVR2 VAE + transformer (ported from MLXUI), the temporal engine, residency manager, component verification, the stage | builds; benchmark passed |
| `hugmac-bench` | the acceptance benchmark, `--install`, `--extract` | builds |

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

## Installing models

```bash
swift run hugmac-bench --install mlx-community/SeedVR2-3B-mlx-int8
```

Installs into `~/Library/Application Support/HugMac` (`models/`, `downloads/`, `installed.json`).
Running it again on a cancelled install **resumes**; on files already in place it **verifies and
adopts** them instead of downloading. Ported from MLXUI's `InstallManager`, with the parts that
didn't hold up at multi-gigabyte sizes changed:

| MLXUI | HugMac |
|---|---|
| temp → staging → models copies; needs 2× disk | streamed into staging, renamed into place; ~1× |
| a failed download restarts from zero | partials survive; resume with `Range`, verified across HF's CDN redirect |
| size checked to ±1 KB, skipped when unknown | SHA-256 (LFS) or git blob SHA-1 on every file |
| files from `resolve/main` | every file from one pinned commit |
| safetensors never inspected | headers validated; a manifest's required tensors checked at install |
| one fixed library location (`static let shared`) | `ModelStore` built from a `StorageRoot` |

The Hugging Face token lives in the Keychain under `com.hugmac` — separate from MLXUI's.

`HUGMAC_NETWORK_TESTS=1 swift test --filter NetworkInstallTests` runs the real-network test:
it cancels a 335 MB download partway, resumes it, and checks it resumed rather than restarted
(~350 MB, cleaned up afterwards).

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

## Acceptance benchmark — passed

`Diner_0.mp4` (243 frames, 672×384) → 1344×768 on an M2 Max / 32 GB, int8 3B, every setting
chosen by the resolver, audio intact.

| | HugMac (MLX) | ComfyUI baseline (PyTorch/MPS) |
|---|---|---|
| Total | **55 m 14 s** | 6 h 33 m wall · 3 h 59 m steady-state |
| VAE encode | 11 m 37 s · 5.09 GB | 1 h 03 m · 1.23 GB |
| Transformer | 7 m 43 s · 13.65 GB | 13 m · 7.57 GB |
| VAE decode | 35 m 54 s · 11.11 GB | 5 h 16 m · 2.65 GB |
| Peak | 13.65 GB | 7.57 GB |

**4.33× faster** than the baseline's steady-state rate (7.1× against its wall clock), with the
predicted peak within 1% of measured (13.52 GB predicted, 13.65 GB actual). Output verified
frame-for-frame against the source at frames 8 and 200 — no drift across 31 chunks, no visible
tile seams, audio remuxed untouched.

The one criterion missed: peak memory. The plan asked for **≤ ~8 GB** and this run took
**13.65 GB**, because the resolver plans to the budget it is given and had 15.3 GB. It is a
deliberate trade, not a leak — the transformer's activations at 9-frame chunks are what cost
the memory, and asking for a smaller peak (a tighter budget, or fewer frames per chunk) gets
one at some cost in time. Worth replacing that criterion with "fits the budget without
swapping", which is the property that actually matters.
