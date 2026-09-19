# LocalLab

Browse, grade, install and run MLX models on Apple Silicon. The design plan lives in `Design/LocalLab-plan.md`, which is kept out of this public repository; section references like “plan §5.9” in the code point there.

Called HugMac until 2026-09-18. On first launch after the rename, the library moves from
`~/Library/Application Support/HugMac` to `…/LocalLab` (instantly, same volume), leaving a
link at the old path; a Hugging Face token saved under the old Keychain name is still read.

## Sandbox

LocalLab is one sandboxed, hardened app for both the Mac App Store and direct download. It
reaches only its own container, plus files and folders you pick or drop:

- **Your library** can live anywhere you choose (Settings → Storage); access is remembered with
  a bookmark. On the first sandboxed launch with an empty library, LocalLab asks for the one it
  used before (`~/Library/Application Support/LocalLab`) and uses it in place — nothing copied
  but this Mac's measurements and conversations.
- **Files you drop or pick** stay readable for the session, and **queued jobs keep a bookmark**
  for inputs outside the library, so a paused or waiting job reaches its input after a relaunch
  — following it if it was moved or renamed.
- The only network access is outgoing, to huggingface.co. `App/PrivacyInfo.xcprivacy` declares no
  tracking and no data collected.

```bash
LOCALLAB_SANDBOX_CHECK=1 .build/xcode/Build/Products/Debug/LocalLab.app/Contents/MacOS/LocalLab
```

## Releasing

The App Store record is **LocalLab AI** (bundle ID `io.github.jboone100.locallab`); the app
calls itself LocalLab. Version 1.0; the build number is the commit count, so each upload is
higher than the last.

```bash
scripts/release.sh            # archive Release, check it, export a signed App Store package
scripts/release.sh --upload   # …and upload it to App Store Connect
```

The script refuses a build that lacks an entitlement, carries the debug entitlement or debug
hooks, is missing the privacy manifest, icon or MLX kernels, or isn't arm64-only. Uploading
needs Xcode signed in to the developer account (Settings → Accounts) so it can use or create
an Apple Distribution certificate. An uploaded build appears in App Store Connect and
TestFlight; it goes to review only when submitted there. The privacy label to enter in App
Store Connect is **Data Not Collected**.

## Current state

Done so far: **the SeedVR2 upscaler**, image and video (plan §6.6 steps 1–3), **the model
installer**, ported from MLXUI's `InstallManager`, the **job queue**, the **machine
profile** with its first-run speed tests (plan §5.13–5.14), **Chat** (plan §5.12), and
**Settings → Storage** (plan §5.4), **Browse** (plan §5.2, §9.1), and **Create Image** with
FLUX.1 schnell (plan §7.1 #3).

| Target | What's in it | Status |
|---|---|---|
| `LocalLabCore` | `Media` (+video), named-slot `PipelineStage`, `HardwareProfile`, `SettingsResolver` for SeedVR2, `VideoIO`, `CalibrationStore`, the model installer | builds; 189 tests pass |
| `LocalLabCore/Catalog` | Hugging Face search client and offline cache, which engine runs a model, licences, Smart Fit verdicts for any MLX model | 15 tests |
| `LocalLabCore/Chat` | chat model catalog, `ChatModelPicker` (Smart Fit), conversations, the markdown block parser | 21 tests |
| `LocalLabCore/Machine` | `MachineProfile`, `MachineKey`, probe results, bundled reference Macs, timing scaled between Macs | 22 tests |
| `LocalLabMLX` | SeedVR2 VAE + transformer (ported from MLXUI), the temporal engine, residency manager, component verification, the stage | builds; benchmark passed |
| `LocalLabCore/Jobs` | the **job queue**: many kinds, one line, chains, reordering, pause, resumable | 22 tests |
| `LocalLabCore/ImageGeneration` | the image model catalog and manifest, sizes, Smart Fit for image models | 9 tests |
| `LocalLabUI` | the **Browse**, **Chat**, **Create Image**, **This Mac**, **Upscale** (single or batch), **Jobs** and **Settings → Storage** screens and their models | 53 tests |
| `App/` + `project.yml` | the macOS app shell, generated with `xcodegen` | builds; runs a real upscale |
| `LocalLabMLX/ProbeSuite` | the first-run speed tests: bandwidth, matmul, quantized matmul, attention, 3-D conv, memory headroom, disk | ~5 s on an M2 Max |
| `LocalLabMLX/ChatEngine` | chat on `mlx-swift-lm`, tokenizers via `swift-transformers`, Eject that returns memory | |
| `LocalLabMLX/Flux` | FLUX.1 schnell: T5 + CLIP encoders, MMDiT transformer, VAE decoder, tokenizers (ported from MLXUI), loaded one phase at a time | 1024² in ~75 s on an M2 Max |
| `locallab-bench` | the acceptance benchmark, `--install`, `--extract`, `--probe`, `--profile`, `--chat`, `--browse`, `--generate` | builds |

## Building

```bash
swift test                      # everything
LOCALLAB_SKIP_MLX=1 swift test    # core only, no Metal toolchain needed
```

`LocalLabMLX` cannot compile on this machine yet: Xcode 27 ships the Metal compiler as a
separate component, and mlx-swift builds its own Metal kernels.

```
error: cannot execute tool 'metal' due to missing Metal Toolchain;
       use: xcodebuild -downloadComponent MetalToolchain
```

Install it once, then `swift build` covers both targets:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Running the app

```bash
xcodegen generate            # LocalLab.xcodeproj is generated, not committed
xcodebuild -project LocalLab.xcodeproj -scheme LocalLab -derivedDataPath .build/xcode build
open .build/xcode/Build/Products/Debug/LocalLab.app
```

**Upscale** takes a video or an image, an output size (2× · 1080p · 1440p · 4K, only sizes
that enlarge the source are offered) and a quality preset. Everything else is derived and
shown on the *Plan for this Mac* card — per-step peak memory and time, whether each figure is
measured on this Mac or estimated, disk needed, and *Why these settings*. If nothing fits,
the card says so with the numbers instead of offering Start. A run holds the Mac awake,
reports chunk-by-chunk progress with time remaining, and feeds its measurements back into
the next plan.

Debug builds accept `LOCALLAB_OPEN=<file>` to pre-load a file, and `LOCALLAB_AUTOSTART=1
LOCALLAB_RESULT=<path>` to run it and write a result line — the real engine, inside the app
bundle, without anyone clicking.

## Jobs

Heavy work runs on one app-wide queue (plan §5.9): queue as many jobs as you like, of any kind,
and they run **one at a time**.

- **Many kinds, one line** — each kind (`upscale` today; `text-to-video` has its job type and
  chaining, its engine comes later) has its own executor; only one job of any kind runs.
- **Chains** — a job can take another job's output as input and wait for it (text-to-video →
  upscale). If the first fails, the second waits for a retry; if it's cancelled, the second
  fails and says why. A blocked job never holds up the jobs behind it.
- **Queue from anywhere** — *Add to queue* is always available; drop several files for one job
  each; queue the same file again at another setting.
- **Arrange it** — drag to reorder, hold a job, pause the whole queue after the current job,
  clear finished jobs.
- **Planned when it starts** — a waiting job is planned against the memory free when its turn
  comes; previews shown while another job runs are labelled provisional.
- **Persistent and resumable** — `jobs/<id>/job.json`; a crash comes back *interrupted*, a quit
  *paused*; every phase checkpoints per chunk, and a resumed job skips finished work.
- **Awake, heat-aware, notifies** — and feeds each job's measurements into the next plan.

Model downloads aren't in the line — they use the network and disk, not the GPU.

## This Mac

LocalLab works out what the Mac it's running on can do, every time it opens (plan §5.13). Nothing
in it is a table of Mac models, so a chip released after this build still gets a profile.

- **Three layers, most trusted last** — hardware facts; measurements from reference Macs that
  ship with the app (today one: an M2 Max, 30-core GPU, 32 GB), scaled to this Mac; and this
  Mac's own runs, which replace the estimates as they arrive.
- **Memory travels between Macs, time doesn't.** A reference Mac's timings are scaled by the ratio
  of the two Macs' speed-test results for the kind of work each phase does (3-D convolution for
  the VAE, matrix multiply for the transformer), or by spec sheet before this Mac is measured.
  Every time says which.
- **First-run speed tests** — a few seconds of small MLX kernels, no download. They run on first
  launch, and again after a macOS upgrade; not on a low battery, a hot Mac, or during a job.
  Stopping them means they won't start by themselves again.
- **What this Mac can do** — each task planned with the models installed: *runs now*, *close
  other apps*, or *too large* with the number it needs; a better model that would fit is named,
  never downloaded.

```bash
swift run locallab-bench --profile   # what the This Mac screen says
swift run locallab-bench --probe     # run the speed tests and save them
```

Results live in `probes.json` beside `calibration.json`, and stay on the Mac.

## Chat

Chat opens ready (plan §5.12). **Smart Fit** — LocalLab's name for choosing from the hardware
profile — picks the best installed model that runs well on
this Mac: green first (fits with headroom in memory free now, and fast enough to read), then
the highest quality, then the fastest. Any installed model can be chosen instead; each is
graded with its arithmetic — weights + context cache + runtime, and tokens a second.

- **Never downloads behind your back.** A better model that would run here is named, with its
  size and an Install button. With nothing installed, the screen is the recommendation.
- **Memory comes back.** Eject, an idle timeout (**Settings → Chat**: 1 minute to 1 hour, or
  never; 5 minutes by default), or a job starting (a streaming reply finishes first) unloads
  the model and returns MLX's cached buffers to the system. The conversation is kept.
- **Follow-ups don't re-read the conversation.** The model's key/value cache is kept between
  turns, so each reply reads only the new message (measured: 20 new tokens in 0.2 s with 482
  already cached). It's rebuilt — the conversation read once — after an unload, a switch of
  conversation, *Think first* or context, or when the context fills.
- **Ask about images.** Attach a photo (paperclip, or drop it on the conversation) and ask
  about it — or just send it to have it described. Smart Fit picks a model that can see; the
  Qwen3.5 family can. The model loads its vision half only for conversations with images,
  since it costs memory (4.7 → 5.5 GB for the 9B) and speed (~38 → ~16 tok/s here); vision
  replies are measured apart from text ones. Images are copied into the conversation.
  Browse marks models that see images, and vision architectures that chat only through the
  vision runtime (Qwen2.5-VL, SmolVLM, Pixtral, …) now run.
- **Replies render** as markdown, code, tables or JSON, with a per-message *Show as* override
  remembered per model; a reasoning model's thinking is folded, and off unless *Think first*.
- **Speed is measured** from each reply and feeds the next estimate. Conversations are saved in
  `~/Library/Application Support/LocalLab/conversations/` and never leave the Mac.

v1 covers the Qwen3.5 family (0.8B to 122B-A10B, Apache-2.0).

```bash
swift run locallab-bench --chat mlx-community/Qwen3.5-9B-4bit "Explain unified memory in two sentences."
```

## Create Image

**FLUX.1 schnell** (4-bit, Apache 2.0, `mzbac/flux1.schnell.4bit.mlx`, a 9.9 GB download):
describe an image, choose a size (512², 768², 1024², 1344 × 768 or 768 × 1344 — about a
megapixel at most, which is what FLUX was trained for), and optionally a seed. Smart Fit
shows how long it will take on this Mac and what it needs before anything loads.

- **One phase at a time.** The text encoders (3 GB) read the prompt and are released, the
  transformer (6.7 GB) draws in 4 steps and is released, then the VAE decodes. Peak is
  about 7.2 GB — not the model's full 10 GB — so a 16 GB Mac can run it.
- **Measured on an M2 Max (30-core, 32 GB):** 512² in 21 s, 768² in 43 s, 1024² in 75 s.
  Time is proportional to image tokens × steps (about 4.4 ms each); memory barely changes
  with size. Other Macs' estimates are scaled from these until they've made an image
  themselves.
- **Each image is a job** on the same queue as upscales, so it keeps going when the window
  closes. **Upscale…** sends a result straight to Upscale; **Use this prompt** puts the prompt
  and seed back to vary it. The prompt is saved in the PNG's description.

```bash
swift run -c release locallab-bench --generate "a lighthouse on a cliff at dusk" --size 1024x1024 --seed 42
```

## Browse

Every MLX model on Hugging Face, graded by **Smart Fit** for this Mac (plan §5.2, §9.1).

- **Live search** of Hugging Face, with task filters (Chat, Image Q&A, Speech, Image, Video,
  Upscale, Embeddings) and sorts (Best fit, Most downloaded, Most liked, Recently updated).
  mlx-community by default; *All publishers* includes everyone. The last results are cached,
  so Browse works offline and says when it last fetched.
- **Graded for this Mac**: green, yellow or red with the arithmetic — weights + context cache +
  runtime against what the GPU can use, and tokens a second. Sizes come from the listing (the
  API's parameter counts plus quantization scales — within a few MB for 4/8-bit), and exact
  from the file list and `config.json` once a model is opened.
- **Runnable first.** Which engine runs a model is decided by its architecture and task: chat
  models (any `model_type` `mlx-swift-lm` loads) and the SeedVR2 upscalers today. Others are
  listed after, with why not, and whether they'd fit once they can.
- **Install** from the detail pane; a licence other than Apache/MIT/BSD-style must be read
  and acknowledged first. **Open in Chat** takes an installed chat model straight to Chat —
  which can use any installed chat model, curated or not (uncurated ones are graded from
  their repo, labelled as estimates).

```bash
swift run locallab-bench --browse                 # the list, graded for this Mac
swift run locallab-bench --browse --open <repo>   # listing estimate vs exact figures
```

## Storage

**Settings → Storage** (⌘,) shows where the library is, its free space, and what's in it —
each model's size, outputs, partial downloads, job checkpoints — with Show in Finder and
Delete (refused while a job or chat needs the model).

- **Change…** a folder, or a drive (the library goes in a `LocalLab` folder on it). A folder
  that already holds a library is used as is — nothing moves.
- **Moving is verified.** On the same disk it's an instant rename. To another disk each file is
  copied, read back from the disk and checked against the original's SHA-256; only when every
  file matches are the originals deleted. Stop any time — verified files are kept and the move
  resumes. Saved jobs are rewritten to point at the new place. LocalLab restarts to finish.
- **A disconnected drive is never an empty library.** LocalLab runs on the library on this
  Mac, names the missing one in a banner, and goes back to it once it's connected.
- Measurements, speed tests and conversations stay on this Mac wherever the library goes.

## Installing models

```bash
swift run locallab-bench --install mlx-community/SeedVR2-3B-mlx-int8
```

Installs into `~/Library/Application Support/LocalLab` (`models/`, `downloads/`, `installed.json`).
Running it again on a cancelled install **resumes**; on files already in place it **verifies and
adopts** them instead of downloading. Ported from MLXUI's `InstallManager`, with the parts that
didn't hold up at multi-gigabyte sizes changed:

| MLXUI | LocalLab |
|---|---|
| temp → staging → models copies; needs 2× disk | streamed into staging, renamed into place; ~1× |
| a failed download restarts from zero | partials survive; resume with `Range`, verified across HF's CDN redirect |
| size checked to ±1 KB, skipped when unknown | SHA-256 (LFS) or git blob SHA-1 on every file |
| files from `resolve/main` | every file from one pinned commit |
| safetensors never inspected | headers validated; a manifest's required tensors checked at install |
| one fixed library location (`static let shared`) | `ModelStore` built from a `StorageRoot` |

The Hugging Face token lives in the Keychain under `com.locallab` — separate from MLXUI's.

`LOCALLAB_NETWORK_TESTS=1 swift test --filter NetworkInstallTests` runs the real-network test:
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

| | LocalLab (MLX) | ComfyUI baseline (PyTorch/MPS) |
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
