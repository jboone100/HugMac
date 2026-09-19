# App Store listing — LocalLab AI 1.0

Drafts for App Store Connect. Every claim matches what version 1.0 does; nothing here promises
video generation, speech or image generation, which aren't built yet. Character counts are
Apple's limits, checked by `scripts/check-listing.py`.

## Name (30)

LocalLab AI

## Subtitle (30)

AI that never leaves your Mac

## Promotional text (170)

Chat with AI models, ask about your photos, and upscale images and video — all on your Mac.
Smart Fit picks the best models for your hardware. Nothing leaves your Mac.

## Description (4000)

LocalLab runs AI models on your Mac — not in the cloud. Your conversations, photos and videos stay on your computer, and once a model is downloaded, everything works offline.

The hard part of running AI locally is knowing what your Mac can handle. LocalLab measures your Mac and does that for you.

SMART FIT: TAILORED TO YOUR MAC
• On first launch, LocalLab runs a few seconds of speed tests on your Mac's GPU and memory — no download needed.
• Every model is rated green, yellow or red for your Mac, with the arithmetic shown: memory needed against memory available, and expected speed.
• Chat picks the best model your Mac runs well. Choose any other model yourself at any time.
• Every setting is chosen for your machine, and "Why these settings" explains each one.

CHAT
• Streaming replies with formatted text, code blocks with copy, tables and more.
• Ask about images: attach a photo, or drop it into the conversation.
• Follow-ups are fast: the model remembers the conversation instead of re-reading it every time.
• Optional "Think first" mode for models that can reason before they answer.
• Eject the model with one click, or let it unload itself when idle, to give the memory back to your other apps.

UPSCALE IMAGES AND VIDEO
• Enlarge photos and videos with the SeedVR2 upscaler, with sound kept on video.
• Queue as many jobs as you like. They run one at a time, so each gets your whole Mac.
• Long video jobs resume where they stopped if you quit or your Mac restarts.
• A clear plan before you start: how much memory each step needs, and how long it should take.

BROWSE THOUSANDS OF MODELS
• Search the MLX models on Hugging Face, filtered by task.
• Each model is rated for your Mac before you download anything, with its licence shown.
• Install with one click; remove models to free space.

YOUR LIBRARY, YOUR WAY
• Keep models wherever you like, including an external drive. Moves are verified before anything is deleted.
• See every model's size, and your outputs, in one place.

PRIVATE BY DESIGN
• No accounts, no analytics, no tracking.
• The only connection LocalLab makes is to Hugging Face, to search for and download models.
• Your conversations, images and results never leave your Mac.

REQUIREMENTS
• A Mac with Apple silicon (M1 or later) and macOS 14 or later.
• Models are downloaded separately and range from under 1 GB to over 60 GB. LocalLab tells you what fits your Mac.
• Each model has its own licence, shown before you install it.

## Keywords (100, comma-separated, no spaces after commas)

offline,private,on-device,llm,chatbot,assistant,upscaler,enhance,photo,video,local model,gpu,image

## What's New (4000)

First release.

## Categories

- Primary: **Productivity**
- Secondary: **Photography** (the upscaler)

## URLs

- Support URL: https://github.com/jboone100/LocalLab
- Privacy Policy URL: https://github.com/jboone100/LocalLab/blob/main/PRIVACY.md
- Contact email (App Review Information and support): locallabai@gmail.com

## App Privacy (the privacy "nutrition label")

**Data Not Collected.** In App Store Connect → App Privacy → Get Started, answer "No, we do not
collect data from this app." LocalLab has no server, no analytics and no accounts; the requests
it sends to Hugging Face to search for and download models go directly from the user's Mac to
Hugging Face and aren't collected by the developer.

## App Review notes (4000)

Paste into App Store Connect → the version page → App Review Information → Notes. Sign-in
required: **off** (no accounts).

```
LocalLab runs AI models entirely on this Mac. No account or sign-in is needed, and the app has no server of its own.

Models are downloaded from Hugging Face the first time they're used. They can be large (from about 0.6 GB to over 20 GB) and may take a while to download. For a quick test, choose Qwen3.5 0.8B (about 0.6 GB) in Chat. It downloads in a minute or two and runs on any Apple Silicon Mac.

Smart Fit grades each model against this Mac's memory and speed. Models that won't run well are marked, and memory warnings explain why.

The download location can be changed in Settings… (⌘,) → Storage, including to an external drive.

Apart from model downloads from Hugging Face, nothing leaves the Mac. Prompts, images and videos are processed on-device.
```
