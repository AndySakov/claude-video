# Changelog

All notable changes to `/watch` are documented here.

## [Unreleased]

### Added
- **Frame descriptions.** Each selected frame can get a one-sentence description from a vision model behind any OpenAI-compatible chat API: local Ollama by default (`qwen2.5vl:3b` at `http://localhost:11434/v1`), or llama-server, LM Studio, Groq, or OpenRouter via `WATCH_DESCRIBE_URL`, `WATCH_DESCRIBE_MODEL`, and `WATCH_DESCRIBE_KEY`. Descriptions ride on `FRAME` lines, and the agent reads the timeline first and opens only the frames whose detail matters. `--describe auto|on|off` (`auto` runs only when the endpoint lists the model). At most `WATCH_DESCRIBE_MAX` (default 40) evenly spaced frames are described. Failures are partial evidence and are never cached. Tested with Qwen2.5-VL-3B through llama-server (accurate; ~26 s per frame on 4 CPU cores). SmolVLM-500M hallucinated, so ≥3B models are recommended.
- **Gemini timeline mode.** `--gemini-mode timeline` / `WATCH_GEMINI_MODE=timeline` asks Gemini for timestamped `SCENE:` descriptions, verbatim speech with speaker labels, and on-screen `TEXT:`, rendered into the same timeline format and cached, so follow-up questions need no new upload or API call. An unreadable response falls back to Gemini's raw answer, uncached.
- `setup.py --json` reports `describe`, `describe_url`, `describe_model`, and `describe_ready`.

## [0.4.0] — 2026-10-24

### Added
- **One interleaved timeline.** The local report's separate Frames and Transcript sections are replaced by a single `## Timeline`: `[MM:SS] FRAME <file> (<reason>)` markers interleaved with speech, so what was said sits under what was on screen. Each run also writes `timeline.md` and a machine-readable `timeline.json`.
- Speech fragments are merged into sentence-sized lines (split at punctuation, pauses, size bounds, `>>` speaker changes, and frame boundaries).
- `## Chapters` from yt-dlp metadata or container chapters, and a truncated `## Description` for URLs.
- **Timeline cache** in `~/.cache/watch/timelines` (50 most recent): a rerun with the same source and evidence options reuses the saved report and frames without network, download, or transcription. `--refresh`, `--no-cache`, and `WATCH_CACHE_DIR` (`off` disables). Partial or failed runs are never cached.
- **On-screen text.** Burned-in captions, slides, UI, and code are read by local OCR from a separate adaptive-rate sample (3 fps for short clips, where captions change every ~0.5 s) and interleaved as `[MM:SS] TEXT: …` lines, each reported once when it appears. Burned-in word-pop captions are merged into sentence-sized `CAPTION:` lines (233 rows → 35 on a real 90-second Reel); captions that repeat the speech transcript and recurring background misreads are dropped. Tesseract reads both the plain frame and a bright-text mask (white/yellow captions with dark outlines, which raw Tesseract cannot read over video); Apple Vision is used on macOS when Tesseract is absent. `--ocr`, `WATCH_OCR`, `WATCH_OCR_LANG`, `WATCH_OCR_FPS`; `setup.py --json` reports `ocr_backend` and an install hint. Without platform captions, `transcript` detail now downloads video instead of audio-only so burned-in captions can be read.
- **whisper.cpp speech backend** (`whispercpp`): local transcription from one native binary and a ggml model, with no Python dependencies. `setup.py --backend whispercpp` installs the binary with Homebrew on macOS (or prints the platform's install step), downloads the model into `~/.cache/watch/whisper.cpp`, and warms it up. The detected language is reported. `WATCH_WHISPERCPP_MODEL` (default `small`), `_LANGUAGE`, `_THREADS`, `_TIMEOUT`, `_BIN`. It's now the wizard's recommended transcription choice; WhisperX remains available.
- `WATCH_GROQ_MODEL`.

### Changed
- Groq transcription defaults to `whisper-large-v3-turbo`: about a third of the price, still with segment timestamps. Set `WATCH_GROQ_MODEL=whisper-large-v3` to keep the previous model.

### Fixed
- YouTube rolling auto-captions no longer repeat every line two or three times: a cue's first line that only re-displays the previous cue's last line, and the 10 ms re-display cues, are dropped. Independent repeated speech is kept.
- OCR on older Tesseract (4.1, e.g. Ubuntu 22.04): heavily misread word-pop reveals ("THES 1S") now fold into the caption they grow into, so the caption keeps its real start time.
- Short on-screen text that stays up longer than 2.5 s (slide lines, labels, code) is reported as `TEXT:`, not merged into a `CAPTION:`.

### CI
- The Tests workflow can be run manually (`workflow_dispatch`) and reports why each test was skipped (`-rs`).
- The OCR test video is built from committed frames, so the Apple Vision OCR test runs on macOS (Homebrew's FFmpeg has no `drawtext`).

## [0.3.2] — 2026-09-25

### Changed
- Setup lets the user choose how to add an API key: paste it in chat, or have the agent open `~/.config/watch/.env` in their text editor so they can paste it themselves.
- README: Watch is supported in Claude Code (Claude Desktop's Code tab, VS Code, or a terminal). Claude Chat and Cowork are not supported: in Cowork's cloud environment the Gemini key does not persist between tasks and most sites block yt-dlp downloads. The standalone Chat upload instructions are removed.

## [0.3.1] — 2026-09-25

### Fixed
- `SKILL.md` frontmatter uses only Agent Skills keys, so `watch.skill` uploads to Claude (the upload rejected `version`, `argument-hint`, `user-invocable`, and other extra keys). The version moves to `metadata.version`.

## [0.3.0] — 2026-09-25

### Added
- **Gemini engine.** With a `GEMINI_API_KEY`, watch sends the video to Google's agentic video-understanding model (`gemini-3.7-flash`) and relays its timestamped answer: YouTube URLs directly, everything else via a streamed Files API upload that is deleted afterwards. New `--engine auto|gemini|local`, `--question`, `WATCH_ENGINE`, `WATCH_GEMINI_MODEL`, `WATCH_GEMINI_TIMEOUT`. `--start/--end` use static clipping. Standard library only; no silent fallback between engines.
- Setup wizard asks for the engine first; a Gemini-only setup no longer requires `ffmpeg`/`yt-dlp` for YouTube URLs.
- Managed, optional WhisperX 3.8.6 setup with uv, a separate Python 3.12 environment, cache warm-up, recoverable installation state, and local CPU transcription. Default model: small, int8, batch 8; language and timeout are configurable. Local failures never select a cloud provider.
- One-time fallback selection in the skill wizard: WhisperX, Groq, OpenAI, or captions only. Existing installations retain the `auto` cloud-key preference.
- Original-language caption selection and provenance (English preferred when the source language is unknown), `--sub-lang`, and explicit cookie-file/browser options shared across downloader stages.
- Reports for unavailable modalities, no speech, and missing cloud-transcription intervals. Offline dependency diagnostics and a cross-platform CI gate for release publication.

### Fixed
- SessionStart hook command now quotes `${CLAUDE_PLUGIN_ROOT}`, so plugin paths containing spaces work.
- Uniform sampling now spans the actual capped range and labels selected source frames with their real timestamps, including fractional seeks and cue frames.
- FFmpeg sync-option capability detection (including FFmpeg 9's `-fps_mode[:<stream_spec>]` help format; tested on FFmpeg 4.4 through 9.0), empty keyframe-window fallback, RGB thumbnail deduplication, and actionable probe/launch errors.
- UTF-8 output and subprocess decoding, BOM-aware shared config parsing, provider-key pairing, atomic config updates, keyless readiness, and advisory setup hooks.
- Bounded caption downloads respect watch-owned output templates while retaining intentional yt-dlp configuration. Fresh run directories and completed final-path validation prevent stale/partial media reuse.
- WebVTT optional-hour timestamps, entities, cue/block parsing, and overlap-scoped deduplication preserve independent repeated speech.
- Full-track caption availability is checked before focus filtering; successful captions survive media/probe failures.
- Conservative 24,000,000-byte cloud upload budgets, actual chunk/multipart size checks, and explicit partial/no-speech metadata.

### Changed
- Only the latest yt-dlp release is supported. An HTTP 403 now tells the agent to update yt-dlp with its owning package manager and retry once; CI tests against the latest release instead of a pinned one.
- First-success installation documentation separates media tools from optional speech backends and describes host/network limitations.
- User-provided output directories now contain a disposable run child; cleanup preserves the parent and source files.
- The base runtime remains standard-library-only; local model environments and caches live outside the self-contained skill folder.

## [0.2.0] — 2026-06-29

### Added
- **`--detail` dial** with four modes — `transcript` (captions only, no frames), `efficient` (fast keyframe pass, cap 50), `balanced` (scene-aware, cap 100, default), and `token-burner` (scene-aware, uncapped). Set the default with `WATCH_DETAIL` in `~/.config/watch/.env`.
- **Frame deduplication** (default on; `--no-dedup` to disable). Before the budget cap, a pass downscales each frame to a 16×16 grayscale thumbnail and drops frames whose mean per-pixel difference from the last *kept* frame is within threshold — so the budget goes to distinct content instead of held slides and static recordings. The **Frames** report line shows how many near-duplicates were dropped.
- **Whisper auto-chunking.** Audio over the 25 MB upload cap is split into evenly sized chunks, transcribed per chunk, with segment timestamps shifted back into source time. Partial failures are tolerated — transcription only fails if *every* chunk fails, so length alone no longer breaks it.
- **`--timestamps T1,T2,…`** — grab a frame at each absolute timestamp; reserved against the cap, and the only frames produced under `--detail transcript`.
- **`--no-whisper`** — disable speech fallback while retaining native captions.
- pytest suite covering config, dedup, download, fixtures, frames, setup, timestamps, watch, and whisper (no network; ffmpeg-synthesized clips).

### Changed
- **Restructured into a self-contained `skills/watch/` package** so `SKILL.md` and its `scripts/` runtime are siblings in one folder. This fixes installs on Codex, Cursor, Copilot, and other Agent Skills hosts: `npx skills add` now copies the skill as a working unit instead of grabbing the root `SKILL.md` without its scripts.
- **Harness-agnostic path resolution** — `SKILL.md` resolves `$SKILL_DIR` from where it was Read instead of the Claude-Code-only `${CLAUDE_SKILL_DIR}`, so script calls work on every host.
- `/watch` is now derived from `SKILL.md` frontmatter; the separate `commands/watch.md` wrapper was dropped to avoid a duplicate slash command.
- `balanced` now full-decodes to detect every scene cut across the whole video. The previous early-exit was faster but kept only the first cuts and dropped the tail of long videos.
- `token-burner` is exempt from the long-video "sparse scan" warning, since it keeps every scene-change frame.
- `--max-frames` is now an override on top of each mode's default cap, rather than a fixed default of 80.

### Fixed
- Non-Claude installs (`npx skills add`) were dead on arrival — the installer copied `SKILL.md` without the `scripts/` it shells out to. The self-contained package layout resolves this.

### Removed
- `V2_PLAN.md` and `V2_CONCERNS.md` planning docs.

## [0.1.3] — 2026-05-09

### Fixed
- Windows: `video.info.json` is read as UTF-8 (#4). Previously `Path.read_text()` defaulted to cp1252 on Windows and crashed on yt-dlp's UTF-8 output, silently dropping Title/Uploader from the report. Same fix applied to `.env` reads/writes in `whisper.py` and `setup.py`.
- `download.py` now logs info.json parse failures to stderr instead of swallowing them.

### Security
- Hardened subprocess argv against option injection (#2): inserted `--` before the URL in the yt-dlp argv, and tightened `is_url` to reject `-`-prefixed sources and require a non-empty netloc. Resolved video/audio paths to absolute via `Path.resolve()` before passing to `ffmpeg`/`ffprobe`, so a relative path starting with `-` can't be misinterpreted as a flag.

## [0.1.2] — 2026-04-24

### Fixed
- Windows console crash: removed the emoji from the long-video warning in `watch.py`; cp1252 consoles couldn't encode it.
- `setup.py` now prints `winget` / `pip` install commands on Windows instead of "unsupported platform" — matches what the README already promised.

### Changed
- `SKILL.md` notes that on Windows the scripts must be invoked with `python`, not `python3` (the latter is the Microsoft Store stub on Windows).

## [0.1.1] — 2026-04-24

### Fixed
- Added `commands/watch.md` shim so `/watch` is callable when installed as a Claude Code plugin. Without it, the plugin loaded but the skill wasn't exposed as a slash command.
- `scripts/build-skill.sh` now strips `commands/` from the claude.ai `.skill` bundle alongside `hooks/` and `.claude-plugin/`.

## [0.1.0] — 2026-04-24

Initial marketplace release.

### Added
- `/watch <url-or-path> [question]` slash command.
- yt-dlp download with native caption extraction (manual + auto-subs).
- ffmpeg frame extraction with auto-scaled fps (≤2 fps, ≤100 frames, duration-aware budget).
- `--start` / `--end` focused mode with denser frame budget and transcript range filtering.
- Whisper fallback (Groq preferred, OpenAI secondary) for videos without captions.
- `setup.py` preflight: silent `--check`, structured `--json`, and installer that auto-runs `brew install` on macOS.
- Session-start hook that prints a one-line status on first run / partial config.
- `.skill` bundle packaging for claude.ai upload via `scripts/build-skill.sh`.
