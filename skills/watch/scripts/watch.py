#!/usr/bin/env python3
"""/watch entry point: download video, extract frames, parse transcript.

Prints a markdown report to stdout listing frame paths + transcript. Claude
then Reads each frame path to see the video.
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from config import ConfigError, frame_cap, get_config, load_gemini_key, resolve_engine  # noqa: E402
import gemini  # noqa: E402
from download import download, fetch_captions, is_url, auth_args  # noqa: E402
from frames import MAX_FPS, auto_fps, auto_fps_focus, extract_at_timestamps, extract_keyframes, extract_scene_or_uniform, format_time, get_metadata, merge_frames, parse_time, parse_timestamps, validate_controls  # noqa: E402
from transcribe import filter_range, parse_vtt  # noqa: E402
import cache  # noqa: E402
import timeline  # noqa: E402
import ocr  # noqa: E402
from whisper import load_api_key, transcribe_video  # noqa: E402
from runtime import configure_stdio  # noqa: E402


LOCAL_ONLY_FLAGS = (("--detail", "detail"), ("--fps", "fps"), ("--max-frames", "max_frames"),
                    ("--timestamps", "timestamps"), ("--whisper", "whisper"), ("--sub-lang", "sub_lang"),
                    ("--ocr", "ocr"))


def run_gemini(args, config, key, start_sec, end_sec, auth) -> int:
    """Google watches the video. No captions, frames, or Whisper; no fallback on failure."""
    ignored = [flag for flag, name in LOCAL_ONLY_FLAGS if getattr(args, name) is not None]
    ignored += [flag for flag, on in (("--no-whisper", args.no_whisper), ("--no-dedup", args.no_dedup)) if on]
    clip = (start_sec, end_sec) if start_sec is not None or end_sec is not None else None
    uploaded, warning, result, error, sent, work = None, None, None, None, "URL sent to Google", None
    try:
        if gemini.is_youtube(args.source):
            video = {"uri": args.source}
        else:
            parent = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
            if parent:
                parent.mkdir(parents=True, exist_ok=True)
            sent = "not sent to Google"
            work = Path(tempfile.mkdtemp(prefix="watch-", dir=parent))
            print("[watch] downloading media…" if is_url(args.source) else "[watch] using local file…", file=sys.stderr)
            media = download(args.source, work / "download", **(auth if is_url(args.source) else {}))
            sent = "upload to Google failed"
            print("[watch] uploading to the Gemini Files API…", file=sys.stderr)
            uploaded = gemini.upload_file(Path(media["video_path"]), key)
            video = {"uri": uploaded["uri"], "mime_type": uploaded["mime_type"]}
            sent = "video uploaded to Google, deleted after the answer"
        print(f"[watch] asking {config['gemini_model']}…", file=sys.stderr)
        result = gemini.ask(video, args.question, model=config["gemini_model"], key=key,
                            clip=clip, timeout=config["gemini_timeout"])
    except SystemExit as exc:
        error = str(exc)
    finally:
        if uploaded:
            warning = gemini.delete_file(uploaded["name"], key)
        if work:  # Run-owned; holds at most a downloaded copy. A local source file lives elsewhere.
            shutil.rmtree(work, ignore_errors=True)

    print()
    print("# watch: video report")
    print()
    print(f"- **Source:** {args.source} ({sent})")
    print(f"- **Engine:** {config['gemini_model']} ({result['processing'] if result else 'failed'})")
    if clip:
        print(f"- **Focus range:** {format_time(start_sec or 0)} → {format_time(end_sec) if end_sec is not None else 'end'}")
    if ignored:
        print(f"- **Ignored local options:** {', '.join(ignored)} (these only apply with --engine local)")
    if result and result["total_tokens"] is not None:
        print(f"- **Gemini tokens:** {result['total_tokens']}")
    if warning:
        print(f"- **Cleanup warning:** {warning}")
    print()
    if error:
        print("## Unavailable evidence")
        print()
        print(f"- {error}")
        return 1
    print("## Answer (from Gemini)")
    print()
    print("_These are Gemini's observations of the video, not frames you viewed yourself. "
          "Relay them as such; rerun with `--engine local` to inspect frames directly._")
    print()
    print(result["text"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="watch",
        description="Download a video, extract auto-scaled frames, and surface the transcript.",
    )
    ap.add_argument("source", help="Video URL or local file path")
    ap.add_argument("--max-frames", type=int, default=None, help="Override frame cap")
    ap.add_argument("--resolution", type=int, default=512, help="Frame width in pixels (default 512)")
    ap.add_argument("--fps", type=float, default=None, help="Override auto-fps")
    ap.add_argument(
        "--detail",
        choices=["transcript", "efficient", "balanced", "token-burner"],
        default=None,
        help="Fidelity/speed dial: transcript (no frames), efficient (fast keyframes, cap 50), "
             "balanced (scene, cap 100), token-burner (scene, uncapped).",
    )
    ap.add_argument(
        "--timestamps",
        type=str,
        default=None,
        help="Comma-separated absolute timestamps (SS, MM:SS, HH:MM:SS) to grab a frame at, "
             "e.g. transcript-flagged 'look here' moments. Added on top of the detail frames "
             "(reserved against the cap); with --detail transcript these become the only frames.",
    )
    ap.add_argument("--start", type=str, default=None, help="Range start (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--end", type=str, default=None, help="Range end (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--out-dir", type=str, default=None, help="Working directory (default: tmp)")
    ap.add_argument(
        "--no-whisper",
        action="store_true",
        help="Disable all local/cloud transcription fallbacks; native captions still work.",
    )
    ap.add_argument(
        "--whisper",
        choices=["groq", "openai", "whisperx"],
        default=None,
        help="Select the fallback backend for this run; native captions still come first.",
    )
    ap.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable near-duplicate frame removal. Keeps visually identical "
             "frames (static screen recordings, held slides) instead of collapsing them.",
    )
    ap.add_argument("--sub-lang", default=None, help="Exact caption language preference (default auto/native)")
    cookies = ap.add_mutually_exclusive_group()
    cookies.add_argument("--cookies", default=None, help="Explicit cookie file (yt-dlp may update this jar)")
    cookies.add_argument("--cookies-from-browser", default=None, help="Explicit yt-dlp browser selector")
    ap.add_argument("--engine", choices=["auto", "gemini", "local"], default=None,
                    help="gemini: Google watches the video and answers (needs GEMINI_API_KEY). "
                         "local: frames + transcript on this machine. Default auto: gemini when a key exists.")
    ap.add_argument("--question", default=None,
                    help="The user's question. Sent to Gemini on a gemini run; unused by the local engine.")
    ap.add_argument("--ocr", choices=list(ocr.CHOICES), default=None,
                    help="Read on-screen text (burned-in captions, slides, UI): auto (tesseract, else Apple "
                         "Vision on macOS, else off), tesseract, vision, or none.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Neither reuse nor save a cached timeline for this run.")
    ap.add_argument("--refresh", action="store_true",
                    help="Rebuild the timeline even when a cached one exists, then replace it.")
    args = ap.parse_args()
    if args.no_whisper and args.whisper:
        ap.error("--no-whisper conflicts with --whisper")

    config = get_config(backend_override="none" if args.no_whisper else args.whisper)
    detail = args.detail or config["detail"]
    max_frames = args.max_frames if args.max_frames is not None else frame_cap(detail)
    budget_cap = max_frames if max_frames is not None else 100
    start_sec, end_sec = parse_time(args.start), parse_time(args.end)
    validate_controls(args.resolution, max_frames, start_sec, end_sec, args.fps)
    cue_timestamps = parse_timestamps(args.timestamps)
    backend_choice = "none" if args.no_whisper else args.whisper or config["whisper_backend"]
    cookies_file = args.cookies if args.cookies is not None else (None if args.cookies_from_browser else config["cookies_file"])
    cookies_browser = args.cookies_from_browser if args.cookies_from_browser is not None else (None if args.cookies else config["cookies_from_browser"])
    auth_args(cookies_file, cookies_browser)  # Validate even before starting network work.
    auth = {"cookies_file": cookies_file, "cookies_from_browser": cookies_browser}

    gemini_key = load_gemini_key()
    if resolve_engine(args.engine or config["engine"], bool(gemini_key)) == "gemini":
        return run_gemini(args, config, gemini_key, start_sec, end_sec, auth)

    sub_lang = args.sub_lang or config["sub_lang"]
    ocr_choice = args.ocr or config["ocr"]
    ocr_backend, ocr_problem = ocr.resolve(ocr_choice)
    cache_root = None if args.no_cache else cache.resolve_root(config["cache_dir"])
    # Every option that changes the evidence is part of the key; --question is not (unused locally).
    cache_key = cache_root and cache.run_key(args.source, {
        "timeline": timeline.VERSION, "detail": detail, "max_frames": max_frames, "resolution": args.resolution,
        "fps": args.fps, "start": start_sec, "end": end_sec, "timestamps": cue_timestamps,
        "dedup": not args.no_dedup, "sub_lang": sub_lang, "backend": backend_choice,
        "whisperx": [config["whisperx_model"], config["whisperx_language"]] if backend_choice == "whisperx" else None,
        "ocr": [ocr_backend, config["ocr_lang"], config["ocr_fps"]] if ocr_backend else None,
    })
    hit = cache.lookup(cache_root, cache_key) if cache_key and not args.refresh else None
    if hit:
        try:
            cached = (hit / "timeline.md").read_text(encoding="utf-8")
        except OSError:
            hit = None  # An unreadable entry is rebuilt and replaced below.
        else:
            print("[watch] reusing cached timeline", file=sys.stderr)
            print(cached)
            print()
            print("---")
            print(f"_Cached timeline: `{hit}` — reused without downloading; pass `--refresh` to rebuild._")
            return 0

    parent = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if parent:
        parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="watch-", dir=parent))
    print(f"[watch] working dir: {work}", file=sys.stderr)
    url_source = is_url(args.source)
    dl = {"subtitle_path": None, "info": {}, "downloaded": False}
    errors = []
    all_segments = []
    track_available = False
    transcript_source = None
    video_path = None
    visual_error = None
    gaps = []

    if url_source:
        print("[watch] checking metadata/captions via yt-dlp…", file=sys.stderr)
        dl = fetch_captions(args.source, work / "download", sub_lang=sub_lang, **auth)
        errors.extend(dl.get("errors", []))
        if dl.get("subtitle_path"):
            try:
                all_segments = parse_vtt(dl["subtitle_path"])
                track_available = bool(all_segments)
                track = dl.get("caption_track") or {}
                transcript_source = (f"captions ({track.get('language', 'unknown')}, "
                                     f"{track.get('kind', 'unknown')}, {track.get('provenance', 'unknown')})")
            except (OSError, ValueError) as exc:
                errors.append(f"Caption parsing failed: {exc}")

    text_only = detail == "transcript" and not cue_timestamps
    # Without captions, burned-in text may be the only words on screen: fetch pixels for OCR.
    audio_only = text_only and not (ocr_backend and not track_available)
    # Captions are assessed before range filtering: a quiet interval is not a missing track.
    need_media = not (audio_only and (track_available or backend_choice == "none") and url_source)
    if need_media:
        try:
            print("[watch] downloading media…" if url_source else "[watch] using local file…", file=sys.stderr)
            media = download(args.source, work / "download", audio_only=audio_only,
                             **({"context": dl, **auth} if url_source else {}))
            dl.update(media)
            video_path = dl["video_path"]
        except SystemExit as exc:
            visual_error = f"Media unavailable: {exc}"
            errors.append(visual_error)

    try:
        duration = float((dl.get("info") or {}).get("duration") or 0)
        if not math.isfinite(duration) or duration < 0:
            duration = 0.0
    except (TypeError, ValueError):
        duration = 0.0
    meta = {"duration_seconds": duration, "width": None, "height": None, "codec": None,
            "has_audio": False, "has_video": False}
    if video_path:
        try:
            meta = get_metadata(video_path)
        except SystemExit as exc:
            visual_error = f"Visuals/audio metadata unavailable: {exc}"
            errors.append(visual_error)
    full_duration = meta["duration_seconds"]
    if full_duration > 0 and start_sec is not None and start_sec >= full_duration:
        raise SystemExit(f"--start {start_sec:.1f}s is past end of video ({full_duration:.1f}s)")
    effective_start = start_sec or 0.0
    effective_end = min(end_sec, full_duration) if end_sec is not None and full_duration > 0 else end_sec or full_duration
    effective_duration = max(0.0, effective_end - effective_start)
    focused = start_sec is not None or end_sec is not None
    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=budget_cap)
    else:
        fps, target = auto_fps(effective_duration, max_frames=budget_cap)
    if args.fps is not None:
        fps = min(args.fps, MAX_FPS)
        target = max(1, int(round(fps * effective_duration)))

    frames, cue_frames = [], []
    frame_meta = {"engine": "none", "candidate_count": 0, "selected_count": 0, "fallback": False}
    cue_meta = {}
    detail_budget = max_frames
    if video_path and meta.get("has_video", bool(meta.get("width"))):
        try:
            if cue_timestamps:
                cue_frames, cue_meta = extract_at_timestamps(
                    video_path, work / "frames", cue_timestamps, resolution=args.resolution,
                    max_frames=max_frames, start_seconds=start_sec, end_seconds=effective_end or end_sec)
            detail_budget = None if max_frames is None else max_frames - len(cue_frames)
            if detail != "transcript" and detail_budget != 0:
                kwargs = dict(resolution=args.resolution, max_frames=detail_budget,
                              start_seconds=start_sec, end_seconds=effective_end or end_sec, dedup=not args.no_dedup)
                if detail == "efficient":
                    frames, frame_meta = extract_keyframes(video_path, work / "frames", **kwargs)
                else:
                    frames, frame_meta = extract_scene_or_uniform(video_path, work / "frames", fps=fps, target_frames=target, **kwargs)
        except SystemExit as exc:
            visual_error = f"Visual extraction unavailable: {exc}"
            errors.append(visual_error)
    elif video_path and not visual_error and (detail != "transcript" or cue_timestamps):
        visual_error = "No video stream; visual evidence is unavailable."
    if cue_frames:
        frames = merge_frames(frames, cue_frames)

    text_events, ocr_meta = [], None
    readable = bool(video_path and meta.get("has_video") and effective_end > effective_start)
    if ocr_problem and readable:
        errors.append(f"On-screen text unavailable: {ocr_problem}")
    elif ocr_backend and readable:
        print(f"[watch] reading on-screen text ({ocr_backend})…", file=sys.stderr)
        try:
            text_events, ocr_meta = ocr.on_screen_text(
                video_path, work, ocr_backend, lang=config["ocr_lang"], fps=config["ocr_fps"],
                start=effective_start, end=effective_end)
        except SystemExit as exc:
            errors.append(f"On-screen text unavailable: {exc}")

    transcript_state = "missing"
    if not track_available and backend_choice != "none" and video_path and meta.get("has_audio"):
        backend, api_key = ("whisperx", None) if backend_choice == "whisperx" else load_api_key(None if backend_choice == "auto" else backend_choice)
        if backend:
            try:
                all_segments, used_backend = transcribe_video(video_path, work / "audio.mp3", backend=backend, api_key=api_key)
                gaps = getattr(all_segments, "gaps", [])
                track_available = True
                transcript_state = "no speech" if not all_segments else "available"
                if used_backend == "whisperx":
                    language = config["whisperx_language"] or "auto (unverified)"
                    transcript_source = f"whisper (whisperx {config['whisperx_model']}, language {language})"
                else:
                    transcript_source = f"whisper ({used_backend})"
            except SystemExit as exc:
                transcript_state = "failed"
                errors.append(f"Transcription failed: {exc}")
        else:
            transcript_state = f"unavailable: no matching API key for {backend_choice}"
    elif not track_available and backend_choice == "none":
        transcript_state = "fallback disabled; no captions available"
    elif not track_available and video_path and not meta.get("has_audio"):
        transcript_state = "no audio stream" if not visual_error else "audio metadata unavailable"

    transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
    if track_available:
        transcript_state = "available" if transcript_segments else ("no speech in selected range" if focused and all_segments else "no speech")
    for error in errors:
        print(f"[watch] {error}", file=sys.stderr)

    info = dl.get("info") or {}
    head = []
    p = head.append
    p("# watch: video report")
    p("")
    p(f"- **Source:** {args.source}")
    if visual_error:
        p(f"- **Visual status:** {visual_error}")
    if errors:
        p("- **Result:** partial evidence" if frames or transcript_segments else "- **Result:** unavailable evidence")
    if info.get("title"):
        p(f"- **Title:** {info['title']}")
    if info.get("uploader"):
        p(f"- **Uploader:** {info['uploader']}")
    p(f"- **Duration:** {format_time(full_duration)} ({full_duration:.1f}s)")
    if focused:
        p(f"- **Focus range:** {format_time(effective_start)} → {format_time(effective_end)} "
          f"({effective_duration:.1f}s)")
    if meta.get("width") and meta.get("height"):
        p(f"- **Resolution:** {meta['width']}x{meta['height']} ({meta.get('codec') or 'unknown codec'})")
    range_mode = "focused" if focused else "full"
    p(f"- **Detail:** {detail}")
    detail_count = frame_meta.get("selected_count", 0)
    if detail != "transcript":
        cap_label = "unlimited" if detail_budget is None else str(detail_budget)
        engine = frame_meta.get("engine", "scene")
        fallback = " fallback" if frame_meta.get("fallback") else ""
        deduped = frame_meta.get("deduped_count", 0)
        dedup_note = f", {deduped} near-duplicate{'s' if deduped != 1 else ''} dropped" if deduped else ""
        p(f"- **Frames:** {detail_count} selected from {frame_meta.get('candidate_count', detail_count)} "
          f"candidates ({engine}{fallback}{dedup_note}, {range_mode} range, budget {target}, cap {cap_label})")
    elif not cue_frames:
        p("- **Frames:** skipped (transcript detail)")
    if cue_frames:
        dropped = cue_meta.get("dropped_out_of_window", 0)
        drop_note = f", {dropped} dropped outside range" if dropped else ""
        p(f"- **Cue frames:** {len(cue_frames)} at transcript-flagged timestamps (transcript-cue{drop_note})")
    if frames:
        p(f"- **Frame size:** max {args.resolution}px wide, max 1998px tall")
    lines = timeline.merge_lines(transcript_segments, [f["timestamp_seconds"] for f in frames])
    if ocr_meta:
        heard = len(text_events)
        text_events = ocr.drop_spoken(text_events, lines)
        captions = sum(e.get("kind") == "caption" for e in text_events)
        counts = ", ".join(part for part in (
            f"{captions} caption line{'s' if captions != 1 else ''}" if captions else "",
            f"{len(text_events) - captions} text line{'s' if len(text_events) - captions != 1 else ''}"
            if len(text_events) > captions else "",
            f"{heard - len(text_events)} spoken captions dropped" if heard > len(text_events) else "") if part)
        p(f"- **On-screen text:** {counts or 'none detected'} (OCR via {ocr_meta['backend']}, "
          f"{ocr_meta['samples']} samples at {ocr_meta['rate']:.2g} fps)")
    elif not ocr_backend and not ocr_problem and video_path and meta.get("has_video"):
        p("- **On-screen text:** off" + (" (no OCR engine installed; install tesseract to read captions and slides)"
                                         if ocr_choice == "auto" else ""))
    if transcript_segments:
        in_range = " in range" if focused else ""
        p(f"- **Transcript:** {len(lines)} lines{in_range} (via {transcript_source or 'captions'})")
    else:
        p(f"- **Transcript:** {transcript_state}")
    if gaps:
        p("- **Transcript status:** partial; missing intervals:")
        for gap in gaps:
            end = gap["end"] if gap["end"] is not None else full_duration
            p(f"  - {format_time(gap['start'])} → {format_time(end)}")
    if detail == "token-burner" and len(frames) > 250:
        p("")
        p(f"> **Warning:** token-burner detail selected {len(frames)} frames. "
          "This may use a large number of image tokens.")
    if not focused and full_duration > 600 and detail not in ("transcript", "token-burner"):
        mins = int(full_duration // 60)
        p("")
        p(f"> **Warning:** This is a {mins}-minute video. Frame coverage is sparse at this length "
          f"under `{detail}` detail — its cap spreads thin across the full clip. For better results, "
          "re-run with `--start HH:MM:SS --end HH:MM:SS` to zoom into a section, or use "
          "`--detail token-burner` to keep every scene-change frame across the whole video.")

    speech = None
    if transcript_segments:
        label = transcript_source or "captions"
        speech = (f"{label}, filtered to {format_time(effective_start)} → {format_time(effective_end)}"
                  if focused else label)
    data = timeline.build(
        frames=frames, lines=lines, text=text_events, speech_source=speech,
        chapters=timeline.chapters_from(info, meta), description=timeline.short_description(info.get("description")),
        header={"source": args.source, "title": info.get("title"), "uploader": info.get("uploader"),
                "duration": round(full_duration, 3), "detail": detail, "transcript": transcript_state,
                "focus": [effective_start, effective_end] if focused else None, "gaps": gaps})

    def render(frames_dir: Path) -> str:
        body = ["\n".join(head), "", timeline.render_sections(data, frames_dir)]
        if not track_available and detail == "transcript" and not transcript_segments:
            body += ["", "_Re-run with `--detail balanced` for visual evidence, or configure a transcription backend with setup.py._"]
        if errors:
            body += ["", "## Unavailable evidence", *[f"- {error}" for error in errors]]
        return "\n".join(body)

    # Only complete, reproducible evidence is cached: a transient failure or a missing key
    # must not be replayed after it is fixed.
    entry = None
    if cache_key and not errors and not transcript_state.startswith("unavailable"):
        staging = None
        try:
            staging = cache.begin(cache_root)
            if frames:
                shutil.copytree(work / "frames", staging / "frames")
            timeline.write(data, staging, render(cache_root / cache_key / "frames"))
            entry = cache.commit(cache_root, cache_key, staging)
            shutil.rmtree(work / "frames", ignore_errors=True)
        except OSError as exc:
            if staging:
                cache.abandon(staging)
            print(f"[watch] cache not saved ({type(exc).__name__}: {exc})", file=sys.stderr)
    home = entry or work
    report = render(home / "frames")
    if not entry:
        timeline.write(data, work, report)

    print()
    print(report)
    print()
    print("---")
    if video_path:
        print(f"_Local media: `{video_path}`_")
    print(f"_Timeline: `{home / 'timeline.md'}` (+ `timeline.json`)"
          + (" — cached; reruns with the same options reuse it._" if entry else "._"))
    if entry and not any(path.is_file() for path in work.rglob("*")):
        shutil.rmtree(work, ignore_errors=True)  # Run-owned and empty: nothing left to clean up.
    else:
        print(f"_Work dir: `{work}` — delete when done._")

    return 1 if errors and not frames and not transcript_segments and not track_available else 0


if __name__ == "__main__":
    configure_stdio()
    try:
        raise SystemExit(main())
    except (ConfigError, OSError) as exc:
        raise SystemExit(str(exc)) from None
