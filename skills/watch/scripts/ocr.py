#!/usr/bin/env python3
"""On-screen text as timed events: burned-in captions, slides, UI, and code.

A uniform OCR sample (adaptive rate, separate from the frames the agent views) is
read by Tesseract or, on macOS, Apple Vision. Lines are tracked across samples so
each piece of text is reported once, when it appears; word-by-word caption reveals
grow into one line instead of one line per word.

Tesseract gets two passes per sample: the plain grayscale frame (dark text on light
backgrounds) and a bright-text mask (white/yellow captions with dark outlines, the
dominant style on Reels, TikTok, and Shorts), which raw Tesseract cannot read over
moving video.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from frames import _emitted_frames, _scale_filter, sync_args
from runtime import diagnostic, run_text

CHOICES = ('auto', 'tesseract', 'vision', 'none')
OCR_RESOLUTION = 1280
MAX_SAMPLES = 600
# Short-form captions change every ~0.5 s, so short clips are sampled densely; measured on a
# real Reel, 3 fps recovered most caption words that 1 fps skipped. (duration ≤ s, fps)
AUTO_RATES = ((180.0, 3.0), (600.0, 2.0))
KEEP_SINGLE = {'I', 'a', 'A'}
BRIGHT_THRESHOLD = 220
MIN_WORD_CONF = 30.0  # Tesseract 0-100; low words are kept when their line is confident
MIN_LINE_CONF = 60.0
MIN_VISION_CONF = 0.3  # Apple Vision 0-1
MAX_LINES_PER_SAMPLE = 15
MAX_LINE_CHARS = 160
SAME_TEXT = 0.75  # SequenceMatcher ratio treated as the same on-screen line
MISSED_SAMPLES = 2  # OCR flicker tolerance before a line counts as gone
REOPEN_SECONDS = 2.0
VISION_SCRIPT = Path(__file__).with_name('ocr_vision.swift')


def resolve(choice: str) -> tuple[str | None, str | None]:
    """Return (backend, problem). ``auto`` quietly resolves to None when nothing is installed."""
    if choice not in CHOICES:
        raise SystemExit(f'--ocr must be one of {", ".join(CHOICES)}')
    if choice == 'none':
        return None, None
    if choice in ('auto', 'tesseract') and shutil.which('tesseract'):
        return 'tesseract', None
    vision_ok = sys.platform == 'darwin' and shutil.which('swift') and VISION_SCRIPT.is_file()
    if choice in ('auto', 'vision') and vision_ok:
        return 'vision', None
    if choice == 'tesseract':
        return None, 'tesseract is not installed (brew install tesseract / apt install tesseract-ocr)'
    if choice == 'vision':
        return None, 'Apple Vision OCR needs macOS with the swift command (xcode-select --install)'
    return None, None


def sample_rate(duration: float, fps: float | None = None) -> float:
    """``fps`` None means adaptive: dense for short clips, 1 fps for long ones, ≤ MAX_SAMPLES."""
    if fps is None:
        fps = next((rate for limit, rate in AUTO_RATES if duration <= limit), 1.0)
    return min(fps, MAX_SAMPLES / duration) if duration > 0 else fps


def clean(text: str) -> str:
    """Drop symbol-only tokens and stray single characters at the ends (texture and edge noise)."""
    tokens = [t for t in text.split() if any(ch.isalnum() for ch in t)]
    while tokens and len(tokens[-1]) == 1 and tokens[-1] not in KEEP_SINGLE:
        tokens.pop()
    while tokens and len(tokens[0]) == 1 and tokens[0] not in KEEP_SINGLE:
        tokens.pop(0)
    return ' '.join(tokens)


def extract_samples(video_path: str, out_dir: Path, *, rate: float, start: float, end: float,
                    mask: bool) -> list[dict]:
    """One decode pass: grayscale samples, plus bright-text masks when ``mask``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    select = (f"select='isnan(prev_selected_t)+gt(floor(t*{rate:.17g}),floor(prev_selected_t*{rate:.17g}))',"
              f"{_scale_filter(OCR_RESOLUTION)},format=gray,showinfo")
    if mask:
        graph = (f"[0:v]{select},split=2[raw][m];"
                 f"[m]lutyuv=y='if(gt(val,{BRIGHT_THRESHOLD}),0,255)'[mask]")
        outputs = ["-map", "[raw]", *sync_args(), "-q:v", "3", str(out_dir / "raw_%04d.jpg"),
                   "-map", "[mask]", *sync_args(), str(out_dir / "mask_%04d.png")]
    else:
        graph = f"[0:v]{select}[raw]"
        outputs = ["-map", "[raw]", *sync_args(), "-q:v", "3", str(out_dir / "raw_%04d.jpg")]
    result = run_text(["ffmpeg", "-hide_banner", "-loglevel", "info", "-y", "-ss", f"{start:.6f}",
                       "-to", f"{end:.6f}", "-i", str(Path(video_path).resolve()),
                       "-filter_complex", graph, *outputs])
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg OCR sampling failed: {diagnostic(result.stderr)}")
    raws = sorted(out_dir.glob("raw_*.jpg"))
    if not raws:
        return []
    samples = _emitted_frames(raws, result.stderr, start, "ocr")
    for sample in samples:
        twin = out_dir / Path(sample["path"]).name.replace("raw_", "mask_").replace(".jpg", ".png")
        sample["mask"] = str(twin) if mask and twin.is_file() else None
    return samples


def plausible(text: str) -> bool:
    """Reject OCR confetti: short or mostly-symbol fragments from textures and edges."""
    dense = text.replace(' ', '')
    alnum = sum(ch.isalnum() for ch in dense)
    return alnum >= 3 and alnum / max(1, len(dense)) >= 0.6 and any(len(w) >= 2 for w in re.findall(r'\w+', text))


def parse_tsv(tsv: str) -> list[dict]:
    """Group Tesseract words into visual lines by position, not block ids.

    Sparse-text mode (used on caption masks) files many words as separate blocks,
    so ids would split one caption into fragments. Words join a line when their
    vertical centres agree and the horizontal gap is word-sized.
    """
    words = []
    for row in tsv.splitlines()[1:]:
        cols = row.split('\t')
        if len(cols) < 12 or cols[0] != '5' or not cols[11].strip():
            continue
        try:
            left, top, width, height, conf = (float(c) for c in cols[6:11])
        except ValueError:
            continue
        if conf >= MIN_WORD_CONF and height > 0:
            words.append({'text': cols[11].strip(), 'conf': conf, 'x': left, 'r': left + width,
                          'cy': top + height / 2, 'h': height})
    lines: list[list[dict]] = []
    # Left to right, each word extends the row whose right end it continues; descenders
    # shift a word's centre, so rows compare against their mean centre.
    for word in sorted(words, key=lambda w: w['x']):
        best, best_offset = None, None
        for line in lines:
            height = max(word['h'], max(w['h'] for w in line))
            offset = abs(word['cy'] - sum(w['cy'] for w in line) / len(line))
            gap = word['x'] - max(w['r'] for w in line)
            if offset <= 0.5 * height and -0.5 * height <= gap <= 2.5 * height and (best is None or offset < best_offset):
                best, best_offset = line, offset
        if best is None:
            lines.append([word])
        else:
            best.append(word)
    out = []
    for line in lines:
        line.sort(key=lambda w: w['x'])
        text = clean(' '.join(w['text'] for w in line))[:MAX_LINE_CHARS]
        conf = sum(w['conf'] for w in line) / len(line)
        if conf >= MIN_LINE_CONF and plausible(text):
            out.append({'text': text, 'conf': conf / 100, 'y': min(w['cy'] for w in line)})
    return sorted(out, key=lambda l: l['y'])


def _tesseract(path: str, lang: str, psm: str) -> list[dict]:
    env = {**os.environ, 'OMP_THREAD_LIMIT': '1'}  # Parallelism comes from the pool, not OpenMP.
    result = run_text(['tesseract', path, 'stdout', '-l', lang, '--psm', psm, 'tsv'], timeout=120, env=env)
    if result.returncode != 0:
        raise SystemExit(f'tesseract failed: {diagnostic(result.stderr, 400)}')
    return parse_tsv(result.stdout)


def merge_passes(*passes: list[dict]) -> list[str]:
    """Union of passes for one sample, near-duplicates collapsed, top-to-bottom order."""
    kept: list[dict] = []
    for line in sorted((l for p in passes for l in p), key=lambda l: -l['conf']):
        if not any(similar(norm(line['text']), norm(k['text'])) for k in kept):
            kept.append(line)
    kept = kept[:MAX_LINES_PER_SAMPLE]
    return [l['text'] for l in sorted(kept, key=lambda l: l['y'])]


def read_tesseract(samples: list[dict], lang: str) -> list[list[str]]:
    def one(sample):
        passes = [_tesseract(sample['path'], lang, '3')]
        if sample.get('mask'):
            passes.append(_tesseract(sample['mask'], lang, '11'))
        return merge_passes(*passes)
    with ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 1))) as pool:
        return list(pool.map(one, samples))


def read_vision(samples: list[dict]) -> list[list[str]]:
    paths = '\n'.join(s['path'] for s in samples) + '\n'
    result = run_text(['swift', str(VISION_SCRIPT)], input=paths, timeout=60 + 3 * len(samples))
    if result.returncode != 0:
        raise SystemExit(f'Apple Vision OCR failed: {diagnostic(result.stderr, 400)}')
    by_path = {}
    for row in result.stdout.splitlines():
        try:
            item = json.loads(row)
            by_path[item['path']] = [
                {'text': clean(str(l['text']))[:MAX_LINE_CHARS], 'conf': float(l['conf']), 'y': float(l['y'])}
                for l in item['lines'] if float(l['conf']) >= MIN_VISION_CONF and plausible(clean(str(l['text'])))]
        except (ValueError, KeyError, TypeError):
            continue
    if samples and not by_path:
        raise SystemExit('Apple Vision OCR returned no readable results')
    return [merge_passes(by_path.get(s['path'], [])) for s in samples]


def norm(text: str) -> str:
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.casefold()).split())


def similar(a: str, b: str) -> bool:
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= SAME_TEXT


def track(samples: list[tuple[float, list[str]]], interval: float) -> list[dict]:
    """Turn per-sample lines into text events reported once, at first appearance."""
    active: list[dict] = []
    retired: list[dict] = []

    def match(pool, key):
        for item in pool:
            # Word-by-word reveals extend a prefix; OCR that drops words reads a fragment.
            if similar(item['norm'], key) or key.startswith(item['norm'] + ' '):
                return item, 'grow'
            if len(key) >= 6 and key in item['norm']:
                return item, 'seen'
        return None, None

    for t, lines in samples:
        for text in lines:
            key = norm(text)
            if not key:
                continue
            item, how = match(active, key)
            if item is None:
                item, how = match([r for r in retired if t - r['last'] <= REOPEN_SECONDS], key)
                if item is not None:
                    retired.remove(item)
                    active.append(item)
            if item is None:
                active.append({'start': t, 'last': t, 'text': text, 'norm': key})
                continue
            item['last'] = t
            if how == 'grow' and len(key) > len(item['norm']):
                item['text'], item['norm'] = text, key
        for item in [a for a in active if t - a['last'] > interval * (MISSED_SAMPLES + 0.5)]:
            active.remove(item)
            retired.append(item)
    events = sorted(({'start': e['start'], 'end': e['last'] + interval, 'text': e['text'], 'norm': e['norm']}
                     for e in retired + active), key=lambda e: e['start'])
    return [{k: e[k] for k in ('start', 'end', 'text')} for e in absorb_reveals(events, interval)]


def contained(part: str, whole: str) -> bool:
    """``part`` reads as a (possibly misread) piece of ``whole``; scattered letters do not count."""
    if len(part) < 3 or len(part) > len(whole):
        return False
    blocks = difflib.SequenceMatcher(None, part, whole, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= 2) >= 0.7 * len(part)


def absorb_reveals(events: list[dict], interval: float) -> list[dict]:
    """Fold a caption's partial reveals ("THIS", "THIS IS") into the line they grow into."""
    out: list[dict] = []
    for event in events:
        for earlier in [e for e in out if event['start'] - e['end'] <= interval * 1.5
                        and contained(e['norm'], event['norm'])]:
            out.remove(earlier)
            event['start'] = min(event['start'], earlier['start'])
        out.append(event)
    return sorted(out, key=lambda e: e['start'])


def on_screen_text(video_path: str, work: Path, backend: str, *, lang: str, fps: float | None,
                   start: float, end: float) -> tuple[list[dict], dict]:
    rate = sample_rate(end - start, fps)
    directory = work / 'ocr'
    try:
        samples = extract_samples(video_path, directory, rate=rate, start=start, end=end,
                                  mask=backend == 'tesseract')
        texts = read_tesseract(samples, lang) if backend == 'tesseract' else read_vision(samples)
    finally:
        shutil.rmtree(directory, ignore_errors=True)  # Samples are inputs, not evidence.
    events = track([(s['timestamp_seconds'], lines) for s, lines in zip(samples, texts)], 1 / rate)
    return events, {'backend': backend, 'samples': len(samples), 'rate': rate}
