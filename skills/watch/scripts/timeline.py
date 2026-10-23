#!/usr/bin/env python3
"""One chronological timeline: frame markers interleaved with merged speech lines.

The timeline is the run's text artifact. Frames are referenced by file name, so the
agent reads the text first and opens pixels for the moments that matter; later
visual backends attach descriptions to the same FRAME lines.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

VERSION = 1
# Merged-line bounds: long enough to carry a sentence, short enough to cite precisely.
MAX_LINE_WORDS = 40
MAX_LINE_SECONDS = 20.0
MIN_SENTENCE_WORDS = 6
PAUSE_SECONDS = 1.5
DESCRIPTION_CHARS = 600
MAX_CHAPTERS = 100
SENTENCE_END = re.compile(r'[.?!…。？！]["\')\]]*$')


def clock(seconds: float) -> str:
    """Floor to whole seconds so a cited line never points after its first word."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    return f'{hours}:{minutes:02d}:{sec:02d}' if hours else f'{minutes:02d}:{sec:02d}'


def merge_lines(segments: list[dict], breaks: list[float] = ()) -> list[dict]:
    """Join caption/ASR fragments into sentence-sized lines.

    A line ends at sentence punctuation (once it has a few words), a pause, a size
    bound, a YouTube ``>>`` speaker change, or a frame boundary in ``breaks`` so
    every FRAME marker lands between lines rather than inside one.
    """
    cuts = sorted(set(breaks))
    lines: list[dict] = []
    current = None
    for seg in segments:
        text = ' '.join(seg['text'].split())
        if not text:
            continue
        if current is not None:
            words = len(current['text'].split())
            crossed = any(current['start'] < cut <= seg['start'] for cut in cuts)
            if (crossed or seg['start'] - current['end'] > PAUSE_SECONDS or text.startswith('>>')
                    or words + len(text.split()) > MAX_LINE_WORDS
                    or seg['end'] - current['start'] > MAX_LINE_SECONDS
                    or (SENTENCE_END.search(current['text']) and words >= MIN_SENTENCE_WORDS)):
                lines.append(current)
                current = None
        if current is None:
            current = {'start': seg['start'], 'end': seg['end'], 'text': text}
        else:
            current['text'] += ' ' + text
            current['end'] = max(current['end'], seg['end'])
    if current is not None:
        lines.append(current)
    return lines


def chapters_from(info: dict, probe: dict | None = None) -> list[dict]:
    """Normalize yt-dlp ``chapters`` or ffprobe ``chapters`` into start/title pairs."""
    out = []
    for raw in (info.get('chapters') or (probe or {}).get('chapters') or [])[:MAX_CHAPTERS]:
        if not isinstance(raw, dict):
            continue
        title = raw.get('title') or (raw.get('tags') or {}).get('title')
        try:
            start = float(raw.get('start_time'))
        except (TypeError, ValueError):
            continue
        if title and start >= 0:
            out.append({'start': start, 'title': ' '.join(str(title).split())})
    return out


def short_description(text) -> str | None:
    text = ' '.join(str(text or '').split())
    if len(text) > DESCRIPTION_CHARS:
        text = text[:DESCRIPTION_CHARS].rsplit(' ', 1)[0] + ' …'
    return text or None


def build(*, frames: list[dict], lines: list[dict], chapters: list[dict] | None = None,
          description: str | None = None, header: dict | None = None,
          speech_source: str | None = None) -> dict:
    return {
        'version': VERSION,
        **(header or {}),
        'chapters': chapters or [],
        'description': description,
        'speech_source': speech_source,
        'frames': [{'file': Path(f['path']).name, 't': round(f['timestamp_seconds'], 3),
                    'reason': f.get('reason', 'selected')} for f in frames],
        'lines': [{'start': round(l['start'], 3), 'end': round(l['end'], 3), 'text': l['text']} for l in lines],
    }


def events(data: dict) -> list[str]:
    """Chronological text rows; a frame sorts before speech starting the same instant."""
    rows = [(f['t'], 0, f"[{clock(f['t'])}] FRAME {f['file']} ({f['reason']})") for f in data['frames']]
    rows += [(l['start'], 1, f"[{clock(l['start'])}] {l['text']}") for l in data['lines']]
    return [row for _, _, row in sorted(rows, key=lambda r: (r[0], r[1]))]


def render_sections(data: dict, frames_dir: Path | None) -> str:
    """The Chapters/Description/Timeline sections shared by the report and timeline.md."""
    out = []
    if data['chapters']:
        out += ['## Chapters', '']
        out += [f"- {clock(c['start'])} {c['title']}" for c in data['chapters']]
        out.append('')
    if data['description']:
        out += ['## Description', '', data['description'], '']
    out += ['## Timeline', '']
    if data['frames'] and frames_dir is not None:
        out.append(f'Frames live at: `{frames_dir}`')
        out.append('')
        out.append('**Read every `FRAME` file below from that directory with the Read tool.** '
                   '`[MM:SS]` is the absolute source time; a frame shows what is on screen from its '
                   'time until the next FRAME line. Other lines are speech.')
        out.append('')
    if data['speech_source']:
        out.append(f"_Speech: {data['speech_source']}._")
        out.append('')
    rows = events(data)
    if rows:
        out += ['```', *rows, '```']
    else:
        out.append('_No frames or speech in this range._')
    return '\n'.join(out)


def write(data: dict, directory: Path, markdown: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'timeline.json').write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    (directory / 'timeline.md').write_text(markdown, encoding='utf-8')
