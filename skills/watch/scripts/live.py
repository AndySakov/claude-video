#!/usr/bin/env python3
"""Record a fixed window of a live feed so the normal pipeline can watch it.

Three kinds of source:

- a live page that yt-dlp understands (YouTube, Twitch, a live ``.m3u8``): yt-dlp
  picks a ≤720p format and ffmpeg records it from the live edge;
- a raw stream URL ffmpeg reads itself (``rtsp://``, ``rtmp://``, ``srt://``, ``udp://``);
- a capture device, written ``<format>:<input>`` (``avfoundation:0`` for a Mac camera or
  screen, ``v4l2:/dev/video0``, ``dshow:video=Integrated Camera``, ``gdigrab:desktop``,
  ``x11grab::0.0``). Devices are only opened when named explicitly.

Each run records one window; rerunning gives the next one. Timeline times are seconds
into the window, and the report gives the window's wall-clock start.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from runtime import diagnostic, run_text

DEFAULT_WINDOW = 60.0
MAX_WINDOW = 3600.0
STREAM_SCHEMES = {'rtsp', 'rtsps', 'rtmp', 'rtmps', 'srt', 'udp', 'rtp', 'tcp'}
DEVICE_FORMATS = {'avfoundation', 'v4l2', 'dshow', 'gdigrab', 'x11grab'}
LIVE_FORMAT = 'b[height<=720]/bv*[height<=720]+ba/b/bv*+ba'
# Seconds allowed beyond the window for connecting, buffering, and finishing the file.
CAPTURE_SLACK = 90.0


def kind(source: str) -> str | None:
    """'device', 'stream', 'url', or None for a local path."""
    prefix, sep, rest = source.partition(':')
    if sep and prefix in DEVICE_FORMATS and rest:
        return 'device'
    parsed = urlparse(source)
    if parsed.scheme in STREAM_SCHEMES and parsed.netloc:
        return 'stream'
    if parsed.scheme in ('http', 'https') and parsed.netloc:
        return 'url'
    return None


def window_seconds(value: float | None) -> float:
    window = DEFAULT_WINDOW if value is None else value
    if not 1 <= window <= MAX_WINDOW:
        raise SystemExit(f'--window must be between 1 second and {MAX_WINDOW / 60:.0f} minutes')
    return window


def _headers(headers: dict) -> list[str]:
    lines = ''.join(f'{k}: {v}\r\n' for k, v in (headers or {}).items()
                    if isinstance(k, str) and isinstance(v, str) and '\n' not in k + v and '\r' not in k + v)
    return ['-headers', lines] if lines else []


def resolve_url(url: str, auth: list[str]) -> dict:
    """yt-dlp metadata plus ffmpeg inputs for the chosen live format(s)."""
    result = run_text(['yt-dlp', '--no-playlist', '-J', '-f', LIVE_FORMAT, *auth, '--', url], timeout=120)
    if result.returncode != 0:
        from download import network_diagnostic
        raise SystemExit(f'yt-dlp failed (exit {result.returncode}): {network_diagnostic(result.stderr)}')
    try:
        info = json.loads(result.stdout)
        formats = info.get('requested_formats') or [info]
        inputs = [(f['url'], f.get('http_headers') or info.get('http_headers') or {},
                   str(f.get('protocol') or '').startswith('m3u8')) for f in formats]
        if not inputs or not all(isinstance(u, str) and u for u, _, _ in inputs):
            raise ValueError
    except (ValueError, KeyError, TypeError, AttributeError):
        raise SystemExit('yt-dlp returned no playable live format for this URL.') from None
    return {'live': bool(info.get('is_live')), 'inputs': inputs,
            'info': {'title': info.get('title'), 'uploader': info.get('uploader') or info.get('channel'),
                     'url': info.get('webpage_url') or url, 'description': info.get('description')}}


def input_args(source: str, resolved: dict | None = None) -> list[str]:
    """ffmpeg input options for a device, a raw stream, or yt-dlp's resolved format(s)."""
    which = kind(source)
    if which == 'device':
        fmt, _, device = source.partition(':')
        return ['-f', fmt, *(['-framerate', '30'] if fmt in ('avfoundation', 'gdigrab', 'x11grab') else []), '-i', device]
    if which == 'stream':
        return [*(['-rtsp_transport', 'tcp'] if urlparse(source).scheme.startswith('rtsp') else []), '-i', source]
    args = []
    for url, headers, hls in (resolved or {}).get('inputs', []):
        # ffmpeg starts an HLS playlist three segments back; begin at the newest one instead.
        args += [*_headers(headers), *(['-live_start_index', '-1'] if hls else []), '-i', url]
    return args


def capture(source: str, out_dir: Path, window: float, resolved: dict | None = None) -> dict:
    """Record ``window`` seconds; returns the file, its wall-clock start, and recorded length."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = kind(source) == 'device'
    out = out_dir / ('live.mp4' if device else 'live.mkv')
    inputs = input_args(source, resolved)
    count = inputs.count('-i')
    maps = ['-map', '0:v:0?', '-map', f'{count - 1}:a:0?']
    # Streams are copied as delivered (Matroska takes any codec); devices deliver raw frames.
    codec = (['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', '-pix_fmt', 'yuv420p', '-c:a', 'aac']
             if device else ['-c', 'copy'])
    started = datetime.now(timezone.utc)
    result = run_text(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', *inputs, *maps,
                       '-t', f'{window:.3f}', *codec, str(out)], timeout=window + CAPTURE_SLACK)
    if result.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        detail = diagnostic(result.stderr, 800) or 'no media recorded'
        hint = ''
        if device and re.search(r'not authori[sz]ed|permission|denied', detail, re.I):
            hint = ' The terminal running watch needs camera or screen-recording permission.'
        raise SystemExit(f'Live capture failed (exit {result.returncode}): {detail}{hint}')
    return {'video_path': str(out), 'started': started.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'started_iso': started.isoformat(timespec='seconds')}
