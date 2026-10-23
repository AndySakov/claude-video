#!/usr/bin/env python3
"""Persistent timeline cache: one directory per (source, options) with frames + timeline.

Entries hold only text and the selected JPEG frames, never media or audio. Keys mix
the source identity with every option that changes the evidence, so a different
detail, range, or language is a different entry. Writes go to a sibling temporary
directory and are renamed into place, so a reader never sees a half-written entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from download import is_url

KEY_VERSION = 1
MAX_ENTRIES = 50
DISABLED = {'off', 'none', 'false', '0', 'no'}


def default_root() -> Path:
    return Path.home() / '.cache' / 'watch' / 'timelines'


def resolve_root(setting: str | None) -> Path | None:
    """``WATCH_CACHE_DIR``: empty → default location; off/none/0 → disabled."""
    value = (setting or '').strip()
    if value.lower() in DISABLED:
        return None
    return Path(value).expanduser().resolve() if value else default_root()


def source_identity(source: str) -> dict | None:
    """URLs key on the URL; local files on resolved path, size, and mtime (edits miss)."""
    if is_url(source):
        return {'url': source.strip()}
    try:
        path = Path(source).expanduser().resolve()
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def run_key(source: str, options: dict) -> str | None:
    identity = source_identity(source)
    if identity is None:
        return None
    payload = json.dumps({'v': KEY_VERSION, 'source': identity, 'options': options}, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]


def lookup(root: Path, key: str) -> Path | None:
    entry = root / key
    if (entry / 'timeline.md').is_file() and (entry / 'timeline.json').is_file():
        try:
            os.utime(entry)  # Recency for pruning.
        except OSError:
            pass
        return entry
    return None


def begin(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='.tmp-', dir=root))


def commit(root: Path, key: str, staging: Path) -> Path:
    """Atomically publish ``staging`` as the entry for ``key``, replacing an older one."""
    entry = root / key
    if entry.exists():
        retired = Path(tempfile.mkdtemp(prefix='.old-', dir=root))
        entry.rename(retired / key)
        shutil.rmtree(retired, ignore_errors=True)
    staging.rename(entry)
    prune(root)
    return entry


def abandon(staging: Path) -> None:
    shutil.rmtree(staging, ignore_errors=True)


def prune(root: Path, keep: int = MAX_ENTRIES) -> None:
    """Drop least-recently-used entries past ``keep``; only touches watch-shaped entries."""
    entries = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')
               and (p / 'timeline.json').is_file()]
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in entries[keep:]:
        shutil.rmtree(stale, ignore_errors=True)
    day_ago = time.time() - 86400
    for leftover in root.glob('.tmp-*'):  # Interrupted runs.
        try:
            if leftover.stat().st_mtime < day_ago:
                shutil.rmtree(leftover, ignore_errors=True)
        except OSError:
            pass
