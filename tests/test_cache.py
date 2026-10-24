"""Timeline cache: reuse, invalidation, opt-outs, and what is never cached."""
import os
import sys
from pathlib import Path

import pytest

import cache
import watch


def run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, 'argv', ['watch', *argv])
    code = watch.main()
    return code, capsys.readouterr().out


def entries(root: Path):
    return sorted(p for p in root.iterdir() if not p.name.startswith('.')) if root.exists() else []


def test_second_run_reuses_cached_timeline_without_work(monkeypatch, capsys, cut_clip):
    code, first = run(monkeypatch, capsys, str(cut_clip), '--no-whisper')
    assert code == 0 and 'cached; reruns' in first
    [entry] = entries(cache.default_root())
    assert list((entry / 'frames').glob('frame_*.jpg'))
    assert f'Frames live at: `{entry / "frames"}`' in first
    monkeypatch.setattr(watch, 'get_metadata', lambda *a: pytest.fail('a cache hit must not probe media'))
    code, second = run(monkeypatch, capsys, str(cut_clip), '--no-whisper')
    assert code == 0 and 'reused without downloading' in second
    assert second.split('\n---\n')[0].strip() == first.split('\n---\n')[0].strip()


def test_refresh_and_no_cache(monkeypatch, capsys, static_clip):
    run(monkeypatch, capsys, str(static_clip), '--no-whisper')
    _, out = run(monkeypatch, capsys, str(static_clip), '--no-whisper', '--refresh')
    assert 'reused' not in out and len(entries(cache.default_root())) == 1
    _, out = run(monkeypatch, capsys, str(static_clip), '--no-whisper', '--no-cache', '--detail', 'efficient')
    assert 'cached' not in out and len(entries(cache.default_root())) == 1


def test_cache_dir_setting(monkeypatch, capsys, static_clip, tmp_path):
    monkeypatch.setenv('WATCH_CACHE_DIR', str(tmp_path / 'mine'))
    run(monkeypatch, capsys, str(static_clip), '--no-whisper')
    assert len(entries(tmp_path / 'mine')) == 1
    monkeypatch.setenv('WATCH_CACHE_DIR', 'off')
    _, out = run(monkeypatch, capsys, str(static_clip), '--no-whisper')
    assert 'cached' not in out


def test_options_and_file_changes_miss(monkeypatch, capsys, static_clip, tmp_path):
    source = tmp_path / 'clip.mp4'
    source.write_bytes(static_clip.read_bytes())
    run(monkeypatch, capsys, str(source), '--no-whisper')
    run(monkeypatch, capsys, str(source), '--no-whisper', '--detail', 'efficient')
    assert len(entries(cache.default_root())) == 2
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    _, out = run(monkeypatch, capsys, str(source), '--no-whisper')
    assert 'reused' not in out and len(entries(cache.default_root())) == 3


def test_partial_evidence_is_never_cached(monkeypatch, capsys, tmp_path):
    subtitle = tmp_path / 'captions.vtt'
    subtitle.write_text('WEBVTT\n\n00:00.000 --> 00:10.000\nhello\n')
    monkeypatch.setattr(watch, 'fetch_captions', lambda *a, **kw: {
        'subtitle_path': str(subtitle), 'info': {'title': 't', 'duration': 30, 'chapters': [{'start_time': 0, 'title': 'Start'}]},
        'caption_track': {'language': 'en', 'kind': 'manual', 'provenance': 'original'}})
    def fail(*a, **kw):
        raise SystemExit('blocked download')
    monkeypatch.setattr(watch, 'download', fail)
    code, out = run(monkeypatch, capsys, 'https://example.com/v')
    assert code == 0 and '00:00] hello' in out and '- 00:00 Start' in out
    assert entries(cache.default_root()) == []


def test_caption_only_url_run_is_cached(monkeypatch, capsys, tmp_path):
    subtitle = tmp_path / 'captions.vtt'
    subtitle.write_text('WEBVTT\n\n00:00.000 --> 00:02.000\nhello there\n')
    calls = []
    def captions(*a, **kw):
        calls.append(1)
        return {'subtitle_path': str(subtitle), 'info': {'title': 't', 'duration': 30, 'description': 'About this.'},
                'caption_track': {'language': 'en', 'kind': 'manual', 'provenance': 'original'}}
    monkeypatch.setattr(watch, 'fetch_captions', captions)
    monkeypatch.setattr(watch, 'download', lambda *a, **kw: pytest.fail('caption-only run needs no media'))
    _, out = run(monkeypatch, capsys, 'https://example.com/v', '--detail', 'transcript')
    assert '## Description' in out and '[00:00] hello there' in out and 'Work dir' not in out
    _, out = run(monkeypatch, capsys, 'https://example.com/v', '--detail', 'transcript')
    assert calls == [1] and 'hello there' in out


def test_prune_keeps_most_recent_watch_entries(tmp_path):
    tmp_path = tmp_path / 'cache'
    for i in range(5):
        entry = tmp_path / f'k{i}'
        entry.mkdir(parents=True)
        (entry / 'timeline.json').write_text('{}')
        os.utime(entry, (i, i))
    (tmp_path / 'not-ours').mkdir()
    cache.prune(tmp_path, keep=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['k3', 'k4', 'not-ours']


def test_missing_local_source_has_no_key():
    assert cache.run_key('does-not-exist.mp4', {}) is None
    assert cache.run_key('https://e.com/a', {'d': 1}) != cache.run_key('https://e.com/a', {'d': 2})
