"""Live feeds: source kinds, ffmpeg inputs, window capture, and watch's live mode (synthetic sources only)."""
import json
import subprocess
import sys

import pytest

import cache
import live
import watch


@pytest.mark.parametrize('source, expected', [
    ('avfoundation:0', 'device'), ('v4l2:/dev/video0', 'device'), ('dshow:video=Integrated Camera', 'device'),
    ('rtsp://cam.local/stream', 'stream'), ('srt://10.0.0.2:9000', 'stream'), ('udp://239.0.0.1:1234', 'stream'),
    ('https://www.youtube.com/watch?v=abc', 'url'), ('clip.mp4', None), ('C:\\videos\\clip.mp4', None),
    ('avfoundation:', None), ('lavfi:testsrc', None)])
def test_kind(source, expected):
    assert live.kind(source) == expected


def test_window_bounds():
    assert live.window_seconds(None) == live.DEFAULT_WINDOW and live.window_seconds(90) == 90
    for bad in (0, 0.5, live.MAX_WINDOW + 1):
        with pytest.raises(SystemExit, match='--window'):
            live.window_seconds(bad)


def test_input_args_for_devices_streams_and_resolved_formats():
    assert live.input_args('avfoundation:1:0') == ['-f', 'avfoundation', '-framerate', '30', '-i', '1:0']
    assert live.input_args('v4l2:/dev/video0') == ['-f', 'v4l2', '-i', '/dev/video0']
    assert live.input_args('rtsp://cam/s') == ['-rtsp_transport', 'tcp', '-i', 'rtsp://cam/s']
    resolved = {'inputs': [('https://v/1.m3u8', {'User-Agent': 'UA', 'Bad': 'x\r\nInjected: 1'}, True),
                           ('https://a/2.mp4', {}, False)]}
    assert live.input_args('https://page', resolved) == [
        '-headers', 'User-Agent: UA\r\n', '-live_start_index', '-1', '-i', 'https://v/1.m3u8', '-i', 'https://a/2.mp4']


def fake_ytdlp(monkeypatch, info, code=0):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, code, json.dumps(info), 'ERROR: boom' if code else '')
    monkeypatch.setattr(live, 'run_text', run)
    return calls


def test_resolve_url_reads_formats_headers_and_live_flag(monkeypatch):
    calls = fake_ytdlp(monkeypatch, {'is_live': True, 'title': 'Launch', 'channel': 'Space', 'webpage_url': 'https://p',
                                     'requested_formats': [
                                         {'url': 'https://v.m3u8', 'protocol': 'm3u8_native', 'http_headers': {'A': 'b'}},
                                         {'url': 'https://a.m3u8', 'protocol': 'm3u8_native'}]})
    out = live.resolve_url('https://p', ['--cookies', '/c.txt'])
    assert out['live'] and out['info']['title'] == 'Launch' and out['info']['uploader'] == 'Space'
    assert out['inputs'] == [('https://v.m3u8', {'A': 'b'}, True), ('https://a.m3u8', {}, True)]
    assert calls[0][-3:] == ['/c.txt', '--', 'https://p'] and live.LIVE_FORMAT in calls[0]


def test_resolve_url_failures(monkeypatch):
    fake_ytdlp(monkeypatch, {}, code=1)
    with pytest.raises(SystemExit, match='yt-dlp failed'):
        live.resolve_url('https://p', [])
    fake_ytdlp(monkeypatch, {'is_live': True})
    with pytest.raises(SystemExit, match='no playable live format'):
        live.resolve_url('https://p', [])


@pytest.fixture
def lavfi_device(monkeypatch):
    """lavfi stands in for a camera: a device-style source ffmpeg can open anywhere."""
    monkeypatch.setattr(live, 'DEVICE_FORMATS', live.DEVICE_FORMATS | {'lavfi'})
    return 'lavfi:testsrc2=s=320x240:r=10'


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type',
                          '-of', 'json', str(path)], capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return float(data['format']['duration']), sorted(s['codec_type'] for s in data['streams'])


def test_capture_device_records_exactly_the_window(lavfi_device, tmp_path):
    rec = live.capture(lavfi_device, tmp_path / 'cap', 2.0)
    duration, kinds = probe(rec['video_path'])
    assert rec['video_path'].endswith('live.mp4') and kinds == ['video'] and 1.8 <= duration <= 2.3
    assert rec['started'].endswith(' UTC')


def test_capture_copies_a_resolved_stream(tmp_path, static_clip):
    resolved = {'inputs': [(str(static_clip), {}, False)]}
    rec = live.capture('https://example.com/live', tmp_path / 'cap', 1.0, resolved)
    duration, kinds = probe(rec['video_path'])
    # Stream copy cuts at a keyframe, so the file can run slightly past the window.
    assert rec['video_path'].endswith('live.mkv') and 'video' in kinds and duration < 3


def test_capture_failure_is_reported(tmp_path):
    with pytest.raises(SystemExit, match='Live capture failed'):
        live.capture('https://example.com/live', tmp_path / 'cap', 1.0, {'inputs': [(str(tmp_path / 'missing.ts'), {}, False)]})


def test_watch_records_a_live_window_and_never_caches_it(lavfi_device, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['watch', lavfi_device, '--window', '3', '--no-whisper'])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert '- **Live capture:** 3 s, recording started' in out and '] FRAME frame_' in out
    root = cache.resolve_root('')
    assert not root or not any(root.glob('*/timeline.md')), 'live windows must not be cached'


@pytest.mark.parametrize('argv, message', [
    (['clip.mp4', '--live'], '--live needs a stream URL'),
    (['rtsp://cam/s', '--start', '5'], '--start/--end do not apply'),
    (['clip.mp4', '--window', '30'], '--window applies to live feeds only')])
def test_live_argument_errors(monkeypatch, capsys, argv, message):
    monkeypatch.setattr(sys, 'argv', ['watch', *argv])
    with pytest.raises(SystemExit):
        watch.main()
    assert message in capsys.readouterr().err


def test_live_feeds_refuse_the_gemini_engine(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.setattr(sys, 'argv', ['watch', 'rtsp://cam/s', '--engine', 'gemini'])
    with pytest.raises(SystemExit, match='--engine local'):
        watch.main()


def test_auto_engine_with_a_gemini_key_still_records_live_feeds_locally(lavfi_device, monkeypatch, capsys):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.setattr(sys, 'argv', ['watch', lavfi_device, '--window', '2', '--no-whisper', '--detail', 'efficient'])
    assert watch.main() == 0
    assert '- **Live capture:**' in capsys.readouterr().out
