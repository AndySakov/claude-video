"""whisper.cpp backend: a fake whisper-cli child, model downloads, and a real run when installed."""
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import config
import local_whispercpp
import setup
import watch
import whisper


@pytest.fixture(scope='session')
def tone_clip(tmp_path_factory):
    """Two seconds of video with a tone: enough audio for the WAV extraction step."""
    path = tmp_path_factory.mktemp('wcpp') / 'tone.mp4'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=blue:s=160x120:r=10',
                    '-f', 'lavfi', '-i', 'sine=f=440', '-t', '2', '-shortest', '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path


@pytest.fixture
def fake_cli(tmp_path, tone_clip):
    """A whisper-cli stand-in that writes -of <base>.json the way the real binary does."""
    bin_dir = tmp_path / 'bin with spaces'
    bin_dir.mkdir()
    script = bin_dir / 'fake.py'
    script.write_text('''import json, os, sys
args = sys.argv[1:]
mode = os.environ.get('WATCH_TEST_MODE', 'good')
assert os.path.isfile(args[args.index('-f') + 1]) and args[args.index('-f') + 1].endswith('.wav')
assert '-oj' in args and '-np' in args
if mode == 'crash':
    sys.stderr.write('failed to load model'); sys.exit(3)
base = args[args.index('-of') + 1]
if mode == 'malformed':
    open(base + '.json', 'w').write('{'); sys.exit(0)
rows = [] if mode == 'empty' else [
    {'offsets': {'from': 0, 'to': 1500}, 'text': ' [BLANK_AUDIO]'},
    {'offsets': {'from': 1500, 'to': 3200}, 'text': ' Hello there.'},
    {'offsets': {'from': 3200, 'to': 4000}, 'text': ' (music)'},
    {'offsets': {'from': 4000, 'to': 5250}, 'text': ' Book now.'}]
json.dump({'result': {'language': args[args.index('-l') + 1] if args[args.index('-l') + 1] != 'auto' else 'en'},
           'transcription': rows}, open(base + '.json', 'w'))
''', encoding='utf-8')
    exe = bin_dir / ('whisper-cli.cmd' if os.name == 'nt' else 'whisper-cli')
    if os.name == 'nt':
        exe.write_text(f'@"{sys.executable}" "%~dp0fake.py" %*\r\n')
    else:
        exe.write_text(f'#!{sys.executable}\n' + script.read_text())
        exe.chmod(0o755)
    model = tmp_path / 'ggml-test.bin'
    model.write_bytes(b'model')
    config.write_settings({'WATCH_WHISPERCPP_BIN': str(exe), 'WATCH_WHISPERCPP_MODEL': str(model),
                           'WATCH_WHISPER_BACKEND': 'whispercpp'})
    return exe, tone_clip


def test_segments_skip_non_speech_markers_and_keep_language(fake_cli, tmp_path):
    segments = local_whispercpp.transcribe_media(str(fake_cli[1]), tmp_path)
    assert segments == [{'start': 1.5, 'end': 3.2, 'text': 'Hello there.'}, {'start': 4.0, 'end': 5.25, 'text': 'Book now.'}]
    assert segments.language == 'en' and not list(tmp_path.glob('watch-whispercpp-*'))


def test_language_hint_is_passed_through(fake_cli, tmp_path, monkeypatch):
    monkeypatch.setenv('WATCH_WHISPERCPP_LANGUAGE', 'sw')
    assert local_whispercpp.transcribe_media(str(fake_cli[1]), tmp_path).language == 'sw'


def test_empty_is_no_speech(fake_cli, tmp_path, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'empty')
    segments = local_whispercpp.transcribe_media(str(fake_cli[1]), tmp_path)
    assert not segments and segments.no_speech


@pytest.mark.parametrize('mode,message', [('crash', 'failed to load model'), ('malformed', 'malformed')])
def test_failures_are_reported(fake_cli, tmp_path, monkeypatch, mode, message):
    monkeypatch.setenv('WATCH_TEST_MODE', mode)
    with pytest.raises(SystemExit, match=message):
        local_whispercpp.transcribe_media(str(fake_cli[1]), tmp_path)


def test_missing_binary_and_model_explain_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(local_whispercpp.shutil, 'which', lambda name: None)
    with pytest.raises(SystemExit, match='brew install whisper-cpp'):
        local_whispercpp.transcribe_media('video.mp4', tmp_path)
    monkeypatch.setenv('WATCH_WHISPERCPP_BIN', sys.executable)
    with pytest.raises(SystemExit, match='setup.py --backend whispercpp'):
        local_whispercpp.transcribe_media('video.mp4', tmp_path)


@pytest.mark.parametrize('name,value', [('WATCH_WHISPERCPP_MODEL', 'gigantic'), ('WATCH_WHISPERCPP_LANGUAGE', 'english'),
                                        ('WATCH_WHISPERCPP_THREADS', '2.5'), ('WATCH_WHISPERCPP_TIMEOUT', '-1')])
def test_invalid_settings(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(config.ConfigError):
        local_whispercpp.settings()


def test_watch_reports_whispercpp_transcript(fake_cli, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['watch', str(fake_cli[1]), '--detail', 'transcript'])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert 'via whisper (whisper.cpp test, language en (detected))' in out and '] Hello there. Book now.' in out


def test_dispatch_rejects_api_keys():
    with pytest.raises(SystemExit, match='does not accept an API key'):
        whisper.transcribe_video('video.mp4', Path('audio.mp3'), backend='whispercpp', api_key='k')


class FakeResponse(io.BytesIO):
    def __init__(self, body, length=None):
        super().__init__(body)
        self.headers = {'Content-Length': str(len(body) if length is None else length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_model_download_is_atomic_and_checked(tmp_path):
    body = b'x' * (2 << 20)
    seen = []
    def opener(url, timeout):
        seen.append(url)
        return FakeResponse(body)
    path = local_whispercpp.download_model('base', tmp_path / 'ggml-base.bin', opener=opener)
    assert path.read_bytes() == body and seen == ['https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin']
    with pytest.raises(SystemExit, match='incomplete'):
        local_whispercpp.download_model('tiny', tmp_path / 'ggml-tiny.bin', opener=lambda u, timeout: FakeResponse(body, len(body) + 1))
    assert not (tmp_path / 'ggml-tiny.bin').exists() and not list(tmp_path.glob('*.part'))
    def offline(url, timeout):
        raise OSError('network unreachable')
    with pytest.raises(SystemExit, match='huggingface.co'):
        local_whispercpp.download_model('tiny', tmp_path / 'ggml-tiny.bin', opener=offline)
    with pytest.raises(SystemExit, match='Unknown whisper.cpp model'):
        local_whispercpp.download_model('gigantic', tmp_path / 'x.bin', opener=opener)


def test_setup_without_binary_gives_platform_hint(monkeypatch, capsys):
    monkeypatch.setattr(local_whispercpp.shutil, 'which', lambda name: None)
    monkeypatch.setattr(setup, '_which', lambda name: None if name in ('brew', 'whisper-cli', 'whisper-cpp') else f'/bin/{name}')
    monkeypatch.setattr(setup, '_check_binaries', lambda: [])
    assert setup.cmd_install('whispercpp') == 2
    assert '--backend whispercpp' in capsys.readouterr().err


def test_setup_downloads_warms_and_saves(fake_cli, monkeypatch):
    config.write_settings({'WATCH_WHISPERCPP_MODEL': 'base'})
    monkeypatch.setattr(setup, '_check_binaries', lambda: [])
    fetched = []
    def fake_download(name, destination):
        fetched.append(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b'model')
        return destination
    monkeypatch.setattr(local_whispercpp, 'download_model', fake_download)
    assert setup.cmd_install('whispercpp') == 0
    saved = config.read_env_file()
    assert fetched == ['base'] and saved['WATCH_WHISPER_BACKEND'] == 'whispercpp' and saved['SETUP_COMPLETE'] == 'true'
    assert Path(saved['WATCH_WHISPERCPP_BIN']) == fake_cli[0]
    status = setup._status()
    assert status['whispercpp_ready'] and status['backend_ready']


def _real_cli():
    found = os.environ.get('WATCH_TEST_WHISPERCPP_BIN') or shutil.which('whisper-cli')
    model = Path(os.environ.get('WATCH_TEST_WHISPERCPP_MODEL', Path.home() / '.cache/watch/whisper.cpp/ggml-base.bin'))
    return (found, model) if found and model.is_file() else (None, None)


REAL_CLI, REAL_MODEL = _real_cli()


@pytest.mark.skipif(REAL_CLI is None, reason='real whisper-cli and model not available')
def test_real_whisper_cli_transcribes_synthesized_speech(tmp_path, monkeypatch):
    clip = tmp_path / 'speech.mp4'
    made = subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                           "flite=text='Book your weekend stay today.'", str(clip)], capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip('ffmpeg flite speech synthesis unavailable')
    monkeypatch.setenv('WATCH_WHISPERCPP_BIN', REAL_CLI)
    monkeypatch.setenv('WATCH_WHISPERCPP_MODEL', str(REAL_MODEL))
    segments = local_whispercpp.transcribe_media(str(clip), tmp_path)
    text = ' '.join(s['text'] for s in segments).lower()
    assert 'stay' in text and 'today' in text, text  # flite's robotic voice blurs the rest
