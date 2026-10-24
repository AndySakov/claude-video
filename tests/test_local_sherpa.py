"""sherpa-onnx backend: pinned downloads, a fake venv Python, speaker turns, and the runner's pure helpers."""
import hashlib
import io
import json
import os
import subprocess
import sys
import wave
from array import array
from pathlib import Path

import pytest
import config
import local_sherpa
import setup
import sherpa_runner
import timeline
import transcribe
import whisper


def tiny(name, data):
    return name, (len(data), hashlib.sha256(data).hexdigest())


@pytest.fixture
def tiny_models(monkeypatch):
    """Small stand-ins for the pinned model files, so readiness and downloads are testable offline."""
    blobs = {'encoder.int8.onnx': b'enc', 'tokens.txt': b'tok', 'silero_vad.onnx': b'vad',
             'model.int8.onnx': b'seg', 'nemo_en_titanet_small.onnx': b'emb'}
    model = dict(local_sherpa.MODELS['parakeet'], files=dict(tiny(n, blobs[n]) for n in ('encoder.int8.onnx', 'tokens.txt')))
    monkeypatch.setattr(local_sherpa, 'MODELS', {**local_sherpa.MODELS, 'parakeet': model})
    for attr, name in (('VAD', 'silero_vad.onnx'), ('SEGMENTATION', 'model.int8.onnx'), ('EMBEDDING', 'nemo_en_titanet_small.onnx')):
        repo, rev = getattr(local_sherpa, attr)[:2]
        monkeypatch.setattr(local_sherpa, attr, (repo, rev, name, len(blobs[name]), hashlib.sha256(blobs[name]).hexdigest()))
    return blobs


def place(blobs, cfg):
    for *_, name, size, sha, dest in local_sherpa.required_files(cfg):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blobs[name])


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture(scope='session')
def tone_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp('sherpa') / 'tone.mp4'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=blue:s=160x120:r=10',
                    '-f', 'lavfi', '-i', 'sine=f=440', '-t', '2', '-shortest', '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path


@pytest.fixture
def fake_venv(tmp_path, tiny_models):
    """A venv Python stand-in that writes --out JSON the way sherpa_runner.py does."""
    bin_dir = tmp_path / 'venv dir' / ('Scripts' if os.name == 'nt' else 'bin')
    bin_dir.mkdir(parents=True)
    script = bin_dir / 'fake.py'
    script.write_text('''import json, os, sys, time
args = sys.argv[1:]
assert args[0] == '-I' and args[1].endswith('sherpa_runner.py') and args[2].endswith('.wav'), args
mode = os.environ.get('WATCH_TEST_MODE', 'good')
out = args[args.index('--out') + 1]
if mode == 'crash':
    sys.stderr.write('onnxruntime: failed to load model\\n'); sys.exit(3)
if mode == 'hang':
    time.sleep(30)
if mode == 'malformed':
    open(out, 'w').write('{'); sys.exit(0)
if mode == 'args':
    json.dump({'segments': [{'start': 0.0, 'end': 1.0, 'text': ' '.join(args)}], 'speakers': 0}, open(out, 'w')); sys.exit(0)
sys.stderr.write('progress 1.0\\n')
segments = [] if mode == 'empty' else [
    {'start': 0.0, 'end': 1.2, 'text': 'How do you grow?', 'speaker': 'SPK1'},
    {'start': 1.4, 'end': 3.0, 'text': 'Post every day.', 'speaker': 'SPK2'},
    {'start': 3.1, 'end': 4.0, 'text': 'Really.', 'speaker': 'SPK2'}]
json.dump({'segments': segments, 'speakers': 2 if segments else 0}, open(out, 'w'))
''', encoding='utf-8')
    exe = bin_dir / ('python.cmd' if os.name == 'nt' else 'python')
    if os.name == 'nt':
        exe.write_text(f'@"{sys.executable}" "%~dp0fake.py" %*\r\n')
    else:
        exe.write_text(f'#!{sys.executable}\n' + script.read_text(encoding='utf-8'), encoding='utf-8')
        exe.chmod(0o755)
    local_sherpa.sentinel_for(exe).write_text(local_sherpa.SHERPA_VERSION, encoding='utf-8')
    config.write_settings({'WATCH_SHERPA_PYTHON': str(exe), 'WATCH_WHISPER_BACKEND': 'sherpa'})
    place(tiny_models, local_sherpa.settings())
    return exe


def test_settings_validate_model_speakers_and_threads(monkeypatch):
    cfg = local_sherpa.settings()
    assert cfg['sherpa_model'] == 'parakeet' and cfg['sherpa_speakers'] == 'auto' and cfg['sherpa_timeout'] is None
    for name, value in (('WATCH_SHERPA_MODEL', 'whisper'), ('WATCH_SHERPA_SPEAKERS', 'many'),
                        ('WATCH_SHERPA_SPEAKERS', '0'), ('WATCH_SHERPA_THREADS', '1.5')):
        monkeypatch.setenv(name, value)
        with pytest.raises(config.ConfigError):
            local_sherpa.settings()
        monkeypatch.delenv(name)
    monkeypatch.setenv('WATCH_SHERPA_SPEAKERS', '3')
    assert local_sherpa.settings()['sherpa_speakers'] == '3'


def test_every_pinned_file_has_a_revision_size_and_sha256():
    entries = [(m['repo'], m['rev'], n, s, h) for m in local_sherpa.MODELS.values() for n, (s, h) in m['files'].items()]
    entries += [local_sherpa.VAD, local_sherpa.SEGMENTATION, local_sherpa.EMBEDDING]
    for repo, rev, name, size, sha in entries:
        assert len(rev) == 40 and size > 1000 and len(sha) == 64, (repo, name)


def test_required_files_skip_diarization_models_when_speakers_are_off(monkeypatch):
    names = {dest.name for *_, dest in local_sherpa.required_files(local_sherpa.settings())}
    assert {'encoder.int8.onnx', 'silero_vad.onnx', 'nemo_en_titanet_small.onnx'} <= names
    monkeypatch.setenv('WATCH_SHERPA_SPEAKERS', 'off')
    names = {dest.name for *_, dest in local_sherpa.required_files(local_sherpa.settings())}
    assert 'silero_vad.onnx' in names and 'nemo_en_titanet_small.onnx' not in names


def test_download_verifies_size_and_sha256_atomically(tmp_path):
    data = b'x' * 2048
    sha = hashlib.sha256(data).hexdigest()
    urls = []
    def opener(url, timeout):
        urls.append(url)
        return Response(data)
    dest = tmp_path / 'm' / 'f.onnx'
    local_sherpa.download_file('o/r', 'a' * 40, 'f.onnx', len(data), sha, dest, opener=opener)
    assert dest.read_bytes() == data and urls == [f"https://huggingface.co/o/r/resolve/{'a' * 40}/f.onnx"]
    bad = tmp_path / 'm' / 'bad.onnx'
    with pytest.raises(SystemExit, match='pinned size and SHA-256'):
        local_sherpa.download_file('o/r', 'a' * 40, 'bad.onnx', len(data), '0' * 64, bad, opener=opener)
    assert not bad.exists() and sorted(p.name for p in bad.parent.iterdir()) == ['f.onnx']


def test_download_models_fetches_only_missing_files(tiny_models):
    cfg = local_sherpa.settings()
    calls = []
    def opener(url, timeout):
        calls.append(url.rsplit('/', 1)[1])
        return Response(tiny_models[calls[-1]])
    assert local_sherpa.download_models(cfg, opener=opener) == 5
    assert local_sherpa.download_models(cfg, opener=opener) == 0 and len(calls) == 5


def test_command_passes_model_vad_and_speaker_options(tmp_path, monkeypatch):
    python, wav, out = Path('/v/bin/python'), tmp_path / 'a.wav', tmp_path / 'o.json'
    cmd = local_sherpa.command(python, wav, out, local_sherpa.settings())
    assert cmd[:3] == [str(python), '-I', str(local_sherpa.RUNNER)] and cmd[cmd.index('--kind') + 1] == 'parakeet'
    assert '--segmentation' in cmd and '--embedding' in cmd and '--speakers' not in cmd
    monkeypatch.setenv('WATCH_SHERPA_SPEAKERS', '2')
    monkeypatch.setenv('WATCH_SHERPA_MODEL', 'moonshine-tiny')
    cmd = local_sherpa.command(python, wav, out, local_sherpa.settings())
    assert cmd[cmd.index('--speakers') + 1] == '2' and cmd[cmd.index('--kind') + 1] == 'moonshine'
    assert cmd[cmd.index('--max-piece') + 1] == '8'
    monkeypatch.setenv('WATCH_SHERPA_SPEAKERS', 'off')
    assert '--segmentation' not in local_sherpa.command(python, wav, out, local_sherpa.settings())


def test_transcribe_media_returns_speaker_segments(fake_venv, tone_clip, tmp_path):
    segments = local_sherpa.transcribe_media(str(tone_clip), tmp_path)
    assert [s['speaker'] for s in segments] == ['SPK1', 'SPK2', 'SPK2'] and segments.speakers == 2
    assert local_sherpa.describe_source(local_sherpa.settings(), segments).endswith('2 speakers labelled)')
    assert not list(tmp_path.glob('watch-sherpa-*'))


def test_whisper_dispatches_to_sherpa_without_a_key(fake_venv, tone_clip, tmp_path):
    segments, used = whisper.transcribe_video(str(tone_clip), tmp_path / 'audio.mp3', backend='sherpa')
    assert used == 'sherpa' and segments[0]['text'] == 'How do you grow?'
    with pytest.raises(SystemExit, match='does not accept an API key'):
        whisper.transcribe_video(str(tone_clip), tmp_path / 'audio.mp3', backend='sherpa', api_key='k')


@pytest.mark.parametrize('mode, message', [('crash', 'failed to load model'), ('malformed', 'malformed')])
def test_worker_failures_are_reported(fake_venv, tone_clip, tmp_path, monkeypatch, mode, message):
    monkeypatch.setenv('WATCH_TEST_MODE', mode)
    with pytest.raises(SystemExit, match=message):
        local_sherpa.transcribe_media(str(tone_clip), tmp_path)


def test_timeout_stops_the_worker(fake_venv, tone_clip, tmp_path, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'hang')
    monkeypatch.setenv('WATCH_SHERPA_TIMEOUT', '1')
    with pytest.raises(SystemExit, match='timed out'):
        local_sherpa.transcribe_media(str(tone_clip), tmp_path)


def test_missing_install_or_models_point_at_setup(tiny_models, tone_clip, tmp_path, fake_venv):
    next(local_sherpa.models_dir().rglob('silero_vad.onnx')).unlink()
    with pytest.raises(SystemExit, match='silero_vad.onnx.*setup.py --backend sherpa'):
        local_sherpa.transcribe_media(str(tone_clip), tmp_path)
    local_sherpa.sentinel_for(fake_venv).unlink()
    with pytest.raises(SystemExit, match='not installed'):
        local_sherpa.transcribe_media(str(tone_clip), tmp_path)


def test_setup_json_reports_sherpa_readiness(fake_venv):
    status = setup._status()
    assert status['sherpa_ready'] is True and status['backend_ready'] is True
    assert status['sherpa_model'] == 'parakeet' and status['sherpa_speakers'] == 'auto'


def test_install_sherpa_downloads_warms_up_and_saves(tiny_models, monkeypatch, tmp_path):
    steps = []
    def fake_step(cmd, step):
        steps.append(step)
        if 'warm-up' in step:
            Path(cmd[-1]).write_bytes(b'')
        elif 'environment' in step:
            python = local_sherpa.venv_python()
            python.parent.mkdir(parents=True)
            python.write_text('')
    monkeypatch.setattr(setup, '_install_step', fake_step)
    monkeypatch.setattr(setup, '_which', lambda name: None if name == 'uv' else f'/usr/bin/{name}')
    monkeypatch.setattr(local_sherpa, 'download_models', lambda cfg: place(tiny_models, cfg) or 5)
    checked = []
    monkeypatch.setattr(local_sherpa, 'transcribe_media', lambda *a, **k: checked.append(k))
    assert setup.install_sherpa() == 0
    assert steps == ['Create sherpa-onnx Python environment', 'Install sherpa-onnx', 'Create warm-up audio']
    assert checked == [{'require_ready': False, 'check': True}]
    values = config.read_env_file()
    assert values['WATCH_WHISPER_BACKEND'] == 'sherpa' and values['WATCH_SHERPA_PYTHON'] == str(local_sherpa.venv_python())
    assert local_sherpa.ready(local_sherpa.settings())


def test_normalize_segments_keeps_only_well_formed_speaker_labels():
    out = transcribe.normalize_segments({'segments': [
        {'start': 0, 'end': 1, 'text': 'a', 'speaker': 'SPK1'},
        {'start': 1, 'end': 2, 'text': 'b', 'speaker': 'Alice'},
        {'start': 2, 'end': 3, 'text': 'c'}]})
    assert [s.get('speaker') for s in out] == ['SPK1', None, None]


def test_timeline_splits_and_labels_speaker_turns():
    lines = timeline.merge_lines([
        {'start': 0.0, 'end': 1.0, 'text': 'How do you', 'speaker': 'SPK1'},
        {'start': 1.0, 'end': 1.5, 'text': 'grow', 'speaker': 'SPK1'},
        {'start': 1.6, 'end': 3.0, 'text': 'Post daily', 'speaker': 'SPK2'}])
    assert [(l['text'], l['speaker']) for l in lines] == [('How do you grow', 'SPK1'), ('Post daily', 'SPK2')]
    data = timeline.build(frames=[], lines=lines)
    assert timeline.events(data) == ['[00:00] SPK1: How do you grow', '[00:01] SPK2: Post daily']
    assert 'speaker turns' in timeline.render_sections(data, None)
    plain = timeline.build(frames=[], lines=timeline.merge_lines([{'start': 0, 'end': 1, 'text': 'hi'}]))
    assert 'speaker turns' not in timeline.render_sections(plain, None) and timeline.events(plain) == ['[00:00] hi']


def test_runner_reads_16k_mono_wav(tmp_path):
    path = tmp_path / 'a.wav'
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        pcm = array('h', [0, 16384, -32768])
        if sys.byteorder == 'big':
            pcm.byteswap()  # WAV is little-endian.
        w.writeframes(pcm.tobytes())
    assert list(sherpa_runner.read_wav(str(path))) == [0.0, 0.5, -1.0]
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b'\0' * 8)
    with pytest.raises(SystemExit, match='16 kHz mono'):
        sherpa_runner.read_wav(str(path))


def test_runner_splits_pieces_at_sentence_ends_with_token_times():
    tokens = ['▁Hi', '▁there', '.', '▁Book', '▁now', '.']
    stamps = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    out = sherpa_runner.split_sentences('Hi there. Book now.', tokens, stamps, 10.0, 12.5)
    assert out == [{'start': 10.0, 'end': 11.2, 'text': 'Hi there.'}, {'start': 11.2, 'end': 12.5, 'text': 'Book now.'}]
    # Without token timestamps (Moonshine) the piece stays whole; very short sentences are not cut off alone.
    assert sherpa_runner.split_sentences('Hi. Bye.', [], [], 0.0, 2.0) == [{'start': 0.0, 'end': 2.0, 'text': 'Hi. Bye.'}]
    short = sherpa_runner.split_sentences('No. Way.', ['No', '.', ' Way', '.'], [0.0, 0.2, 0.4, 0.6], 0.0, 0.9)
    assert [s['text'] for s in short] == ['No. Way.']


def test_runner_labels_speakers_by_overlap_in_order_of_appearance():
    segments = [{'start': 0.0, 'end': 2.0}, {'start': 2.0, 'end': 4.0}, {'start': 4.0, 'end': 5.0}, {'start': 9.0, 'end': 9.5}]
    turns = [(0.0, 1.5, 7), (1.5, 2.1, 3), (2.1, 4.2, 3), (4.2, 5.0, 7)]
    assert sherpa_runner.label(segments, turns) == 2
    assert [s.get('speaker') for s in segments] == ['SPK1', 'SPK2', 'SPK1', None]
