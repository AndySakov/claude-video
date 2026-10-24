"""sherpa-onnx adapter: NVIDIA Parakeet, Moonshine, or SenseVoice speech, plus speaker labels.

``sherpa_runner.py`` runs inside a small managed venv (the sherpa-onnx wheels, ~45 MB;
no torch or numpy) so watch itself stays standard-library only. Model files are pinned
to a Hugging Face revision and SHA-256, downloaded once into ~/.cache/watch/sherpa-onnx.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from config import ConfigError, get_config, positive_number
from local_whispercpp import extract_wav
from runtime import diagnostic
from transcribe import normalize_segments

SHERPA_VERSION = '2.0.0'
FILE_URL = 'https://huggingface.co/{repo}/resolve/{rev}/{name}'
RUNNER = Path(__file__).resolve().parent / 'sherpa_runner.py'
PARAKEET_FILES = ('encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt')
MOONSHINE_FILES = ('preprocess.onnx', 'encode.int8.onnx', 'uncached_decode.int8.onnx', 'cached_decode.int8.onnx', 'tokens.txt')

# name → runner kind, Hugging Face repo@revision, {file: (bytes, sha256)}, languages, VAD piece length.
# Moonshine gives no token timestamps, so its pieces are shorter to keep lines citable.
MODELS = {
    'parakeet': {
        'kind': 'parakeet', 'repo': 'csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8',
        'rev': '2bda32ec70b097a55adaa07d9a7173915b43cc78', 'languages': '25 European languages incl. English',
        'max_piece': 15, 'files': {
            'encoder.int8.onnx': (652184281, 'acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247'),
            'decoder.int8.onnx': (11845275, '179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e'),
            'joiner.int8.onnx': (6355277, '3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3'),
            'tokens.txt': (93939, 'd58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d')}},
    'parakeet-en': {
        'kind': 'parakeet', 'repo': 'csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8',
        'rev': '1ab9323565ddb038682214b292f588070a538ce2', 'languages': 'English', 'max_piece': 15, 'files': {
            'encoder.int8.onnx': (652184296, 'a32b12d17bbbc309d0686fbbcc2987b5e9b8333a7da83fa6b089f0a2acd651ab'),
            'decoder.int8.onnx': (7257753, 'b6bb64963457237b900e496ee9994b59294526439fbcc1fecf705b31a15c6b4e'),
            'joiner.int8.onnx': (1739080, '7946164367946e7f9f29a122407c3252b680dbae9a51343eb2488d057c3c43d2'),
            'tokens.txt': (9384, 'ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d')}},
    'moonshine-tiny': {
        'kind': 'moonshine', 'repo': 'csukuangfj/sherpa-onnx-moonshine-tiny-en-int8',
        'rev': 'bf2b762c076d8ea61e2af0b3851c9564fb77552e', 'languages': 'English', 'max_piece': 8, 'files': {
            'preprocess.onnx': (6800738, 'f33addce61a143460fe753b5ee5b7db255e5140b5b779c065b94f6c83ff0bf4e'),
            'encode.int8.onnx': (18249187, '8774dfba578de027ec6595c2c654a0836434489bc963a0db124a7f181f571acb'),
            'uncached_decode.int8.onnx': (53216096, '216737000dd5881a17aa043f6bbd286add33e4c3b0ae257153e2ec15438bdc41'),
            'cached_decode.int8.onnx': (45264830, '2aff28bba6a03d8dcf5c9feac45462629bae37317442299f28115ad09da773f6'),
            'tokens.txt': (436688, '1165c2aeb9f72f457a83be2d459a09054f27490acd9b41bd43794dfd25e296ea')}},
    'moonshine-base': {
        'kind': 'moonshine', 'repo': 'csukuangfj/sherpa-onnx-moonshine-base-en-int8',
        'rev': '052b0798ad1bf046a140fdd4efcd9426530fa3f5', 'languages': 'English', 'max_piece': 8, 'files': {
            'preprocess.onnx': (14077290, 'ffa630d395c5ccf76f5d4954be5b882df76aaf6491519ec01fd82ea7a3819fb2'),
            'encode.int8.onnx': (50311494, '7e38770f776f2e5583a53b052936005df2ba5c833d7e09c2a5fd796b94bf73e2'),
            'uncached_decode.int8.onnx': (122120451, 'c01f4b35093bcac20d352d23a75a539e772964579f9d024a90e5e6f09cae9987'),
            'cached_decode.int8.onnx': (99983837, '2db74e51cedf64a8b1be3c8192e0bb5e4923af0e90bd9e87f8e8771873f8ea03'),
            'tokens.txt': (436688, '1165c2aeb9f72f457a83be2d459a09054f27490acd9b41bd43794dfd25e296ea')}},
    'sensevoice': {
        'kind': 'sensevoice', 'repo': 'csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17',
        'rev': '2365baeacb507f821a0c8120fcee3d484dba7a07', 'languages': 'Chinese, Cantonese, English, Japanese, Korean',
        'max_piece': 15, 'files': {
            'model.int8.onnx': (239233841, 'c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51'),
            'tokens.txt': (315894, 'f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc')}},
}
# Shared helpers: Silero VAD always; pyannote segmentation + a speaker embedding for speaker labels.
VAD = ('csukuangfj/vad', 'fba88cd2e921609e7675c3aaf51e0b9b295da4bc', 'silero_vad.onnx',
       1807522, 'a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28')
SEGMENTATION = ('csukuangfj/sherpa-onnx-pyannote-segmentation-3-0', '9403a6902bb58e3d5ae8c7e77c3422de279db2e0',
                'model.int8.onnx', 1540506, 'd582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d')
# NVIDIA TitaNet-small: of five embeddings tried, the only one that kept a creator filmed in different
# rooms as one speaker while still splitting two-speaker recordings (clustering threshold 0.7).
EMBEDDING = ('csukuangfj/speaker-embedding-models', '0743f301363dec56491a490f6d6cbc9d67f9a3bf',
             'nemo_en_titanet_small.onnx', 40257283,
             'ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e')
MAX_SPEAKERS = 20
PROGRESS_SECONDS = 30.0


def models_dir() -> Path:
    return Path.home() / '.cache' / 'watch' / 'sherpa-onnx'


def venv_dir() -> Path:
    return Path.home() / '.cache' / 'watch' / 'sherpa-venv'


def venv_python(venv: Path | None = None) -> Path:
    venv = venv or venv_dir()
    return venv / ('Scripts/python.exe' if platform.system() == 'Windows' else 'bin/python')


def settings(cfg=None) -> dict:
    cfg = dict(get_config(backend_override='sherpa') if cfg is None else cfg)
    if cfg['sherpa_model'] not in MODELS:
        raise ConfigError(f"WATCH_SHERPA_MODEL must be one of {', '.join(MODELS)} (default parakeet).")
    speakers = cfg['sherpa_speakers'] or 'auto'
    if speakers not in ('auto', 'off') and not (speakers.isdigit() and 1 <= int(speakers) <= MAX_SPEAKERS):
        raise ConfigError(f'WATCH_SHERPA_SPEAKERS must be auto, off, or a speaker count from 1 to {MAX_SPEAKERS}.')
    cfg['sherpa_speakers'] = speakers
    threads = positive_number(cfg['sherpa_threads'] or str(min(8, os.cpu_count() or 4)), 'WATCH_SHERPA_THREADS')
    if threads != int(threads):
        raise ConfigError('WATCH_SHERPA_THREADS must be a positive integer.')
    cfg['sherpa_threads'] = int(threads)
    cfg['sherpa_timeout'] = positive_number(cfg['sherpa_timeout'], 'WATCH_SHERPA_TIMEOUT') if cfg['sherpa_timeout'] else None
    return cfg


def python_path(cfg: dict) -> Path:
    return Path(cfg['sherpa_python']).expanduser() if cfg['sherpa_python'] else venv_python()


def sentinel_for(python: Path) -> Path:
    return python.parent.parent / '.deps-ok'


def required_files(cfg: dict) -> list[tuple[str, str, str, int, str, Path]]:
    """(repo, rev, name, size, sha256, destination) for everything this configuration runs."""
    model = MODELS[cfg['sherpa_model']]
    root = models_dir()
    files = [(model['repo'], model['rev'], name, size, sha, root / cfg['sherpa_model'] / name)
             for name, (size, sha) in model['files'].items()]
    helpers = [VAD] + ([SEGMENTATION, EMBEDDING] if cfg['sherpa_speakers'] != 'off' else [])
    files += [(repo, rev, name, size, sha, root / helper_name(repo, name)) for repo, rev, name, size, sha in helpers]
    return files


def helper_name(repo: str, name: str) -> str:
    return 'pyannote-segmentation-3.0.int8.onnx' if 'pyannote' in repo else name


def ready(cfg: dict) -> bool:
    python = python_path(cfg)
    return (python.is_file() and sentinel_for(python).is_file()
            and all(dest.is_file() and dest.stat().st_size == size for *_, size, _, dest in required_files(cfg)))


def download_file(repo: str, rev: str, name: str, size: int, sha256: str, destination: Path, *, opener=urlopen) -> Path:
    """Stream one pinned file into place atomically; size and SHA-256 must match."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, partial = tempfile.mkstemp(prefix=destination.name + '.', suffix='.part', dir=destination.parent)
    digest, done, reported = hashlib.sha256(), 0, -1
    try:
        with os.fdopen(handle, 'wb') as out, opener(FILE_URL.format(repo=repo, rev=rev, name=name), timeout=60) as response:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                step = done * 10 // size
                if size > 50 << 20 and step != reported:
                    reported = step
                    print(f'[setup] {name}: {done >> 20} / {size >> 20} MB', file=sys.stderr, flush=True)
        if done != size or digest.hexdigest() != sha256:
            raise SystemExit(f'{name} download did not match its pinned size and SHA-256 ({done} of {size} bytes); rerun setup.')
        os.replace(partial, destination)
    except (URLError, OSError, TimeoutError) as exc:
        raise SystemExit(f'Could not download {name} ({type(exc).__name__}: {getattr(exc, "reason", exc)}). '
                         'Check network access to huggingface.co and rerun setup.') from None
    finally:
        Path(partial).unlink(missing_ok=True)
    return destination


def download_models(cfg: dict, *, opener=urlopen) -> int:
    """Fetch every missing or wrong-sized file; returns how many were downloaded."""
    count = 0
    for repo, rev, name, size, sha, dest in required_files(cfg):
        if not (dest.is_file() and dest.stat().st_size == size):
            download_file(repo, rev, name, size, sha, dest, opener=opener)
            count += 1
    return count


def command(python: Path, wav: Path, out: Path, cfg: dict) -> list[str]:
    model = MODELS[cfg['sherpa_model']]
    root = models_dir()
    cmd = [str(python), '-I', str(RUNNER), str(wav), '--out', str(out), '--kind', model['kind'],
           '--model', str(root / cfg['sherpa_model']), '--vad', str(root / VAD[2]),
           '--threads', str(cfg['sherpa_threads']), '--max-piece', str(model['max_piece'])]
    if cfg['sherpa_speakers'] != 'off':
        cmd += ['--segmentation', str(root / helper_name(SEGMENTATION[0], SEGMENTATION[2])),
                '--embedding', str(root / EMBEDDING[2])]
        if cfg['sherpa_speakers'] != 'auto':
            cmd += ['--speakers', cfg['sherpa_speakers']]
    return cmd


def run(cmd: list[str], timeout: float | None) -> None:
    """Run the worker, relaying progress about every 30 s and keeping a diagnostic tail."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONWARNINGS='ignore')
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    except OSError as exc:
        raise SystemExit(f'sherpa-onnx Python is not runnable ({type(exc).__name__}); rerun setup.py --backend sherpa.') from None
    expired = threading.Event()
    def stop():
        expired.set()
        process.kill()
    timer = threading.Timer(timeout, stop) if timeout else None
    if timer:
        timer.start()
    tail, last = [], time.monotonic()
    try:
        for raw in process.stderr:
            line = raw.decode('utf-8', errors='replace').strip()
            if line.startswith('progress '):
                if time.monotonic() - last >= PROGRESS_SECONDS:
                    last = time.monotonic()
                    print(f'[watch] sherpa-onnx: {float(line.split()[1]):.0f} s of audio transcribed', file=sys.stderr, flush=True)
            elif line:
                tail = (tail + [line])[-40:]
        process.wait()
    except KeyboardInterrupt:
        process.kill()
        process.wait()
        raise SystemExit('sherpa-onnx cancelled and stopped; no cloud fallback was attempted.') from None
    finally:
        if timer:
            timer.cancel()
        process.stderr.close()
    if expired.is_set():
        raise SystemExit('sherpa-onnx timed out and was stopped. Increase WATCH_SHERPA_TIMEOUT or leave it unset.')
    if process.returncode != 0:
        raise SystemExit(f'sherpa-onnx failed (exit {process.returncode}): {diagnostic(chr(10).join(tail), 1500)}')


def parse(data: dict):
    segments = normalize_segments(data)
    segments.speakers = data.get('speakers') or 0
    return segments


def transcribe_media(video_path: str, work: Path, cfg=None, *, require_ready=True, check=False):
    cfg = settings(cfg)
    python = python_path(cfg)
    if require_ready and not (python.is_file() and sentinel_for(python).is_file()):
        raise SystemExit('sherpa-onnx is not installed. Run setup.py --backend sherpa.')
    missing = [dest.name for *_, size, _, dest in required_files(cfg) if not (dest.is_file() and dest.stat().st_size == size)]
    if missing:
        raise SystemExit(f"sherpa-onnx model files missing ({', '.join(missing)}). Run setup.py --backend sherpa.")
    labels = '' if cfg['sherpa_speakers'] == 'off' else ' with speaker labels'
    print(f"[watch] local sherpa-onnx {cfg['sherpa_model']} transcription{labels}…", file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix='watch-sherpa-', dir=work) as temporary:
        wav = extract_wav(video_path, Path(temporary) / 'audio.wav')
        out = Path(temporary) / 'transcript.json'
        run(command(python, wav, out, cfg) + (['--check'] if check else []), cfg['sherpa_timeout'])
        try:
            segments = parse(json.loads(out.read_text(encoding='utf-8')))
        except (OSError, ValueError, TypeError) as exc:
            raise SystemExit(f'sherpa-onnx output missing or malformed ({type(exc).__name__}).') from None
    if segments.no_speech and not check:
        print('[watch] sherpa-onnx detected no speech.', file=sys.stderr)
    return segments


def describe_source(cfg: dict, segments) -> str:
    """The transcript provenance line: model, languages, and how many speakers were told apart."""
    model = MODELS[cfg['sherpa_model']]
    speakers = getattr(segments, 'speakers', 0)
    who = f', {speakers} speakers labelled' if speakers >= 2 else ''
    return f"local speech (sherpa-onnx {cfg['sherpa_model']}, {model['languages']}{who})"
