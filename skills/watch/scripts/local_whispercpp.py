"""whisper.cpp adapter: one native binary plus a ggml model file, no Python dependencies.

The binary comes from the platform package manager (Homebrew `whisper-cpp` installs
`whisper-cli`, Metal-accelerated on Apple Silicon) or a source build; models are
downloaded once into ~/.cache/watch/whisper.cpp by setup.py.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from config import ConfigError, get_config, positive_number
from runtime import diagnostic, run_text
from transcribe import normalize_segments

# Names published at huggingface.co/ggerganov/whisper.cpp (ggml-<name>.bin).
MODELS = {'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en',
          'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo', 'large-v3-turbo-q5_0', 'large-v3-turbo-q8_0',
          'small-q5_1', 'small.en-q5_1', 'medium-q5_0', 'large-v3-q5_0'}
MODEL_URL = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin'
# Homebrew and current source builds install whisper-cli; older formulae named it whisper-cpp.
EXECUTABLES = ('whisper-cli', 'whisper-cpp')
# Non-speech markers whisper.cpp emits for silence, music, and noise.
MARKER_RE = re.compile(r'^\s*[\[(][^\])]*[\])]\s*$')


def models_dir() -> Path:
    return Path.home() / '.cache' / 'watch' / 'whisper.cpp'


def settings(cfg=None) -> dict:
    cfg = dict(get_config(backend_override='whispercpp') if cfg is None else cfg)
    model = cfg['whispercpp_model']
    if model not in MODELS and not (model.endswith('.bin') and Path(model).expanduser().is_absolute()):
        raise ConfigError('WATCH_WHISPERCPP_MODEL must be a whisper.cpp model name (default small) '
                          'or an absolute path to a ggml .bin file.')
    if cfg['whispercpp_language'] != 'auto' and not re.fullmatch('[a-z]{2}', cfg['whispercpp_language']):
        raise ConfigError('WATCH_WHISPERCPP_LANGUAGE must be auto or a two-letter lowercase code.')
    threads = positive_number(cfg['whispercpp_threads'] or str(min(8, os.cpu_count() or 4)), 'WATCH_WHISPERCPP_THREADS')
    if threads != int(threads):
        raise ConfigError('WATCH_WHISPERCPP_THREADS must be a positive integer.')
    cfg['whispercpp_threads'] = int(threads)
    cfg['whispercpp_timeout'] = (positive_number(cfg['whispercpp_timeout'], 'WATCH_WHISPERCPP_TIMEOUT')
                                 if cfg['whispercpp_timeout'] else None)
    return cfg


def find_executable(cfg: dict) -> Path | None:
    value = cfg['whispercpp_bin']
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() and path.is_file() else None
    found = next((shutil.which(name) for name in EXECUTABLES if shutil.which(name)), None)
    return Path(found) if found else None


def model_path(cfg: dict) -> Path:
    model = cfg['whispercpp_model']
    return Path(model).expanduser() if model.endswith('.bin') else models_dir() / f'ggml-{model}.bin'


def ready(cfg: dict) -> bool:
    return bool(find_executable(cfg)) and model_path(cfg).is_file()


def download_model(name: str, destination: Path, *, opener=urlopen) -> Path:
    """Stream a published model into place atomically, reporting progress on stderr."""
    if name not in MODELS:
        raise SystemExit(f'Unknown whisper.cpp model {name!r}; choose one of: {", ".join(sorted(MODELS))}.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, partial = tempfile.mkstemp(prefix=destination.name + '.', suffix='.part', dir=destination.parent)
    try:
        with os.fdopen(handle, 'wb') as out, opener(MODEL_URL.format(name=name), timeout=60) as response:
            total = int(response.headers.get('Content-Length') or 0)
            done, reported = 0, -1
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                step = done * 10 // total if total else -1
                if step != reported and total:
                    reported = step
                    print(f'[setup] whisper.cpp {name}: {done >> 20} / {total >> 20} MB', file=sys.stderr, flush=True)
        if total and done != total or done < 1 << 20:
            raise SystemExit(f'whisper.cpp model download was incomplete ({done} of {total or "?"} bytes); rerun setup.')
        os.replace(partial, destination)
    except (URLError, OSError, TimeoutError) as exc:
        raise SystemExit(f'Could not download the whisper.cpp {name} model ({type(exc).__name__}: '
                         f'{getattr(exc, "reason", exc)}). Check network access to huggingface.co and rerun setup.') from None
    finally:
        Path(partial).unlink(missing_ok=True)
    return destination


def extract_wav(video_path: str, out_path: Path) -> Path:
    """whisper.cpp reads 16 kHz mono PCM WAV everywhere; other formats depend on the build."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_text(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(Path(video_path).resolve()),
                       '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(out_path.resolve())])
    if result.returncode != 0 or not out_path.is_file() or out_path.stat().st_size <= 44:
        raise SystemExit(f'ffmpeg audio extraction failed: {diagnostic(result.stderr) or "no audio produced"}')
    return out_path


def command(executable: Path, model: Path, wav: Path, output_base: Path, cfg: dict) -> list[str]:
    return [str(executable), '-m', str(model), '-f', str(wav), '-oj', '-of', str(output_base),
            '-l', cfg['whispercpp_language'], '-t', str(cfg['whispercpp_threads']), '-np']


def parse(data: dict):
    """whisper-cli JSON (offsets in milliseconds) → validated segments with detected language."""
    try:
        rows = [{'start': item['offsets']['from'] / 1000, 'end': item['offsets']['to'] / 1000,
                 'text': item['text'].strip()} for item in data['transcription']]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f'unexpected whisper.cpp JSON ({type(exc).__name__})') from None
    segments = normalize_segments({'segments': [r for r in rows if r['text'] and not MARKER_RE.match(r['text'])]})
    segments.language = (data.get('result') or {}).get('language')
    return segments


def transcribe_media(video_path: str, work: Path, cfg=None):
    cfg = settings(cfg)
    executable = find_executable(cfg)
    if not executable:
        raise SystemExit('whisper.cpp is not installed (whisper-cli). macOS: brew install whisper-cpp; '
                         'then run setup.py --backend whispercpp.')
    model = model_path(cfg)
    if not model.is_file():
        raise SystemExit(f'whisper.cpp model missing at {model}. Run setup.py --backend whispercpp to download it.')
    print(f"[watch] local whisper.cpp {model.stem.removeprefix('ggml-')} transcription…", file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix='watch-whispercpp-', dir=work) as temporary:
        wav = extract_wav(video_path, Path(temporary) / 'audio.wav')
        output = Path(temporary) / 'transcript'
        result = run_text(command(executable, model, wav, output, cfg), timeout=cfg['whispercpp_timeout'])
        if result.returncode != 0:
            raise SystemExit(f'whisper.cpp failed (exit {result.returncode}): {diagnostic(result.stderr, 1500)}')
        try:
            segments = parse(json.loads(output.with_suffix('.json').read_text(encoding='utf-8')))
        except (OSError, ValueError) as exc:
            raise SystemExit(f'whisper.cpp output missing or malformed ({exc}).') from None
    if segments.no_speech:
        print('[watch] whisper.cpp detected no speech.', file=sys.stderr)
    return segments
