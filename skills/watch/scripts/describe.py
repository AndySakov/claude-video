#!/usr/bin/env python3
"""One-sentence frame descriptions from any OpenAI-compatible vision endpoint.

The same chat-completions request works against local Ollama (default,
http://localhost:11434/v1), llama.cpp's llama-server, LM Studio, and cheap cloud
endpoints such as Groq or OpenRouter. Descriptions turn FRAME lines into text, so
the agent can read the timeline and open only the frames whose detail matters.
"""
from __future__ import annotations

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from config import ConfigError, positive_number
from runtime import diagnostic

CHOICES = ('auto', 'on', 'off')
DEFAULT_URL = 'http://localhost:11434/v1'  # Ollama's OpenAI-compatible API
DEFAULT_MODEL = 'qwen2.5vl:3b'  # Sub-1B models (SmolVLM-500M) hallucinated people and text in testing.
# A 3B model takes ~25 s per frame on a 4-core CPU (far less with Metal or a GPU), so a run describes at
# most this many frames, evenly spaced; the rest stay plain FRAME lines.
DEFAULT_MAX = 40
MAX_CHARS = 220
PROMPT = ('Describe this video frame in one sentence of at most 25 words for someone who cannot see it: '
          'the setting, who is visible and what they are doing, and notable objects or graphics. '
          'Do not transcribe on-screen text; it is captured separately. Reply with the sentence only.')


def settings(cfg: dict) -> dict:
    choice = cfg['describe']
    if choice not in CHOICES:
        raise ConfigError('WATCH_DESCRIBE must be auto, on, or off.')
    url = (cfg['describe_url'] or DEFAULT_URL).rstrip('/')
    if urlparse(url).scheme not in ('http', 'https'):
        raise ConfigError('WATCH_DESCRIBE_URL must be an http(s) URL ending at the API root, e.g. http://localhost:11434/v1.')
    timeout = positive_number(cfg['describe_timeout'] or '120', 'WATCH_DESCRIBE_TIMEOUT')
    limit = positive_number(cfg.get('describe_max') or str(DEFAULT_MAX), 'WATCH_DESCRIBE_MAX')
    if limit != int(limit):
        raise ConfigError('WATCH_DESCRIBE_MAX must be a positive integer.')
    return {'choice': choice, 'url': url, 'model': cfg['describe_model'] or DEFAULT_MODEL,
            'explicit_url': bool(cfg['describe_url']), 'key': cfg['describe_key'] or None, 'timeout': timeout,
            'max': int(limit)}


def display_name(model: str) -> str:
    """Servers that list a model file path get reported by file name."""
    return Path(model).name if '/' in model or '\\' in model else model


def is_local(url: str) -> bool:
    return (urlparse(url).hostname or '') in ('localhost', '127.0.0.1', '::1')


def _request(method: str, url: str, s: dict, payload=None, timeout: float = 10.0) -> dict:
    headers = {'Content-Type': 'application/json'}
    if s['key']:
        headers['Authorization'] = f"Bearer {s['key']}"
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    try:
        with urlopen(Request(url, data=data, method=method, headers=headers), timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        detail = diagnostic(exc.read(), 300)
        if s['key']:
            detail = detail.replace(s['key'], '[redacted]')
        raise SystemExit(f'vision endpoint {urlparse(url).netloc} returned HTTP {exc.code}: {detail}') from None
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise SystemExit(f'vision endpoint {urlparse(url).netloc} unreachable ({type(exc).__name__}: '
                         f'{getattr(exc, "reason", exc)})') from None


def resolve(s: dict) -> tuple[bool, str | None]:
    """(enabled, problem). ``auto`` turns on only when the endpoint answers and lists the model."""
    if s['choice'] == 'off':
        return False, None
    try:
        listed = {m.get('id') for m in _request('GET', f"{s['url']}/models", s, timeout=3.0).get('data', [])
                  if isinstance(m, dict)}
    except (SystemExit, AttributeError) as exc:
        return False, (str(exc) if s['choice'] == 'on' else None)
    listed.discard(None)
    if len(listed) == 1 and s['explicit_url'] and s['model'] not in listed:
        # A server you pointed at with one model loaded (llama-server, LM Studio) serves that model.
        s['model'] = next(iter(listed))
        return True, None
    if listed and s['model'] not in listed and not any(str(i).startswith(s['model'] + ':') for i in listed):
        hint = f"model {s['model']} is not available at {s['url']}"
        if is_local(s['url']) and ':11434' in s['url']:
            hint += f" (ollama pull {s['model']})"
        return False, (hint if s['choice'] == 'on' else None)
    return True, None


def clean(text: str) -> str:
    text = ' '.join(str(text).split()).strip().strip('"')
    text = re.sub(r'^(this|the) (image|frame|video frame|screenshot) (shows|depicts)\s+', '', text, flags=re.I)
    return (text[:1].upper() + text[1:])[:MAX_CHARS]


def describe_frame(path: str, s: dict) -> str:
    image = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    payload = {'model': s['model'], 'temperature': 0, 'max_tokens': 80,
               'messages': [{'role': 'user', 'content': [
                   {'type': 'text', 'text': PROMPT},
                   {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}}]}]}
    data = _request('POST', f"{s['url']}/chat/completions", s, payload, timeout=s['timeout'])
    try:
        return clean(data['choices'][0]['message']['content'])
    except (KeyError, IndexError, TypeError):
        raise SystemExit('vision endpoint returned no description') from None


def pick(frames: list[dict], limit: int) -> list[dict]:
    """At most ``limit`` frames, evenly spaced and keeping the first and last."""
    if len(frames) <= limit:
        return list(frames)
    if limit == 1:
        return [frames[0]]
    return [frames[round(i * (len(frames) - 1) / (limit - 1))] for i in range(limit)]


def describe_frames(frames: list[dict], s: dict) -> dict:
    """Map frame path → description. Local models run one at a time; cloud endpoints in parallel."""
    frames = pick(frames, s['max'])
    workers = 1 if is_local(s['url']) else 4
    with ThreadPoolExecutor(max_workers=workers) as pool:
        texts = list(pool.map(lambda f: describe_frame(f['path'], s), frames))
    return {f['path']: t for f, t in zip(frames, texts) if t}
