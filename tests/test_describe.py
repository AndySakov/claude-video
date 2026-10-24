"""Frame descriptions: a fake OpenAI-compatible server stands in for Ollama / llama-server / cloud."""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import config
import describe
import watch


@pytest.fixture
def server():
    state = {'models': ['qwen2.5vl:3b'], 'requests': [], 'status': 200, 'reply': 'A man at a desk talks into a microphone.'}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {'data': [{'id': m} for m in state['models']]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['requests'].append({'body': body, 'auth': self.headers.get('Authorization')})
            if state['status'] != 200:
                return self._send(state['status'], {'error': {'message': 'model crashed'}})
            self._send(200, {'choices': [{'message': {'content': state['reply']}}]})

    httpd = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state['url'] = f'http://127.0.0.1:{httpd.server_port}/v1'
    yield state
    httpd.shutdown()


def vision(state, **overrides):
    cfg = {'describe': 'auto', 'describe_url': state['url'], 'describe_model': '', 'describe_key': '',
           'describe_timeout': '', 'describe_max': '', **overrides}
    return describe.settings(cfg)


def test_auto_enables_only_when_the_model_is_listed(server):
    assert describe.resolve(vision(server)) == (True, None)
    server['models'] = ['llava:7b', 'llama3.2:3b']  # several models, none of them the configured one
    assert describe.resolve(vision(server)) == (False, None)
    enabled, problem = describe.resolve(vision(server, describe='on'))
    assert not enabled and 'qwen2.5vl:3b is not available' in problem


def test_auto_is_silent_when_nothing_listens_and_on_reports_it():
    closed = {'describe': 'auto', 'describe_url': 'http://127.0.0.1:9/v1', 'describe_model': '',
              'describe_key': '', 'describe_timeout': ''}
    assert describe.resolve(describe.settings(closed)) == (False, None)
    enabled, problem = describe.resolve(describe.settings({**closed, 'describe': 'on'}))
    assert not enabled and 'unreachable' in problem
    assert describe.resolve(describe.settings({**closed, 'describe': 'off'})) == (False, None)


def test_frame_is_sent_as_an_image_and_the_reply_is_cleaned(server, tmp_path):
    frame = tmp_path / 'frame_0001.jpg'
    frame.write_bytes(b'\xff\xd8jpeg')
    server['reply'] = ' "This image shows a white slide with a bar chart."\n'
    s = vision(server, describe_key='secret')
    assert describe.describe_frame(str(frame), s) == 'A white slide with a bar chart.'
    request = server['requests'][0]
    content = request['body']['messages'][0]['content']
    assert content[1]['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert request['auth'] == 'Bearer secret' and request['body']['model'] == 'qwen2.5vl:3b'


def test_endpoint_errors_are_reported_without_the_key(server, tmp_path):
    frame = tmp_path / 'frame.jpg'
    frame.write_bytes(b'x')
    server['status'] = 500
    with pytest.raises(SystemExit, match='HTTP 500') as caught:
        describe.describe_frame(str(frame), vision(server, describe_key='secret'))
    assert 'secret' not in str(caught.value)


@pytest.mark.parametrize('name,value', [('describe', 'maybe'), ('describe_url', 'localhost:11434'),
                                        ('describe_timeout', '0'), ('describe_max', '2.5')])
def test_invalid_settings(server, name, value):
    with pytest.raises(config.ConfigError):
        vision(server, **{name: value})


def test_watch_puts_descriptions_on_frame_lines_and_caches_them(server, monkeypatch, capsys, cut_clip):
    monkeypatch.setenv('WATCH_DESCRIBE', 'auto')
    monkeypatch.setenv('WATCH_DESCRIBE_URL', server['url'])
    monkeypatch.setattr(sys, 'argv', ['watch', str(cut_clip), '--no-whisper'])
    assert watch.main() == 0
    captured = capsys.readouterr()
    out = captured.out
    frames = [line for line in out.splitlines() if '] FRAME ' in line]
    assert frames and all(line.endswith('): A man at a desk talks into a microphone.') for line in frames)
    assert f'[watch] described {len(frames)}/{len(frames)} (' in captured.err
    timing = next(line for line in captured.err.splitlines() if line.startswith('[watch] time: '))
    assert re.fullmatch(r'\[watch\] time: (.+ · )?total \d+ s', timing), timing
    assert f'**Frame descriptions:** {len(frames)} via qwen2.5vl:3b' in out
    assert 'open only the frame files' in out and 'Read every `FRAME` file' not in out
    asked = len(server['requests'])
    assert watch.main() == 0
    assert len(server['requests']) == asked, 'a cache hit must not ask the model again'


def test_failed_descriptions_are_partial_evidence_not_cached(server, monkeypatch, capsys, static_clip):
    server['status'] = 503
    monkeypatch.setenv('WATCH_DESCRIBE', 'on')
    monkeypatch.setenv('WATCH_DESCRIBE_URL', server['url'])
    monkeypatch.setattr(sys, 'argv', ['watch', str(static_clip), '--no-whisper'])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert 'Frame descriptions unavailable' in out and 'partial evidence' in out and 'cached;' not in out


def test_a_server_you_point_at_with_one_model_serves_that_model(server):
    server['models'] = ['/models/SmolVLM-500M-Instruct-Q8_0.gguf']  # llama-server lists its file path
    s = vision(server)
    assert describe.resolve(s) == (True, None)
    assert s['model'] == server['models'][0] and describe.display_name(s['model']) == 'SmolVLM-500M-Instruct-Q8_0.gguf'


def test_default_ollama_address_requires_the_configured_model(server, monkeypatch):
    server['models'] = ['llama3.2:3b']  # a single non-vision model pulled
    s = vision(server)
    s['explicit_url'] = False  # as if WATCH_DESCRIBE_URL were unset
    assert describe.resolve(s) == (False, None)


def test_frame_cap_describes_evenly_spaced_frames_and_labels_the_rest(server, monkeypatch, capsys, cut_clip):
    assert [f['i'] for f in describe.pick([{'i': i} for i in range(10)], 4)] == [0, 3, 6, 9]
    monkeypatch.setenv('WATCH_DESCRIBE', 'auto')
    monkeypatch.setenv('WATCH_DESCRIBE_URL', server['url'])
    monkeypatch.setenv('WATCH_DESCRIBE_MAX', '3')
    monkeypatch.setattr(sys, 'argv', ['watch', str(cut_clip), '--no-whisper', '--no-cache'])
    assert watch.main() == 0
    out = capsys.readouterr().out
    frames = [line for line in out.splitlines() if '] FRAME ' in line]
    assert sum(line.endswith('microphone.') for line in frames) == 3 < len(frames)
    assert f'**Frame descriptions:** 3 of {len(frames)}' in out and 'were not described' in out
