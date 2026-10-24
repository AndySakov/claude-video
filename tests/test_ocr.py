"""On-screen text: TSV line grouping, event tracking, engine choice, and real OCR runs."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import ocr
import watch

HEADER = 'level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext'


def word(text, left, top, width=80, height=40, conf=90, block=1):
    return f'5\t1\t{block}\t1\t1\t1\t{left}\t{top}\t{width}\t{height}\t{conf}\t{text}'


def test_words_join_lines_by_position_not_block_ids():
    # Sparse mode files each word as its own block; descenders shift centres slightly.
    tsv = '\n'.join([HEADER, word('18', 400, 502, block=3), word('percent', 500, 506, height=46, block=4),
                     word('Bookings', 100, 504, 180, 46, block=1), word('up', 300, 506, height=46, block=2),
                     word('Q3', 100, 300, block=5), word('Revenue', 200, 300, 200, block=5)])
    assert [l['text'] for l in ocr.parse_tsv(tsv)] == ['Q3 Revenue', 'Bookings up 18 percent']


def test_far_apart_words_and_low_confidence_lines_are_separate_or_dropped():
    tsv = '\n'.join([HEADER, word('left', 0, 0), word('right', 2000, 0), word('zzq', 0, 500, conf=40)])
    assert sorted(l['text'] for l in ocr.parse_tsv(tsv)) == ['left', 'right']


def page(text, number, left=100, top=100):
    return f'5\t{number}\t1\t1\t1\t1\t{left}\t{top}\t200\t40\t90\t{text}'


def test_one_tesseract_process_reads_a_list_and_splits_pages(monkeypatch, tmp_path):
    calls = []
    def fake(cmd, **kwargs):
        calls.append(cmd)
        listed = Path(cmd[1]).read_text(encoding='utf-8').splitlines()
        assert listed == ['/a b/raw_1.jpg', '/a b/raw_2.jpg', '/a b/raw_3.jpg']
        # Page 2 has no words; blank separator rows between pages are ignored.
        out = '\n'.join([HEADER, '1\t1\t0\t0\t0\t0\t0\t0\t720\t1280\t-1\t', page('Hello', 1), page('there', 1, 330), '',
                         '1\t2\t0\t0\t0\t0\t0\t0\t720\t1280\t-1\t', page('Book', 3), page('now', 3, 330)])
        return subprocess.CompletedProcess(cmd, 0, out, '')
    monkeypatch.setattr(ocr, 'run_text', fake)
    pages = ocr._tesseract(['/a b/raw_1.jpg', '/a b/raw_2.jpg', '/a b/raw_3.jpg'], 'eng', '3', tmp_path / 'list.txt')
    assert [[l['text'] for l in p] for p in pages] == [['Hello there'], [], ['Book now']]
    assert len(calls) == 1 and calls[0][2:] == ['stdout', '-l', 'eng', '--psm', '3', 'tsv']


def test_read_tesseract_keeps_sample_order_across_batches(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr.os, 'cpu_count', lambda: 2)
    samples = [{'path': str(tmp_path / f'raw_{i}.jpg'), 'mask': str(tmp_path / f'mask_{i}.png') if i % 2 else None}
               for i in range(13)]
    def fake(paths, lang, psm, listing):
        return [[{'text': f"{psm}:{Path(p).stem}", 'conf': 0.9, 'y': 1 if psm == '3' else 2}] for p in paths]
    monkeypatch.setattr(ocr, '_tesseract', fake)
    out = ocr.read_tesseract(samples, 'eng')
    assert out == [[f'3:raw_{i}'] + ([f'11:mask_{i}'] if i % 2 else []) for i in range(13)]


def test_vision_helper_is_compiled_once_and_cached(monkeypatch):
    compiles = []
    def fake_run(cmd, **kwargs):
        compiles.append(cmd)
        Path(cmd[cmd.index('-o') + 1]).write_text('binary')
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(ocr.shutil, 'which', lambda name: '/usr/bin/swiftc' if name == 'swiftc' else None)
    monkeypatch.setattr(ocr, 'run_text', fake_run)
    first = ocr.vision_command()
    assert first == [str(ocr.vision_binary())] and Path(first[0]).read_text() == 'binary'
    assert ocr.vision_command() == first and len(compiles) == 1, 'a cached helper must not be rebuilt'
    assert compiles[0][:3] == ['/usr/bin/swiftc', '-O', str(ocr.VISION_SCRIPT)]
    assert not list(Path(first[0]).parent.glob('*.part'))


def test_vision_falls_back_to_the_interpreter_when_compiling_fails(monkeypatch, capsys):
    monkeypatch.setattr(ocr.shutil, 'which', lambda name: '/usr/bin/swiftc' if name == 'swiftc' else None)
    monkeypatch.setattr(ocr, 'run_text', lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, '', 'error: no SDK'))
    assert ocr.vision_command() == ['swift', str(ocr.VISION_SCRIPT)]
    assert 'no SDK' in capsys.readouterr().err and not ocr.vision_binary().exists()
    monkeypatch.setattr(ocr.shutil, 'which', lambda name: None)
    assert ocr.vision_command() == ['swift', str(ocr.VISION_SCRIPT)]


def test_read_vision_splits_samples_across_helpers_and_keeps_order(monkeypatch, tmp_path):
    helper = tmp_path / 'helper.py'
    helper.write_text("""import json, os, sys
for path in sys.stdin.read().split(chr(10)):
    if path:
        name = os.path.basename(path).split('.')[0]
        print(json.dumps({'path': path, 'lines': [{'text': 'Frame ' + name, 'conf': 0.9, 'y': 0.5}]}))
""", encoding='utf-8')
    monkeypatch.setattr(ocr, 'vision_command', lambda: [sys.executable, str(helper)])
    monkeypatch.setattr(ocr, 'VISION_WORKERS', 3)
    calls = []
    real = ocr.run_text
    monkeypatch.setattr(ocr, 'run_text', lambda cmd, **kw: calls.append((kw['input'].count(chr(10)), kw['timeout'])) or real(cmd, **kw))
    samples = [{'path': str(tmp_path / f'raw_{i:04d}.jpg')} for i in range(10)]
    assert ocr.read_vision(samples) == [[f'Frame raw_{i:04d}'] for i in range(10)]
    # One sample alone first, with the long warm-up limit; then the rest side by side.
    assert calls[0] == (1, ocr.VISION_WARMUP_SECONDS) and sorted(n for n, _ in calls[1:]) == [3, 3, 3]
    calls.clear()
    assert ocr.read_vision(samples[:1]) == [['Frame raw_0000']] and calls == [(1, ocr.VISION_WARMUP_SECONDS)]


def test_clean_strips_symbol_tokens_and_stray_end_characters():
    assert ocr.clean('4 Never open a call 4 ©') == 'Never open a call'
    assert ocr.clean("you've got © their full |") == "you've got their full"
    assert ocr.clean('I said a') == 'I said a'


def test_adaptive_rate_is_dense_for_short_clips_and_capped():
    assert ocr.sample_rate(55) == 3.0 and ocr.sample_rate(300) == 2.0 and ocr.sample_rate(400) == 1.5 and ocr.sample_rate(1200) == 0.5
    assert ocr.sample_rate(55, 1.0) == 1.0


@pytest.mark.parametrize('text,ok', [('Book now', True), ('a.', False), ('=|]', False), ('Q3', False), ('+18% YoY', True)])
def test_plausible_rejects_ocr_confetti(text, ok):
    assert ocr.plausible(text) is ok


def test_track_reports_each_line_once_and_grows_word_reveals():
    samples = [(0, ['THIS']), (1, ['THIS IS']), (2, ['THIS IS HOW', '@delicious.homes']),
               (3, ['THIS IS HOW YOU CLOSE', '@delicious.homes']), (4, ['@delicious.homes']), (5, [])]
    events = ocr.track(samples, 1.0)
    assert [(e['start'], e['text']) for e in events] == [(0, 'THIS IS HOW YOU CLOSE'), (2, '@delicious.homes')]


def test_track_tolerates_flicker_and_misreads():
    samples = [(0, ['Book now at delicious.homes']), (1, []), (2, ['Book now at delic1ous.homes']), (3, ['Book now at delicious.homes'])]
    assert len(ocr.track(samples, 1.0)) == 1


def test_track_separates_distinct_captions_and_late_repeats():
    samples = [(0, ['first caption']), (1, ['second caption here']), (2, []), (3, []), (4, []), (9, ['first caption'])]
    assert [(e['start'], e['text']) for e in ocr.track(samples, 1.0)] == [
        (0, 'first caption'), (1, 'second caption here'), (9, 'first caption')]


def test_misread_partial_reveals_fold_into_the_full_caption():
    samples = [(1, ['THIS 1S']), (2, ['HOW']), (3, ['THIS IS HOW YOU CLOSE'])]
    assert [(e['start'], e['text']) for e in ocr.track(samples, 1.0)] == [(1, 'THIS IS HOW YOU CLOSE')]


def no_ocr_engines(monkeypatch):
    real = shutil.which
    monkeypatch.setattr(ocr.shutil, 'which', lambda name: None if name in ('tesseract', 'swift') else real(name))


def test_resolve_choices(monkeypatch):
    no_ocr_engines(monkeypatch)
    assert ocr.resolve('none') == (None, None)
    assert ocr.resolve('auto') == (None, None)
    assert ocr.resolve('tesseract')[1] and ocr.resolve('vision')[1]
    monkeypatch.setattr(ocr.shutil, 'which', lambda name: f'/bin/{name}')
    assert ocr.resolve('auto') == ('tesseract', None)
    with pytest.raises(SystemExit):
        ocr.resolve('magic')


def test_explicit_missing_engine_is_reported_not_fatal(monkeypatch, capsys, static_clip):
    no_ocr_engines(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['watch', str(static_clip), '--no-whisper', '--ocr', 'tesseract'])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert 'On-screen text unavailable: tesseract is not installed' in out and 'partial evidence' in out


def test_auto_without_engine_says_how_to_enable(monkeypatch, capsys, static_clip):
    no_ocr_engines(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['watch', str(static_clip), '--no-whisper', '--ocr', 'auto'])
    assert watch.main() == 0
    assert 'install tesseract' in capsys.readouterr().out


FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'ocr'


@pytest.fixture(scope='session')
def text_reel(tmp_path_factory):
    """Pop-in caption reveal, the full caption, then a slide: the shapes OCR must handle.

    Built from pre-rendered frames so it needs no drawtext filter or fonts; Homebrew's
    FFmpeg lacks drawtext, which used to skip the Apple Vision test on macOS.
    """
    path = tmp_path_factory.mktemp('ocr') / 'reel.mp4'
    inputs = []
    for name, seconds in (('caption-reveal.jpg', 2), ('caption-full.jpg', 2), ('slide.jpg', 3)):
        inputs += ['-loop', '1', '-framerate', '10', '-t', str(seconds), '-i', str(FIXTURES / name)]
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', *inputs, '-filter_complex',
                    '[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]', '-map', '[v]', str(path)], check=True)
    return path


def _assert_reads_reel(events):
    texts = [ocr.norm(e['text']) for e in events]
    assert any(ocr.similar(t, 'this is how you close') for t in texts), texts
    assert any('revenue' in t for t in texts), texts
    caption = next(e for e in events if 'close' in ocr.norm(e['text']))
    assert caption['start'] <= 2.0


@pytest.mark.skipif(not shutil.which('tesseract'), reason='tesseract not installed')
def test_tesseract_reads_captions_and_slides(text_reel, tmp_path):
    events, meta = ocr.on_screen_text(str(text_reel), tmp_path, 'tesseract', lang='eng', fps=1.0, start=0, end=7)
    assert meta['samples'] == 7 and not (tmp_path / 'ocr').exists()
    _assert_reads_reel(events)


@pytest.mark.skipif(sys.platform != 'darwin' or not shutil.which('swift'), reason='Apple Vision needs macOS + swift')
def test_apple_vision_reads_captions_and_slides(text_reel, tmp_path):
    events, _ = ocr.on_screen_text(str(text_reel), tmp_path, 'vision', lang='eng', fps=1.0, start=0, end=7)
    _assert_reads_reel(events)


@pytest.mark.skipif(not shutil.which('tesseract'), reason='tesseract not installed')
def test_watch_interleaves_text_rows_and_caches_them(monkeypatch, capsys, text_reel):
    monkeypatch.setenv('WATCH_OCR', 'tesseract')
    monkeypatch.setattr(sys, 'argv', ['watch', str(text_reel), '--no-whisper'])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert '] TEXT: ' in out and 'OCR via tesseract, 21 samples at 3 fps' in out and 'cached;' in out
    monkeypatch.setenv('WATCH_OCR', 'none')
    assert watch.main() == 0
    assert '] TEXT: ' not in capsys.readouterr().out, 'OCR settings are part of the cache key'


def ev(start, text, end=None):
    return {'start': start, 'end': end if end is not None else start + 0.4, 'text': text}


def test_word_pop_fragments_merge_into_caption_lines_and_slides_stay_text():
    events = [ev(0, 'how to structure'), ev(0.6, 'your sales calls'), ev(0.6, 'STAGES OF A SALES CALL: CLARIFY, LABEL'),
              ev(1.2, 'like the top 1%'), ev(5, 'label them'), ev(9, 'alone')]
    out = ocr.caption_runs(events)
    assert [(e['start'], e['kind'], e['text']) for e in out] == [
        (0, 'caption', 'how to structure your sales calls like the top 1%'),
        (0.6, 'text', 'STAGES OF A SALES CALL: CLARIFY, LABEL'),
        (5, 'text', 'label them'), (9, 'text', 'alone')]


def test_caption_runs_split_at_sentence_ends_and_size():
    out = ocr.caption_runs([ev(0, 'never open a call'), ev(0.5, 'like this.'), ev(1, 'do this instead')])
    assert [e['text'] for e in out] == ['never open a call like this.', 'do this instead']
    long = ocr.caption_runs([ev(i * 0.5, 'one two three four five') for i in range(10)])
    assert all(len(e['text'].split()) <= ocr.CAPTION_RUN_WORDS for e in long)


def test_captions_repeating_speech_are_dropped_but_text_is_kept():
    events = [{**ev(0, 'so open with their world', 2), 'kind': 'caption'},
              {**ev(3, 'quick question for you', 5), 'kind': 'caption'},
              {**ev(0, 'Book now at delicious.homes', 2), 'kind': 'text'}]
    speech = [{'start': 0, 'end': 2.5, 'text': 'So open with their world.'}]
    assert [e['text'] for e in ocr.drop_spoken(events, speech)] == ['quick question for you', 'Book now at delicious.homes']
    assert ocr.drop_spoken(events, []) == events


def test_recurring_short_misreads_are_background_noise():
    events = [ev(0, 'Cae the Vaceten'), ev(1, 'so people'), ev(4, 'Cae the Vacetan'), ev(8, 'Cae tho Vaceten'),
              ev(9, '@delicious.homes', 60)]
    assert [e['text'] for e in ocr.drop_recurring(events)] == ['so people', '@delicious.homes']


def test_misread_captions_and_short_fragments_that_were_spoken_are_dropped():
    speech = [{'start': 26, 'end': 35, 'text': "Hey John, quick question. I was looking at your operation and I'm calling about your competitor."},
              {'start': 8, 'end': 11, 'text': 'Salesman. Stop it.'}]
    events = [{**ev(26, 'Hey Joh quick vce quick question a looking at a your operation and im calling abou your com pet ito', 30), 'kind': 'caption'},
              {**ev(10, 'Stop it a', 11), 'kind': 'text'},
              {**ev(27, 'COMPETITOR INTEL CHECKLIST: locations, pricing, hiring', 30), 'kind': 'text'}]
    assert [e['text'] for e in ocr.drop_spoken(events, speech)] == ['COMPETITOR INTEL CHECKLIST: locations, pricing, hiring']


def test_heavily_misread_reveal_prefix_still_folds_into_its_caption():
    # Tesseract 4.1 (Ubuntu 22.04) reads the "THIS IS" reveal as "THES 1S".
    samples = [(1, ['THES 1S']), (2, []), (3, ['THIS IS HOW YOU CLOSE']), (4, ['THIS IS HOW YOU CLOSE'])]
    assert [(e['start'], e['text']) for e in ocr.track(samples, 1.0)] == [(1, 'THIS IS HOW YOU CLOSE')]


def test_short_text_that_stays_on_screen_is_not_a_caption():
    events = [ev(4, 'Q3 Revenue', 7.3), ev(4, 'Bookings up 18 percent', 7.3), ev(8, 'def handler(event):', 12),
              ev(8, 'return book(event)', 12), ev(9, 'so open'), ev(9.5, 'with their world')]
    out = ocr.caption_runs(events)
    assert [(e['kind'], e['text']) for e in out if e['kind'] == 'caption'] == [('caption', 'so open with their world')]
    assert sum(e['kind'] == 'text' for e in out) == 4
