"""Rolling-caption collapse, sentence merging, and the interleaved timeline."""
import json

import timeline
from transcribe import parse_vtt

# yt-dlp's YouTube auto-caption VTT: each cue repeats the previous line, then a
# 10 ms cue re-displays the finished line.
YOUTUBE_ROLLING = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.590 align:start position:0%
<SP>
all<00:00:00.320><c> right</c><00:00:00.560><c> so</c><00:00:00.960><c> here</c>

00:00:02.590 --> 00:00:02.600 align:start position:0%
all right so here
<SP>

00:00:02.600 --> 00:00:05.150 align:start position:0%
all right so here
we<00:00:02.879><c> are</c><00:00:03.040><c> in</c><00:00:03.200><c> front</c>

00:00:05.150 --> 00:00:05.160 align:start position:0%
we are in front
<SP>

00:00:05.160 --> 00:00:07.000 align:start position:0%
we are in front
of<00:00:05.400><c> the</c><00:00:05.600><c> elephants</c>
""".replace('<SP>', ' ')


def parse(tmp_path, body):
    path = tmp_path / 'captions.vtt'
    path.write_bytes(body.encode('utf-8'))
    return parse_vtt(str(path))


def test_youtube_rolling_captions_keep_each_word_once(tmp_path):
    out = parse(tmp_path, YOUTUBE_ROLLING)
    assert [s['text'] for s in out] == ['all right so here', 'we are in front', 'of the elephants']
    assert [s['start'] for s in out] == [0.0, 2.6, 5.16]


def test_lone_repeated_line_after_a_cue_is_independent_speech(tmp_path):
    out = parse(tmp_path, 'WEBVTT\n\n00:01.000 --> 00:02.000\nno\n\n00:02.000 --> 00:03.000\nno\n')
    assert [s['text'] for s in out] == ['no', 'no']


def test_repeated_first_line_after_a_gap_is_not_rolling(tmp_path):
    out = parse(tmp_path, 'WEBVTT\n\n00:01.000 --> 00:02.000\nchorus\n\n00:09.000 --> 00:11.000\nchorus\nagain\n')
    assert [s['text'] for s in out] == ['chorus', 'chorus again']


def seg(start, end, text):
    return {'start': start, 'end': end, 'text': text}


def test_fragments_merge_into_one_line_until_a_pause():
    lines = timeline.merge_lines([seg(0, 1, 'all right'), seg(1, 2, 'so here we'), seg(2, 3, 'are'),
                                  seg(6, 7, 'next thought')])
    assert [(l['start'], l['end'], l['text']) for l in lines] == [
        (0, 3, 'all right so here we are'), (6, 7, 'next thought')]


def test_sentence_end_splits_only_once_the_line_has_some_words():
    lines = timeline.merge_lines([seg(0, 1, 'Yes.'), seg(1, 2, 'We start with the first step.'),
                                  seg(2, 3, 'Then more.')])
    assert [l['text'] for l in lines] == ['Yes. We start with the first step.', 'Then more.']


def test_frame_boundary_and_speaker_change_split_lines():
    lines = timeline.merge_lines([seg(0, 1, 'one'), seg(1, 2, 'two'), seg(2, 3, '>> three'), seg(3, 4, 'four')],
                                 breaks=[0.0, 1.0])
    assert [l['text'] for l in lines] == ['one', 'two', '>> three four']


def test_size_bounds_split_long_runs():
    words = [seg(i, i + 1, 'word ' * 10) for i in range(5)]
    assert all(len(l['text'].split()) <= timeline.MAX_LINE_WORDS for l in timeline.merge_lines(words))
    slow = [seg(i * 1.4, i * 1.4 + 1, 'w') for i in range(30)]
    assert all(l['end'] - l['start'] <= timeline.MAX_LINE_SECONDS for l in timeline.merge_lines(slow))


def test_events_interleave_frames_before_speech_at_the_same_time():
    data = timeline.build(
        frames=[{'path': '/x/frame_0001.jpg', 'timestamp_seconds': 0.0, 'reason': 'first-frame'},
                {'path': '/x/frame_0002.jpg', 'timestamp_seconds': 3725.4, 'reason': 'scene-change'}],
        lines=[seg(0.0, 2.0, 'hello'), seg(3725.4, 3727, 'later')])
    assert timeline.events(data) == [
        '[00:00] FRAME frame_0001.jpg (first-frame)', '[00:00] hello',
        '[1:02:05] FRAME frame_0002.jpg (scene-change)', '[1:02:05] later']


def test_chapters_from_ytdlp_and_ffprobe():
    assert timeline.chapters_from({'chapters': [{'start_time': 0, 'end_time': 5, 'title': ' Intro  part '}]}) == [
        {'start': 0.0, 'title': 'Intro part'}]
    probe = {'chapters': [{'start_time': '4.000000', 'tags': {'title': 'Demo'}}, {'start_time': 'bad'}]}
    assert timeline.chapters_from({}, probe) == [{'start': 4.0, 'title': 'Demo'}]


def test_description_is_flattened_and_bounded():
    assert timeline.short_description(' a\n\nb ') == 'a b'
    long = timeline.short_description('word ' * 500)
    assert len(long) <= timeline.DESCRIPTION_CHARS + 2 and long.endswith(' …')
    assert timeline.short_description(None) is None


def test_write_emits_markdown_and_json(tmp_path):
    data = timeline.build(frames=[], lines=[seg(1, 2, 'hi')], speech_source='captions')
    timeline.write(data, tmp_path / 'out', 'body')
    assert (tmp_path / 'out' / 'timeline.md').read_text(encoding='utf-8') == 'body'
    assert json.loads((tmp_path / 'out' / 'timeline.json').read_text(encoding='utf-8'))['lines'][0]['text'] == 'hi'
