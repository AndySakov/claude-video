#!/usr/bin/env python3
"""sherpa-onnx transcription worker. Runs inside watch's sherpa-onnx venv, never inside watch.

Reads a 16 kHz mono 16-bit WAV, cuts speech with Silero VAD, recognizes each piece
with an offline model (NeMo Parakeet, Moonshine, or SenseVoice), optionally labels
speakers with pyannote segmentation + a speaker-embedding model, and prints
``{"segments": [...], "speakers": N}`` JSON to ``--out``. Standard library + sherpa_onnx only;
samples stay in a float32 ``array`` (an hour of audio is ~230 MB, not a list of Python floats).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from array import array
from pathlib import Path

SAMPLE_RATE = 16000
SENTENCE_END = re.compile(r'[.?!…。？！]$')
MIN_SPLIT_SECONDS = 1.0  # don't cut a piece into fragments shorter than this


def read_wav(path: str) -> array:
    with wave.open(path, 'rb') as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
            raise SystemExit('sherpa_runner expects 16 kHz mono 16-bit PCM WAV')
        pcm = array('h', w.readframes(w.getnframes()))
    if sys.byteorder == 'big':
        pcm.byteswap()
    return array('f', (s / 32768.0 for s in pcm))


def recognizer(kind: str, model: Path, threads: int):
    import sherpa_onnx
    if kind == 'parakeet':
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model / 'encoder.int8.onnx'), decoder=str(model / 'decoder.int8.onnx'),
            joiner=str(model / 'joiner.int8.onnx'), tokens=str(model / 'tokens.txt'),
            num_threads=threads, model_type='nemo_transducer')
    if kind == 'moonshine':
        return sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=str(model / 'preprocess.onnx'), encoder=str(model / 'encode.int8.onnx'),
            uncached_decoder=str(model / 'uncached_decode.int8.onnx'),
            cached_decoder=str(model / 'cached_decode.int8.onnx'), tokens=str(model / 'tokens.txt'),
            num_threads=threads)
    if kind == 'sensevoice':
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model / 'model.int8.onnx'), tokens=str(model / 'tokens.txt'),
            num_threads=threads, use_itn=True)
    raise SystemExit(f'unknown model kind {kind}')


def speech_pieces(samples: array, vad_model: Path, max_seconds: float):
    """(start_seconds, samples) for each VAD speech region."""
    import sherpa_onnx
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(vad_model)
    config.silero_vad.min_silence_duration = 0.3
    config.silero_vad.max_speech_duration = max_seconds
    config.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=max(60.0, max_seconds * 3))
    window = config.silero_vad.window_size
    pieces = []
    def drain():
        while not vad.empty():
            pieces.append((vad.front.start / SAMPLE_RATE, vad.front.samples))
            vad.pop()
    for i in range(0, len(samples), window):
        vad.accept_waveform(samples[i:i + window])
        drain()
    vad.flush()
    drain()
    return pieces


def split_sentences(text: str, tokens: list[str], stamps: list[float], start: float, end: float) -> list[dict]:
    """Cut one recognized piece at sentence ends using token timestamps, when the model gives them."""
    if not tokens or len(tokens) != len(stamps):
        return [{'start': start, 'end': end, 'text': text}]
    out, current, piece_start = [], '', start
    for i, token in enumerate(tokens):
        current += token
        t = start + stamps[i]
        nxt = start + stamps[i + 1] if i + 1 < len(tokens) else end
        if SENTENCE_END.search(current.strip()) and i + 1 < len(tokens) and nxt - piece_start >= MIN_SPLIT_SECONDS:
            out.append({'start': piece_start, 'end': nxt, 'text': current.strip()})
            current, piece_start = '', nxt
    if current.strip():
        out.append({'start': piece_start, 'end': end, 'text': current.strip()})
    # BPE pieces use "▁" (or a leading space) as the word boundary.
    for seg in out:
        seg['text'] = ' '.join(seg['text'].replace('▁', ' ').split())
    return [s for s in out if s['text']]


def transcribe(samples, args) -> list[dict]:
    rec = recognizer(args.kind, Path(args.model), args.threads)
    segments = []
    for start, piece in speech_pieces(samples, Path(args.vad), args.max_piece):
        stream = rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, piece)
        rec.decode_stream(stream)
        result = stream.result
        text = ' '.join(result.text.split())
        if not text:
            continue
        end = start + len(piece) / SAMPLE_RATE
        segments += split_sentences(text, list(getattr(result, 'tokens', []) or []),
                                    list(getattr(result, 'timestamps', []) or []), start, end)
        print(f'progress {end:.1f}', file=sys.stderr, flush=True)
    return segments


def diarizer(args):
    import sherpa_onnx
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=args.segmentation),
            num_threads=args.threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=args.embedding, num_threads=args.threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=args.speakers or -1, threshold=args.threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    if not config.validate():
        raise SystemExit('invalid speaker-diarization models')
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def diarize(samples, args) -> list[tuple[float, float, int]]:
    result = diarizer(args).process(samples).sort_by_start_time()
    return [(r.start, r.end, r.speaker) for r in result]


def label(segments: list[dict], turns: list[tuple[float, float, int]]) -> int:
    """Give each segment the speaker who talks most during it; speakers numbered by first appearance."""
    order: dict[int, int] = {}
    for seg in segments:
        overlap: dict[int, float] = {}
        for start, end, speaker in turns:
            shared = min(end, seg['end']) - max(start, seg['start'])
            if shared > 0:
                overlap[speaker] = overlap.get(speaker, 0.0) + shared
        if overlap:
            speaker = max(overlap, key=overlap.get)
            seg['speaker'] = f'SPK{order.setdefault(speaker, len(order) + 1)}'
    return len(order)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('wav')
    parser.add_argument('--out', required=True)
    parser.add_argument('--kind', required=True, choices=['parakeet', 'moonshine', 'sensevoice'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--vad', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-piece', type=float, default=15.0)
    parser.add_argument('--segmentation')
    parser.add_argument('--embedding')
    parser.add_argument('--speakers', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.7)  # tuned for TitaNet-small embeddings
    parser.add_argument('--check', action='store_true', help='load every model, then exit (setup warm-up)')
    args = parser.parse_args()
    samples = read_wav(args.wav)
    if args.check:
        recognizer(args.kind, Path(args.model), args.threads)
        speech_pieces(samples, Path(args.vad), args.max_piece)
        if args.segmentation and args.embedding:
            diarizer(args)
        Path(args.out).write_text('{"segments": [], "speakers": 0}', encoding='utf-8')
        return 0
    segments = transcribe(samples, args)
    speakers = 0
    if args.segmentation and args.embedding and segments:
        speakers = label(segments, diarize(samples, args))
        if speakers < 2:
            for seg in segments:
                seg.pop('speaker', None)
    Path(args.out).write_text(json.dumps({'segments': segments, 'speakers': speakers}, ensure_ascii=False),
                              encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
