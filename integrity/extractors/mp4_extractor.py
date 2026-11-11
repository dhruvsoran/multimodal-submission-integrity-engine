"""MP4 extractor: audio-track extraction + Whisper transcription.

Pipeline per video:
1. Locate ffmpeg (PATH or the static binary shipped with ``imageio-ffmpeg``).
2. Extract the audio track to a 16 kHz mono WAV.
3. Detect empty / fully silent audio tracks (Scenario 3).
4. Transcribe with faster-whisper (tiny/base) using word-level timestamps.
5. Group transcribed words into sentences and overlapping chunks that carry
   real video timestamps, so cross-modal matches can cite "00:00:24 - 00:00:36".
"""

from __future__ import annotations

import re
import subprocess  # nosec B404 - fixed argv, no shell
import tempfile
import wave
from pathlib import Path
from typing import List, Optional, Tuple

from ..models import ExtractedDoc, fmt_ts
from ..normalize import chunk_units
from ..utils import get_ffmpeg_exe, get_logger

logger = get_logger(__name__)

# Module-level whisper model cache so multiple videos reuse one model load.
_whisper_model_cache: dict = {}


def _ffmpeg_run(args: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
    exe = get_ffmpeg_exe()
    if exe is None:
        raise RuntimeError(
            "ffmpeg binary not found. Install ffmpeg or 'pip install imageio-ffmpeg'."
        )
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y"] + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def extract_audio(path: Path, tmpdir: str) -> Tuple[Path, Optional[str]]:
    """Extract 16 kHz mono PCM audio; return (wav_path, error)."""
    wav = Path(tmpdir) / f"{path.stem}_audio.wav"
    proc = _ffmpeg_run(
        ["-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)]
    )
    stderr = (proc.stderr or "").lower()
    if proc.returncode != 0 or not wav.exists() or wav.stat().st_size <= 44:
        if "does not contain any stream" in stderr or "output file #0 does not" in stderr:
            return wav, "Video has no audio track (silent video)."
        if "is not a valid" in stderr or "invalid data" in stderr:
            return wav, "File is not a readable video (corrupted container)."
        return wav, f"Audio extraction failed: {stderr.strip()[:200] or 'unknown ffmpeg error'}"
    return wav, None


def _load_wav_as_float(wav_path: Path) -> Tuple[int, "np.ndarray"]:
    """Decode PCM WAV with the stdlib (avoids PyAV decoder incompatibilities).

    faster-whisper's bundled PyAV path breaks with PyAV>=15/19 on newer Python
    versions (``open() got an unexpected keyword argument 'metadata_errors'``),
    but its transcribe() accepts a float32 numpy array directly.  Since the
    pipeline always extracts audio via ffmpeg to 16 kHz mono WAV first, we can
    hand the decoded array straight to the model - no PyAV involved.
    """
    import array as _array

    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if width == 2:
        samples = _array.array("h")
        samples.frombytes(raw[: len(raw) // 2 * 2])
        audio = np.frombuffer(samples, dtype=np.int16).astype("float32") / 32768.0
    elif width == 4:
        samples = _array.array("i")
        samples.frombytes(raw[: len(raw) // 4 * 4])
        audio = np.frombuffer(samples, dtype=np.int32).astype("float32") / 2147483648.0
    elif width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype("float32") - 128.0) / 128.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {width} bytes")
    return rate, audio


def _is_silent(wav_path: Path) -> bool:
    """True when the decoded track carries (near-)zero amplitude."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            width = wf.getsampwidth()
    except Exception:
        return True
    if not frames:
        return True
    if width == 2:
        import array

        samples = array.array("h")
        samples.frombytes(frames[: len(frames) // 2 * 2])
        peak = max((abs(s) for s in samples), default=0)
        return peak < 150  # effectively digital silence
    return False


def _get_transcriber(model_size: str, device: str):
    """Lazy-load faster-whisper; returns (model_fn_factory, backend_name)."""
    key = (model_size, device)
    if key in _whisper_model_cache:
        return _whisper_model_cache[key], "faster-whisper"
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc
    model = WhisperModel(model_size, device=device, compute_type="int8")
    _whisper_model_cache[key] = model
    return model, "faster-whisper"


_WORD_RE = re.compile(r"[\w']+")
_SENT_END = (".", "!", "?")


def _words_to_sentence_units(
    words: List[Tuple[float, float, str]], max_words_per_sentence: int = 26
) -> List[Tuple[str, float, float]]:
    """Group the timed word stream into sentence-like units.

    Splits after sentence-final punctuation produced by the ASR; when the
    model emits no punctuation (some languages/domains) we fall back to
    fixed-size windows so chunking still works.
    """
    units: List[Tuple[str, float, float]] = []
    buf: List[str] = []
    start: Optional[float] = None
    end: Optional[float] = None
    since_split = 0
    for w_start, w_end, token in words:
        if start is None:
            start = w_start
        end = w_end
        buf.append(token)
        since_split += 1
        ends_sentence = token.endswith(_SENT_END)
        if ends_sentence or since_split >= max_words_per_sentence:
            text = " ".join(buf).strip()
            if text:
                units.append((text, start or 0.0, end or 0.0))
            buf, start, since_split = [], None, 0
    text = " ".join(buf).strip()
    if text:
        units.append((text, start or 0.0, end or 0.0))
    return units


def extract_mp4(
    path: str | Path,
    whisper_model: str = "base",
    device: str = "cpu",
    chunk_target_words: int = 140,
    chunk_overlap_words: int = 35,
    min_words: int = 12,
) -> ExtractedDoc:
    """Transcribe a short video pitch into timestamped chunks.

    Malformed / silent / audio-less videos return ``ok=False`` or an empty
    chunk list with a warning instead of crashing the pipeline.
    """
    path = Path(path)
    doc = ExtractedDoc(file_name=path.name, file_path=str(path), modality="mp4")
    try:
        if path.stat().st_size == 0:
            doc.error = "File is empty (zero bytes) - corrupted submission."
            logger.warning(f"[mp4] {path.name}: {doc.error}")
            return doc
        with tempfile.TemporaryDirectory() as tmpdir:
            wav, audio_err = extract_audio(path, tmpdir)
            if audio_err:
                doc.error = audio_err
                logger.warning(f"[mp4] {path.name}: {doc.error}")
                return doc
            if _is_silent(wav):
                doc.error = "Audio track exists but is completely silent (no speech)."
                logger.warning(f"[mp4] {path.name}: {doc.error}")
                return doc

            model, backend = _get_transcriber(whisper_model, device)
            rate, audio = _load_wav_as_float(wav)
            if rate != 16000:
                doc.warnings.append(f"Audio sample rate {rate} != 16 kHz; resampling recommended.")
            segments, info = model.transcribe(
                audio,
                word_timestamps=True,
                vad_filter=True,
                beam_size=1,
                language="en",
            )
            words: List[Tuple[float, float, str]] = []
            for seg in segments:
                for w in seg.words or []:
                    token = (w.word or "").strip()
                    if token:
                        words.append((w.start or 0.0, w.end or 0.0, token))
            doc.language = info.language
            doc.duration_s = float(info.duration or 0.0)
            if not words:
                doc.error = "Transcription produced no words (possibly music/noise only)."
                logger.warning(f"[mp4] {path.name}: {doc.error}")
                return doc

            units = _words_to_sentence_units(words)
            sent_units = [
                (text, None, start, end) for text, start, end in units
            ]
            doc.chunks = chunk_units(
                sent_units,
                path.name,
                "mp4",
                target_words=chunk_target_words,
                overlap_words=chunk_overlap_words,
                min_words=min_words,
            )
            doc.text_length = sum(len(t) for t, _, _ in units)
            doc.full_text = " ".join(t for t, _, _ in units)
            span = (
                f"{fmt_ts(doc.chunks[0].start_ts)}-{fmt_ts(doc.chunks[-1].end_ts)}"
                if doc.chunks
                else "n/a"
            )
            logger.info(
                f"[mp4] {path.name}: {doc.duration_s:.1f}s, lang={doc.language} "
                f"({backend}) -> {len(doc.chunks)} chunks, span {span}"
            )
            return doc
    except Exception as exc:  # noqa: BLE001
        doc.error = f"Failed to process video: {exc}"
        logger.warning(f"[mp4] {path.name}: {doc.error}")
        return doc


def synthesis_note() -> str:  # pragma: no cover - documentation helper
    return "Word timestamps enable precise multimodal citations."
