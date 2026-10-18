import os
import re
import json
import tempfile
import subprocess
from pathlib import Path

import gradio as gr
import imageio_ffmpeg
from groq import Groq

MODEL = "whisper-large-v3"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
API_KEY = os.environ.get("GROQ_API_KEY")


def run_ffmpeg(args):
    result = subprocess.run(
        [FFMPEG, "-y"] + args,
        capture_output=True,
        text=True,
        timeout=600
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1500:])


def duration(path):
    result = subprocess.run(
        [FFMPEG, "-i", path, "-f", "null", "-"],
        capture_output=True,
        text=True,
        timeout=120
    )
    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        result.stderr
    )
    if not match:
        raise RuntimeError("Audio duration could not be read.")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def transcribe(client, path, language):
    wav = os.path.join(
        tempfile.gettempdir(),
        next(tempfile._get_candidate_names()) + ".wav"
    )
    try:
        run_ffmpeg([
            "-i", path, "-vn", "-ac", "1",
            "-ar", "16000", "-c:a", "pcm_s16le", wav
        ])

        if os.path.getsize(wav) > 24 * 1024 * 1024:
            raise RuntimeError(
                "Converted file exceeds the API size limit."
            )

        with open(wav, "rb") as audio:
            response = client.audio.transcriptions.create(
                file=(Path(wav).name, audio),
                model=MODEL,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"]
            )

        words = []
        for item in (getattr(response, "words", None) or []):
            if isinstance(item, dict):
                word = item.get("word", "")
                start = item.get("start", 0)
                end = item.get("end", 0)
            else:
                word = item.word
                start = item.start
                end = item.end

            words.append({
                "word": str(word),
                "start": float(start),
                "end": float(end)
            })

        if not words:
            raise RuntimeError(
                "The transcription API returned no word timestamps."
            )

        return words

    finally:
        if os.path.exists(wav):
            os.remove(wav)


def tempo_filter(speed):
    if speed < 0.5 or speed > 2.0:
        raise RuntimeError(
            "A speech segment needs an extreme speed change. "
            "The automatic match is uncertain."
        )

    return f"atempo={speed:.6f}"


def synchronize(english_audio, hindi_audio):
    if not API_KEY:
        raise gr.Error("GROQ_API_KEY is missing in Render Environment.")

    if not english_audio or not hindi_audio:
        raise gr.Error("Upload both English and Hindi audio files.")

    client = Groq(api_key=API_KEY)
    work = tempfile.mkdtemp(prefix="audiosync_")

    try:
        en_words = transcribe(client, english_audio, "en")
        hi_words = transcribe(client, hindi_audio, "hi")

        en_duration = duration(english_audio)
        hi_duration = duration(hindi_audio)

        # Without translation-aware alignment, word positions cannot
        # reliably identify corresponding translated phrases.
        # This fallback creates speech blocks by proportional timing.
        # It is an approximation, not semantic sentence alignment.

        blocks = min(len(en_words), len(hi_words))
        if blocks < 2:
            raise RuntimeError("Not enough timestamped speech detected.")

        output = os.path.join(work, "synchronized_hindi.mp3")
        piece_paths = []

        # Divide each recording into equal-count groups.
        # This provides a rough timing approximation only.
        group_count = min(20, blocks)

        for i in range(group_count):
            e0 = round(i * len(en_words) / group_count)
            e1 = round((i + 1) * len(en_words) / group_count)
            h0 = round(i * len(hi_words) / group_count)
            h1 = round((i + 1) * len(hi_words) / group_count)

            if e1 <= e0 or h1 <= h0:
                continue

            target_start = en_words[e0]["start"]
            target_end = en_words[e1 - 1]["end"]
            source_start = hi_words[h0]["start"]
            source_end = hi_words[h1 - 1]["end"]

            target_len = target_end - target_start
            source_len = source_end - source_start

            if target_len <= 0 or source_len <= 0:
                continue

            # atempo speed = source length / desired target length
            speed = source_len / target_len
            if speed < 0.5 or speed > 2.0:
                # Skip unreliable extreme blocks rather than distort
                # the voice severely.
                continue

            piece = os.path.join(work, f"piece_{i:03d}.wav")

            run_ffmpeg([
                "-ss", str(source_start),
                "-t", str(source_len),
                "-i", hindi_audio,
                "-af", tempo_filter(speed),
                "-ar", "44100",
                "-ac", "2",
                piece
            ])

            piece_paths.append((target_start, piece))

        if not piece_paths:
            raise RuntimeError(
                "No segments could be aligned safely. "
                "Try shorter, clearer audio."
            )

        # Place each adjusted piece at its estimated English start time.
        inputs = []
        filters = []
        labels = []

        for i, (start, piece) in enumerate(piece_paths):
            inputs.extend(["-i", piece])
            delay_ms = max(0, int(start * 1000))
            label = f"a{i}"
            filters.append(
                f"[{i}:a]adelay={delay_ms}|{delay_ms}[{label}]"
            )
            labels.append(f"[{label}]")

        filters.append(
            "".join(labels) +
            f"amix=inputs={len(labels)}:duration=longest:normalize=0,"
            f"atrim=duration={en_duration}[out]"
        )

        run_ffmpeg(
            inputs + [
                "-filter_complex", ";".join(filters),
                "-map", "[out]",
                "-codec:a", "libmp3lame",
                "-b:a", "192k",
                output
            ]
        )

        return output, (
            "Audio processed. This is an approximate alignment: "
            "the system has not semantically verified translated "
            "sentence matches. Please listen and check synchronization."
        )

    except Exception as exc:
        raise gr.Error(str(exc)[:700])


with gr.Blocks(title="AudioSync AI") as app:
    gr.Markdown(
        "# AudioSync AI\n"
        "Upload English and Hindi audio. Scripts are not required."
    )

    with gr.Row():
        english = gr.Audio(label="English Audio", type="filepath")
        hindi = gr.Audio(label="Hindi Audio", type="filepath")

    button = gr.Button("Synchronize Audio", variant="primary")
    result = gr.Audio(label="Hindi Audio Output", type="filepath")
    status = gr.Textbox(label="Status")

    button.click(
        synchronize,
        inputs=[english, hindi],
        outputs=[result, status]
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
            )
