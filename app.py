import os
import re
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
import imageio_ffmpeg
from groq import Groq

# ---------------- CONFIGURATION ----------------

MODEL = "whisper-large-v3"
MAX_FILE_MB = 24

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
API_KEY = os.environ.get("GROQ_API_KEY")

# ---------------- AUDIO UTILITIES ----------------

def run_ffmpeg(args):
    command = [FFMPEG, "-y"] + args
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=600
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr[-2000:] or "FFmpeg failed."
        )


def get_duration(path):
    result = subprocess.run(
        [
            FFMPEG,
            "-i", path,
            "-f", "null",
            "-"
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    output = result.stderr

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        output
    )

    if not match:
        raise RuntimeError("Could not determine audio duration.")

    hours, minutes, seconds = match.groups()

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + float(seconds)
    )


def convert_to_wav(source, destination):
    run_ffmpeg([
        "-i", source,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        destination
    ])


def split_sentences(text):
    text = text.strip()

    if not text:
        return []

    parts = re.split(r"(?<=[.!?।])\s+", text)

    return [p.strip() for p in parts if p.strip()]


# ---------------- SPEECH RECOGNITION ----------------

def transcribe_audio(client, path, language):
    wav_path = str(Path(path).with_suffix(".wav"))
    convert_to_wav(path, wav_path)

    try:
        size_mb = os.path.getsize(wav_path) / (1024 * 1024)

        if size_mb > MAX_FILE_MB:
            raise RuntimeError(
                "Converted audio exceeds the API file-size limit. "
                "Please use a shorter audio file."
            )

        with open(wav_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=(Path(wav_path).name, audio_file),
                model=MODEL,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"]
            )

        words = []

        for item in (getattr(response, "words", None) or []):
            words.append({
    "word": item.get("word") if isinstance(item, dict) else item.word,
    "start": float(item.get("start") if isinstance(item, dict) else item.start),
    "end": float(item.get("end") if isinstance(item, dict) else item.end)
})

        return {
            "text": response.text,
            "words": words,
            "duration": get_duration(path)
        }

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ---------------- ALIGNMENT HELPERS ----------------

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_words_by_script(words, script):
    """
    Approximate script alignment using recognized word order.
    This is not a guaranteed forced aligner.
    """

    sentences = split_sentences(script)

    if not sentences:
        raise ValueError("A script is empty.")

    if not words:
        raise ValueError(
            "Word timestamps were not returned by the API."
        )

    script_tokens = [
        len(normalize(sentence).split())
        for sentence in sentences
    ]

    total_tokens = sum(script_tokens)

    if total_tokens == 0:
        raise ValueError("Could not read the script.")

    result = []
    cursor = 0
    total_words = len(words)

    for index, sentence in enumerate(sentences):
        if index == len(sentences) - 1:
            end_cursor = total_words
        else:
            proportion = sum(script_tokens[:index + 1]) / total_tokens
            end_cursor = round(proportion * total_words)

        end_cursor = max(cursor + 1, end_cursor)
        end_cursor = min(end_cursor, total_words)

        group = words[cursor:end_cursor]

        if not group:
            continue

        result.append({
            "sentence": sentence,
            "start": float(group[0]["start"]),
            "end": float(group[-1]["end"])
        })

        cursor = end_cursor

    return result


def build_tempo_filter(source_duration, target_duration):
    """
    FFmpeg atempo preserves pitch reasonably well.
    Extreme speed changes are intentionally rejected.
    """

    if source_duration <= 0 or target_duration <= 0:
        raise ValueError("Invalid sentence duration.")

    speed = source_duration / target_duration

    if speed < 0.40 or speed > 2.50:
    raise ValueError(
        "Sentence timing is too different. "
        "The sentence mapping needs correction."
    )

    filters = []

    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0

    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5

    filters.append(f"atempo={speed:.6f}")

    return ",".join(filters)


# ---------------- MAIN PROCESSING ----------------

def synchronize(
    english_audio,
    hindi_audio,
    english_script,
    hindi_script
):
    if not API_KEY:
        raise gr.Error(
            "GROQ_API_KEY is missing. Add it in Render Environment."
        )

    if not all([
        english_audio,
        hindi_audio,
        english_script,
        hindi_script
    ]):
        raise gr.Error("Upload both audios and both scripts.")

    client = Groq(api_key=API_KEY)

    temp_dir = tempfile.mkdtemp(prefix="audiosync_")

    try:
        english_data = transcribe_audio(
            client,
            english_audio,
            "en"
        )

        hindi_data = transcribe_audio(
            client,
            hindi_audio,
            "hi"
        )

        english_words = english_data["words"]
        hindi_words = hindi_data["words"]

        if not english_words or not hindi_words:
            raise gr.Error(
                "Word timestamps are unavailable. "
                "The API response did not contain word-level timing."
            )

        english_segments = split_words_by_script(
            english_words,
            english_script
        )

        hindi_segments = split_words_by_script(
            hindi_words,
            hindi_script
        )

        # IMPORTANT:
        # This first version pairs sentences by sequence.
        # It cannot reliably resolve many-to-one translations.

        if len(english_segments) != len(hindi_segments):
            raise gr.Error(
                "The scripts contain different numbers of sentences: "
                f"English={len(english_segments)}, "
                f"Hindi={len(hindi_segments)}. "
                "Automatic pairing is ambiguous. "
                "The next version needs explicit sentence mapping."
            )

        english_duration = english_data["duration"]
        hindi_duration = hindi_data["duration"]

        output_path = os.path.join(
            temp_dir,
            "synchronized_hindi.mp3"
        )

        pieces = []
        timeline = 0.0

        # Create one adjusted Hindi segment per English segment.
        for i, (en, hi) in enumerate(
            zip(english_segments, hindi_segments)
        ):
            source_start = max(0.0, hi["start"])
            source_end = min(hindi_duration, hi["end"])

            target_start = max(0.0, en["start"])
            target_end = min(english_duration, en["end"])

            source_duration = source_end - source_start
            target_duration = target_end - target_start

            if source_duration <= 0 or target_duration <= 0:
                raise gr.Error(
                    f"Invalid timing in segment {i + 1}."
                )

            tempo = build_tempo_filter(
                source_duration,
                target_duration
            )

            piece_path = os.path.join(
                temp_dir,
                f"piece_{i:03d}.mp3"
            )

            run_ffmpeg([
                "-ss", str(source_start),
                "-t", str(source_duration),
                "-i", hindi_audio,
                "-af", tempo,
                "-ar", "44100",
                "-ac", "2",
                "-b:a", "192k",
                piece_path
            ])

            pieces.append(piece_path)

        # Concatenate the adjusted pieces.
        # This basic version does not yet insert silence at
        # the original English sentence timestamps.

        concat_file = os.path.join(temp_dir, "concat.txt")

        with open(concat_file, "w", encoding="utf-8") as f:
            for piece in pieces:
                safe_path = piece.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        run_ffmpeg([
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            output_path
        ])

        return (
            output_path,
            "Processing completed. Please review the output; "
            "this prototype does not guarantee exact synchronization."
        )

    except gr.Error:
        raise

    except Exception as exc:
        raise gr.Error(
            f"Processing failed: {str(exc)[:700]}"
        )

    finally:
        # Do not delete output files here; Gradio needs the file.
        # Temporary files will be cleaned by the OS/service later.
        pass


# ---------------- USER INTERFACE ----------------

with gr.Blocks(title="AudioSync AI") as app:
    gr.Markdown(
        "# AudioSync AI\n"
        "Upload English and Hindi audio with their scripts."
    )

    with gr.Row():
        english_audio = gr.Audio(
            label="English Audio",
            type="filepath"
        )

        hindi_audio = gr.Audio(
            label="Hindi Audio",
            type="filepath"
        )

    with gr.Row():
        english_script = gr.Textbox(
            label="English Script",
            lines=8
        )

        hindi_script = gr.Textbox(
            label="Hindi Script",
            lines=8
        )

    run_button = gr.Button(
        "Synchronize Audio",
        variant="primary"
    )

    output_audio = gr.Audio(
        label="Synchronized Hindi MP3",
        type="filepath"
    )

    status = gr.Textbox(label="Status")

    run_button.click(
        fn=synchronize,
        inputs=[
            english_audio,
            hindi_audio,
            english_script,
            hindi_script
        ],
        outputs=[output_audio, status]
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )
