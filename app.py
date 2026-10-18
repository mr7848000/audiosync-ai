import os
import re
import math
import tempfile
import subprocess
import unicodedata
from pathlib import Path

import gradio as gr
import imageio_ffmpeg
from groq import Groq

# ---------------- CONFIG ----------------

TRANSCRIPTION_MODEL = "whisper-large-v3"
API_KEY = os.environ.get("GROQ_API_KEY")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

MAX_FILE_MB = 24

# Matching limits. These are heuristic, not guaranteed alignment.
MAX_GROUP = 4
MIN_SPEED = 0.75
MAX_SPEED = 1.35


# ---------------- AUDIO HELPERS ----------------

def run_ffmpeg(args):
    result = subprocess.run(
        [FFMPEG, "-y"] + args,
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr[-1800:] or "FFmpeg processing failed."
        )


def get_duration(path):
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
        raise RuntimeError("Could not read audio duration.")

    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


# ---------------- TRANSCRIPTION ----------------

def transcribe(client, audio_path, language):
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    if size_mb > MAX_FILE_MB:
        raise RuntimeError(
            f"{Path(audio_path).name} is {size_mb:.1f} MB. "
            "This version accepts files up to 24 MB. "
            "Compress the MP3 and try again."
        )

    with open(audio_path, "rb") as audio:
        response = client.audio.transcriptions.create(
            file=(Path(audio_path).name, audio),
            model=TRANSCRIPTION_MODEL,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    segments = []

    for item in getattr(response, "segments", None) or []:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            text = item.get("text", "")
        else:
            start = item.start
            end = item.end
            text = item.text

        if (
            start is not None
            and end is not None
            and str(text).strip()
            and float(end) > float(start)
        ):
            segments.append({
                "start": float(start),
                "end": float(end),
                "text": str(text).strip()
            })

    if not segments:
        raise RuntimeError(
            f"No timestamped speech returned for {language}."
        )

    segments.sort(key=lambda item: item["start"])
    return segments


# ---------------- TEXT FEATURES ----------------

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def segment_duration(segment):
    return max(0.05, segment["end"] - segment["start"])


def group_segments(segments, start, count):
    selected = segments[start:start + count]

    if not selected:
        return None

    return {
        "first": start,
        "last": start + len(selected) - 1,
        "start": selected[0]["start"],
        "end": selected[-1]["end"],
        "duration": max(
            0.05,
            selected[-1]["end"] - selected[0]["start"]
        ),
        "text": " ".join(item["text"] for item in selected)
    }


# ---------------- SEQUENCE ALIGNMENT ----------------

def align_sequences(english_segments, hindi_segments):
    """
    Monotonic approximate alignment using dynamic programming.

    It groups consecutive transcript segments while preserving
    the order of both transcripts. It does NOT understand translation
    semantics, so the resulting alignment must be reviewed.
    """

    n = len(english_segments)
    m = len(hindi_segments)

    if not n or not m:
        raise RuntimeError("One transcript contains no speech.")

    # Normalize cumulative duration to compare relative timeline position.
    en_total = max(0.1, english_segments[-1]["end"])
    hi_total = max(0.1, hindi_segments[-1]["end"])

    en_mid = [
        (s["start"] + s["end"]) / 2 / en_total
        for s in english_segments
    ]
    hi_mid = [
        (s["start"] + s["end"]) / 2 / hi_total
        for s in hindi_segments
    ]

    # DP state: (i, j) means first i English and j Hindi segments
    # have been assigned. Each transition consumes 1..MAX_GROUP
    # consecutive segments from each side.
    infinity = float("inf")
    dp = [[infinity] * (m + 1) for _ in range(n + 1)]
    parent = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    def group_midpoint(values, begin, count):
        return sum(values[begin:begin + count]) / count

    for i in range(n + 1):
        for j in range(m + 1):
            current = dp[i][j]

            if math.isinf(current):
                continue

            # Match one group to one group, allowing different
            # numbers of transcript segments on each side.
            for a in range(1, min(MAX_GROUP, n - i) + 1):
                en_group = group_segments(english_segments, i, a)

                for b in range(1, min(MAX_GROUP, m - j) + 1):
                    hi_group = group_segments(hindi_segments, j, b)

                    en_midpoint = group_midpoint(en_mid, i, a)
                    hi_midpoint = group_midpoint(hi_mid, j, b)

                    # Penalize groups far apart in normalized timeline.
                    timeline_cost = abs(en_midpoint - hi_midpoint)

                    # Penalize wildly different group durations.
                    duration_ratio = (
                        en_group["duration"] / hi_group["duration"]
                    )
                    duration_cost = abs(math.log(
                        max(0.05, duration_ratio)
                    ))

                    # Penalize many-to-many groupings slightly.
                    grouping_cost = 0.025 * (a + b - 2)

                    # This is a timing heuristic, not semantic matching.
                    cost = (
                        2.0 * timeline_cost
                        + 0.10 * duration_cost
                        + grouping_cost
                    )

                    ni = i + a
                    nj = j + b
                    candidate = current + cost

                    if candidate < dp[ni][nj]:
                        dp[ni][nj] = candidate
                        parent[ni][nj] = (i, j, a, b)

    if math.isinf(dp[n][m]):
        raise RuntimeError("Could not create a complete alignment.")

    groups = []
    i, j = n, m

    while i > 0 or j > 0:
        step = parent[i][j]

        if step is None:
            raise RuntimeError("Alignment reconstruction failed.")

        old_i, old_j, a, b = step

        groups.append({
            "en": list(range(old_i, old_i + a)),
            "hi": list(range(old_j, old_j + b))
        })

        i, j = old_i, old_j

    groups.reverse()
    return groups


# ---------------- SPEED ADJUSTMENT ----------------

def make_atempo(speed):
    if speed <= 0:
        raise ValueError("Invalid speed ratio.")

    filters = []

    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0

    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5

    filters.append(f"atempo={speed:.6f}")
    return ",".join(filters)


# ---------------- SYNCHRONIZATION ----------------

def synchronize(english_audio, hindi_audio):
    if not API_KEY:
        raise gr.Error(
            "GROQ_API_KEY is missing in Render Environment."
        )

    if not english_audio or not hindi_audio:
        raise gr.Error("Upload both English and Hindi audio files.")

    client = Groq(api_key=API_KEY)
    work_dir = tempfile.mkdtemp(prefix="audiosync_")

    try:
        en_duration = get_duration(english_audio)
        hi_duration = get_duration(hindi_audio)

        en_segments = transcribe(client, english_audio, "en")
        hi_segments = transcribe(client, hindi_audio, "hi")

        groups = align_sequences(en_segments, hi_segments)

        pieces = []
        warnings = []

        for index, group in enumerate(groups):
            en_ids = group["en"]
            hi_ids = group["hi"]

            en_first = en_segments[en_ids[0]]
            en_last = en_segments[en_ids[-1]]
            hi_first = hi_segments[hi_ids[0]]
            hi_last = hi_segments[hi_ids[-1]]

            source_start = max(0.0, hi_first["start"])
            source_end = min(hi_duration, hi_last["end"])

            target_start = max(0.0, en_first["start"])
            target_end = min(en_duration, en_last["end"])

            source_len = source_end - source_start
            target_len = target_end - target_start

            if source_len <= 0 or target_len <= 0:
                warnings.append(
                    f"Group {index + 1}: invalid timing."
                )
                continue

            speed = source_len / target_len

            # Avoid unnatural extreme stretching.
            if speed < MIN_SPEED or speed > MAX_SPEED:
                warnings.append(
                    f"Group {index + 1}: needed speed {speed:.2f}x; "
                    "kept original speed to protect voice quality."
                )
                speed = 1.0

            piece_path = os.path.join(
                work_dir,
                f"piece_{index:04d}.wav"
            )

            run_ffmpeg([
                "-ss", str(source_start),
                "-t", str(source_len),
                "-i", hindi_audio,
                "-af", make_atempo(speed),
                "-ar", "44100",
                "-ac", "2",
                piece_path
            ])

            pieces.append({
                "path": piece_path,
                "start": target_start
            })

        if not pieces:
            raise RuntimeError("No usable Hindi segments were created.")

        inputs = []
        filters = []
        labels = []

        for i, piece in enumerate(pieces):
            inputs.extend(["-i", piece["path"]])

            delay_ms = int(piece["start"] * 1000)
            label = f"seg{i}"

            filters.append(
                f"[{i}:a]adelay={delay_ms}|{delay_ms}[{label}]"
            )
            labels.append(f"[{label}]")

        filters.append(
            "".join(labels)
            + f"amix=inputs={len(labels)}:duration=longest:"
              f"normalize=0[mixed]"
        )

        output_path = os.path.join(
            work_dir,
            "synchronized_hindi.mp3"
        )

        run_ffmpeg(
            inputs + [
                "-filter_complex", ";".join(filters),
                "-map", "[mixed]",
                "-t", str(en_duration),
                "-codec:a", "libmp3lame",
                "-b:a", "192k",
                output_path
            ]
        )

        final_duration = get_duration(output_path)

        status = (
            "Processing completed.\n"
            f"English duration: {en_duration:.2f}s\n"
            f"Hindi duration: {hi_duration:.2f}s\n"
            f"Output duration: {final_duration:.2f}s\n"
            f"English speech segments: {len(en_segments)}\n"
            f"Hindi speech segments: {len(hi_segments)}\n"
            f"Alignment groups: {len(groups)}\n"
            "IMPORTANT: Alignment is timing-based and approximate. "
            "Listen to verify that sentences match."
        )

        if warnings:
            status += (
                "\n\nWarnings:\n- "
                + "\n- ".join(warnings[:15])
            )

        return output_path, status

    except gr.Error:
        raise

    except Exception as exc:
        raise gr.Error(
            f"Processing failed: {str(exc)[:900]}"
        )


# ---------------- USER INTERFACE ----------------

with gr.Blocks(title="AudioSync AI") as app:
    gr.Markdown(
        "# AudioSync AI\n"
        "Upload English and Hindi audio. "
        "Transcription and timestamps are generated automatically.\n\n"
        "No scripts are required. Alignment is approximate and "
        "should be reviewed before publishing."
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

    run_button = gr.Button(
        "Synchronize Audio",
        variant="primary"
    )

    output_audio = gr.Audio(
        label="Synchronized Hindi MP3",
        type="filepath"
    )

    status = gr.Textbox(
        label="Processing Status",
        lines=10
    )

    run_button.click(
        fn=synchronize,
        inputs=[english_audio, hindi_audio],
        outputs=[output_audio, status]
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
