import os
import re
import json
import tempfile
import subprocess
from pathlib import Path

import gradio as gr
from groq import Groq
import imageio_ffmpeg

# ---------------- CONFIG ----------------

TRANSCRIPTION_MODEL = "whisper-large-v3"
ALIGNMENT_MODEL = "llama-3.3-70b-versatile"

API_KEY = os.environ.get("GROQ_API_KEY")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ---------------- AUDIO HELPERS ----------------

def run_ffmpeg(args):
    result = subprocess.run(
        [FFMPEG, "-y"] + args,
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1800:])


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


def transcribe(client, audio_path, language):
    size = os.path.getsize(audio_path)

    if size > 24 * 1024 * 1024:
        raise RuntimeError(
            "Audio file is too large for this API request. "
            "Please use an MP3 smaller than 24 MB."
        )

    with open(audio_path, "rb") as audio:
        result = client.audio.transcriptions.create(
            file=(Path(audio_path).name, audio),
            model=TRANSCRIPTION_MODEL,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    segments = []

    for item in (getattr(result, "segments", None) or []):
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            text = item.get("text", "")
        else:
            start = item.start
            end = item.end
            text = item.text

        if start is not None and end is not None and text.strip():
            segments.append({
                "start": float(start),
                "end": float(end),
                "text": text.strip()
            })

    if not segments:
        raise RuntimeError(
            f"No timestamped speech was returned for {language}."
        )

    return segments


# ---------------- AI ALIGNMENT ----------------

def align_segments(client, english_segments, hindi_segments):
    """
    Ask the language model to map English transcript groups
    to corresponding Hindi transcript groups.

    Segment indices refer to the timestamped transcription arrays.
    """

    english_text = [
        {"i": i, "text": seg["text"]}
        for i, seg in enumerate(english_segments)
    ]

    hindi_text = [
        {"i": i, "text": seg["text"]}
        for i, seg in enumerate(hindi_segments)
    ]

    prompt = f"""
You are aligning two transcripts of the SAME spoken video.
Transcript A is English. Transcript B is Hindi.

Each item has an index and timestamped speech text.
The segmentation differs between languages.

Group consecutive English items and consecutive Hindi items
that express the same content in the same order.

Rules:
- Every English item must be included exactly once.
- Every Hindi item must be included exactly once.
- Never drop an item.
- Groups must follow the original order.
- A group may contain multiple English items and multiple Hindi items.
- Do not pair unrelated sentences merely to make the counts equal.
- If content is missing from one transcript, use an empty list for
  that side and set "uncertain" to true.
- Return valid JSON only, with this structure:
{{
  "groups": [
    {{"en": [0], "hi": [0, 1], "uncertain": false}}
  ]
}}

ENGLISH:
{json.dumps(english_text, ensure_ascii=False)}

HINDI:
{json.dumps(hindi_text, ensure_ascii=False)}
"""

    response = client.chat.completions.create(
        model=ALIGNMENT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. Do not use markdown."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)
    groups = data.get("groups", [])

    if not groups:
        raise RuntimeError("AI could not align the transcripts.")

    # Validate that every segment is accounted for.
    en_indices = []
    hi_indices = []

    for group in groups:
        en_indices.extend(group.get("en", []))
        hi_indices.extend(group.get("hi", []))

    if sorted(en_indices) != list(range(len(english_segments))):
        raise RuntimeError(
            "AI alignment did not include every English segment. "
            "Nothing was exported; please retry."
        )

    if sorted(hi_indices) != list(range(len(hindi_segments))):
        raise RuntimeError(
            "AI alignment did not include every Hindi segment. "
            "Nothing was exported; please retry."
        )

    # Each side must remain in chronological order.
    if en_indices != list(range(len(english_segments))):
        raise RuntimeError("English alignment order was invalid.")

    if hi_indices != list(range(len(hindi_segments))):
        raise RuntimeError("Hindi alignment order was invalid.")

    return groups


# ---------------- SPEED ADJUSTMENT ----------------

def make_atempo(speed):
    """
    FFmpeg atempo changes speed while largely preserving pitch.
    It is not perfect for extreme speed changes.
    """

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
        raise gr.Error("Upload both English and Hindi audio.")

    client = Groq(api_key=API_KEY)

    work_dir = tempfile.mkdtemp(prefix="audiosync_")

    try:
        # 1. Automatic speech recognition and timestamps
        en_segments = transcribe(client, english_audio, "en")
        hi_segments = transcribe(client, hindi_audio, "hi")

        en_duration = get_duration(english_audio)
        hi_duration = get_duration(hindi_audio)

        # 2. Translation-aware grouping
        groups = align_segments(client, en_segments, hi_segments)

        # 3. Build each matched group as a separate audio clip.
        pieces = []
        warnings = []

        for index, group in enumerate(groups):
            en_ids = group.get("en", [])
            hi_ids = group.get("hi", [])

            # A Hindi-only group has no English target timing.
            # Keep it at its approximate original timeline location.
            if not en_ids and hi_ids:
                first = hi_segments[min(hi_ids)]
                target_start = first["start"]
                target_end = first["end"]

            # English-only group: no Hindi speech exists to place.
            # This is reported instead of inventing speech.
            elif en_ids and not hi_ids:
                warnings.append(
                    f"English group {index + 1} has no matching Hindi speech."
                )
                continue

            else:
                target_start = en_segments[min(en_ids)]["start"]
                target_end = en_segments[max(en_ids)]["end"]

            source_start = hi_segments[min(hi_ids)]["start"]
            source_end = hi_segments[max(hi_ids)]["end"]

            target_len = max(0.05, target_end - target_start)
            source_len = max(0.05, source_end - source_start)

            # atempo speed = source duration / target duration
            speed = source_len / target_len

            # Preserve all speech, but warn about extreme speed ratios.
            if speed < 0.65 or speed > 1.55:
                warnings.append(
                    f"Group {index + 1} needs a large speed change "
                    f"({speed:.2f}x). Voice quality may be affected."
                )

            piece_path = os.path.join(
                work_dir,
                f"piece_{index:03d}.wav"
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
                "start": max(0.0, target_start),
                "target_end": max(target_start + 0.05, target_end)
            })

        if not pieces:
            raise RuntimeError(
                "No Hindi segments could be exported. "
                "Check the transcript alignment."
            )

        # 4. Place segments at their English timeline positions.
        # No segment is silently discarded because of speed limits.
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

        mix_label = "mixed"

        filters.append(
            "".join(labels)
            + f"amix=inputs={len(labels)}:duration=longest:"
              f"normalize=0[{mix_label}]"
        )

        output_path = os.path.join(
            work_dir,
            "synchronized_hindi.mp3"
        )

        run_ffmpeg(
            inputs + [
                "-filter_complex", ";".join(filters),
                "-map", f"[{mix_label}]",
                "-t", str(en_duration),
                "-codec:a", "libmp3lame",
                "-b:a", "192k",
                output_path
            ]
        )

        final_duration = get_duration(output_path)

        status = (
            f"Finished. English: {en_duration:.2f}s; "
            f"output: {final_duration:.2f}s. "
            "Please listen to verify alignment."
        )

        if warnings:
            status += "\nWarnings:\n- " + "\n- ".join(warnings[:10])

        return output_path, status

    except gr.Error:
        raise

    except Exception as exc:
        raise gr.Error(str(exc)[:1000])


# ---------------- USER INTERFACE ----------------

with gr.Blocks(title="AudioSync AI") as app:
    gr.Markdown(
        "# AudioSync AI\n"
        "Upload English and Hindi audio. "
        "Transcription and timestamps are generated automatically."
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
        lines=8
    )

    run_button.click(
        synchronize,
        inputs=[english_audio, hindi_audio],
        outputs=[output_audio, status]
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
