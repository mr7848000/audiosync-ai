import os
import re
import json
import tempfile
import subprocess
import shutil
from pathlib import Path

import gradio as gr
from groq import Groq
import imageio_ffmpeg

# =====================================================
# CONFIGURATION
# =====================================================

TRANSCRIPTION_MODEL = "whisper-large-v3"
ALIGNMENT_MODEL = "openai/gpt-oss-20b"

API_KEY = os.environ.get("GROQ_API_KEY")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

MAX_FILE_MB = 24
ALIGNMENT_BATCH_SIZE = 4

MIN_SPEED = 0.75
MAX_SPEED = 1.35


# =====================================================
# FFMPEG HELPERS
# =====================================================

def run_ffmpeg(args, timeout=600):
    result = subprocess.run(
        [FFMPEG, "-y"] + args,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg error: " + result.stderr[-1600:]
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


def make_atempo(speed):
    if speed <= 0:
        raise ValueError("Invalid speed ratio.")

    filters = []

    while speed > 2:
        filters.append("atempo=2.0")
        speed /= 2

    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5

    filters.append(f"atempo={speed:.6f}")

    return ",".join(filters)


# =====================================================
# TRANSCRIPTION
# =====================================================

def transcribe(client, audio_path, language):
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    if size_mb > MAX_FILE_MB:
        raise RuntimeError(
            f"{Path(audio_path).name} is {size_mb:.1f} MB. "
            "Please use an audio file smaller than 24 MB."
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

    for item in getattr(result, "segments", None) or []:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            text = item.get("text", "")
        else:
            start = item.start
            end = item.end
            text = item.text

        if start is None or end is None:
            continue

        start = float(start)
        end = float(end)
        text = str(text).strip()

        if text and end > start:
            segments.append({
                "start": start,
                "end": end,
                "text": text
            })

    if not segments:
        raise RuntimeError(
            f"No timestamped speech found for language: {language}"
        )

    return segments


# =====================================================
# BATCHED AI ALIGNMENT
# =====================================================

def ask_alignment_model(client, english_batch, hindi_batch):
    """
    Align two small transcript batches to avoid oversized
    model responses.

    Each batch contains the original global segment indices.
    """

    prompt = f"""
You are a professional bilingual audio alignment assistant.

Two lists contain timestamped speech from the SAME video.
List EN is English. List HI is Hindi.

The Hindi is a translation or narration of the English.
The segmentation may differ between languages.

Match consecutive items that express the same meaning.
Preserve the order of speech.

Rules:
- Use only indices supplied in the lists.
- Every item must appear exactly once in the groups.
- Never invent or repeat an index.
- One group can contain multiple EN and HI items.
- If speech exists only on one side, use an empty list
  for the other side.
- Do not match unrelated sentences just to complete a group.
- Return JSON only.

Required format:
{{
  "groups": [
    {{"en": [0, 1], "hi": [0]}}
  ]
}}

EN:
{json.dumps(english_batch, ensure_ascii=False)}

HI:
{json.dumps(hindi_batch, ensure_ascii=False)}
"""

    response = client.chat.completions.create(
        model=ALIGNMENT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return a valid JSON object only. "
                    "No markdown or explanation."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=4000
    )

    if not response.choices:
        raise RuntimeError("Alignment model returned no choices.")

    choice = response.choices[0]
    content = choice.message.content

    if not content or not content.strip():
        raise RuntimeError(
            "Alignment model returned empty content. "
            f"Finish reason: {choice.finish_reason}. "
            "Try a shorter audio file or retry later."
        )

    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    first = content.find("{")
    last = content.rfind("}")

    if first < 0 or last < first:
        raise RuntimeError(
            "AI did not return a complete JSON response. "
            f"Finish reason: {choice.finish_reason}"
        )

    data = json.loads(content[first:last + 1])
    groups = data.get("groups")

    if not isinstance(groups, list) or not groups:
        raise RuntimeError("AI returned no alignment groups.")

    return groups


def validate_batch(groups, english_batch, hindi_batch):
    expected_en = [x["i"] for x in english_batch]
    expected_hi = [x["i"] for x in hindi_batch]

    seen_en = []
    seen_hi = []

    for group in groups:
        if not isinstance(group, dict):
            raise RuntimeError("Invalid alignment group.")

        en_ids = group.get("en", [])
        hi_ids = group.get("hi", [])

        if not isinstance(en_ids, list) or not isinstance(hi_ids, list):
            raise RuntimeError("Alignment indices must be lists.")

        if not en_ids and not hi_ids:
            raise RuntimeError("Empty alignment group returned.")

        for index in en_ids:
            if type(index) is not int or index not in expected_en:
                raise RuntimeError("Invalid English alignment index.")

        for index in hi_ids:
            if type(index) is not int or index not in expected_hi:
                raise RuntimeError("Invalid Hindi alignment index.")

        seen_en.extend(en_ids)
        seen_hi.extend(hi_ids)

    if sorted(seen_en) != sorted(expected_en):
        raise RuntimeError(
            "AI omitted or duplicated English transcript segments."
        )

    if sorted(seen_hi) != sorted(expected_hi):
        raise RuntimeError(
            "AI omitted or duplicated Hindi transcript segments."
        )

    # Each list must preserve chronological order.
    if seen_en != expected_en:
        raise RuntimeError("English segment order is invalid.")

    if seen_hi != expected_hi:
        raise RuntimeError("Hindi segment order is invalid.")

    return groups


def align_segments(client, english_segments, hindi_segments):
    """
    Process small batches to avoid the model's output-length limit.

    Batches are created from chronological transcript sections.
    Each batch is independently validated before being accepted.
    """

    result_groups = []

    en_cursor = 0
    hi_cursor = 0

    while (
        en_cursor < len(english_segments)
        or hi_cursor < len(hindi_segments)
    ):
        en_batch = english_segments[
            en_cursor:en_cursor + ALIGNMENT_BATCH_SIZE
        ]

        hi_batch = hindi_segments[
            hi_cursor:hi_cursor + ALIGNMENT_BATCH_SIZE
        ]

        en_items = [
            {"i": en_cursor + j, "text": seg["text"]}
            for j, seg in enumerate(en_batch)
        ]

        hi_items = [
            {"i": hi_cursor + j, "text": seg["text"]}
            for j, seg in enumerate(hi_batch)
        ]

        if not en_items or not hi_items:
            # Remaining speech exists on only one side.
            if en_items:
                result_groups.append({
                    "en": [x["i"] for x in en_items],
                    "hi": []
                })

            if hi_items:
                result_groups.append({
                    "en": [],
                    "hi": [x["i"] for x in hi_items]
                })

            en_cursor += len(en_batch)
            hi_cursor += len(hi_batch)
            continue

        groups = ask_alignment_model(
            client,
            en_items,
            hi_items
        )

        validate_batch(groups, en_items, hi_items)

        result_groups.extend(groups)

        en_cursor += len(en_batch)
        hi_cursor += len(hi_batch)

    return result_groups


# =====================================================
# AUDIO TIMELINE
# =====================================================

def synchronize_audio(
    english_audio,
    hindi_audio,
    progress=gr.Progress()
):
    if not API_KEY:
        raise gr.Error(
            "GROQ_API_KEY is missing. Add it in Render Environment."
        )

    if not english_audio or not hindi_audio:
        raise gr.Error("Upload both English and Hindi audio.")

    client = Groq(api_key=API_KEY)
    work_dir = tempfile.mkdtemp(prefix="audiosync_")

    try:
        progress(0.05, desc="Reading audio durations")

        english_duration = get_duration(english_audio)
        hindi_duration = get_duration(hindi_audio)

        progress(0.12, desc="Transcribing English audio")

        english_segments = transcribe(
            client, english_audio, "en"
        )

        progress(0.32, desc="Transcribing Hindi audio")

        hindi_segments = transcribe(
            client, hindi_audio, "hi"
        )

        progress(0.52, desc="Matching transcript sections")

        groups = align_segments(
            client,
            english_segments,
            hindi_segments
        )

        # Map each matched Hindi group to the corresponding
        # English timeline interval.
        placements = []
        warnings = []

        for group_number, group in enumerate(groups, start=1):
            en_ids = group["en"]
            hi_ids = group["hi"]

            if not hi_ids:
                warnings.append(
                    f"Group {group_number}: English speech has "
                    "no matched Hindi speech."
                )
                continue

            hi_ids = sorted(hi_ids)

            source_start = hindi_segments[hi_ids[0]]["start"]
            source_end = hindi_segments[hi_ids[-1]]["end"]

            if en_ids:
                en_ids = sorted(en_ids)

                target_start = english_segments[en_ids[0]]["start"]
                target_end = english_segments[en_ids[-1]]["end"]
            else:
                # No English target exists, so retain the
                # approximate Hindi timestamp instead.
                target_start = source_start
                target_end = source_end

                warnings.append(
                    f"Group {group_number}: Hindi-only speech "
                    "retained near its original timestamp."
                )

            target_start = max(0.0, target_start)
            target_end = min(english_duration, target_end)
            source_start = max(0.0, source_start)
            source_end = min(hindi_duration, source_end)

            source_length = source_end - source_start
            target_length = target_end - target_start

            if source_length <= 0 or target_length <= 0:
                warnings.append(
                    f"Group {group_number}: invalid timing."
                )
                continue

            speed = source_length / target_length

            # Never force a large speed change. Place the speech
            # at the target start and preserve natural voice speed.
            if speed < MIN_SPEED or speed > MAX_SPEED:
                warnings.append(
                    f"Group {group_number}: required speed "
                    f"{speed:.2f}x exceeds the natural range; "
                    "speech kept at original speed."
                )
                speed = 1.0

            piece_path = os.path.join(
                work_dir,
                f"piece_{group_number:04d}.wav"
            )

            run_ffmpeg([
                "-ss", f"{source_start:.4f}",
                "-t", f"{source_length:.4f}",
                "-i", hindi_audio,
                "-af", make_atempo(speed),
                "-ar", "44100",
                "-ac", "2",
                piece_path
            ])

            placements.append({
                "path": piece_path,
                "start": target_start,
                "end": target_end
            })

        if not placements:
            raise RuntimeError("No Hindi audio segments could be placed.")

        progress(0.72, desc="Building synchronized timeline")

        # Mix clips at their target timestamps. Silence is preserved
        # between clips; clips are not simply concatenated together.
        inputs = []
        filters = []
        labels = []

        for i, piece in enumerate(placements):
            inputs.extend(["-i", piece["path"]])

            delay = max(0, int(piece["start"] * 1000))
            label = f"a{i}"

            filters.append(
                f"[{i}:a]adelay={delay}|{delay}[{label}]"
            )
            labels.append(f"[{label}]")

        filters.append(
            "".join(labels)
            + f"amix=inputs={len(labels)}:"
              "duration=longest:normalize=0[mixed]"
        )

        output_path = os.path.join(
            work_dir,
            "synchronized_hindi.mp3"
        )

        run_ffmpeg(
            inputs + [
                "-filter_complex", ";".join(filters),
                "-map", "[mixed]",
                "-t", f"{english_duration:.4f}",
                "-codec:a", "libmp3lame",
                "-b:a", "192k",
                output_path
            ],
            timeout=900
        )

        progress(0.95, desc="Checking exported audio")

        final_duration = get_duration(output_path)

        status = (
            "Processing completed.\n"
            f"English duration: {english_duration:.2f}s\n"
            f"Hindi duration: {hindi_duration:.2f}s\n"
            f"Output duration: {final_duration:.2f}s\n"
            f"English speech segments: {len(english_segments)}\n"
            f"Hindi speech segments: {len(hindi_segments)}\n"
            f"Alignment groups: {len(groups)}\n\n"
            "IMPORTANT: Transcript matching is AI-based and approximate. "
            "Please listen to verify sentence alignment."
        )

        if warnings:
            status += "\n\nWarnings:\n- " + "\n- ".join(warnings[:15])

        progress(1.0, desc="Finished")

        return output_path, status

    except gr.Error:
        raise

    except Exception as exc:
        raise gr.Error(
            f"Processing failed: {str(exc)[:1000]}"
        )

    finally:
        # Keep the output directory alive so Gradio can serve
        # the generated file. Render may remove it on restart.
        pass


# =====================================================
# USER INTERFACE
# =====================================================

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
        lines=12
    )

    run_button.click(
        fn=synchronize_audio,
        inputs=[english_audio, hindi_audio],
        outputs=[output_audio, status]
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
