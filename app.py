import os
import re
import json
import time
import tempfile
import subprocess
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

# Small batches reduce the chance of truncated responses.
ALIGNMENT_BATCH_SIZE = 4

# Avoid unnatural speed changes.
MIN_SPEED = 0.75
MAX_SPEED = 1.35

# Maximum number of segments per matching window.
MAX_WINDOW_SEGMENTS = 4


# =====================================================
# AUDIO HELPERS
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
            "FFmpeg error: " + result.stderr[-1500:]
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
            "Please use a file smaller than 24 MB."
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

        if start is None or end is None:
            continue

        start = float(start)
        end = float(end)
        text = str(text).strip()

        if end > start and text:
            segments.append({
                "start": start,
                "end": end,
                "text": text
            })

    if not segments:
        raise RuntimeError(
            f"No timestamped speech found for {language}."
        )

    segments.sort(key=lambda x: x["start"])

    return segments


# =====================================================
# AI ALIGNMENT
# =====================================================

def parse_json_response(content):
    if not content:
        raise RuntimeError("AI returned an empty response.")

    content = content.strip()

    content = re.sub(
        r"^```(?:json)?\s*",
        "",
        content
    )

    content = re.sub(r"\s*```$", "", content)

    start = content.find("{")
    end = content.rfind("}")

    if start < 0 or end < start:
        raise RuntimeError(
            "AI did not return a complete JSON response."
        )

    return json.loads(content[start:end + 1])


def match_small_window(client, en_items, hi_items):
    """
    Align two short, chronological transcript windows.

    Indices in the response refer to global segment indices.
    """

    prompt = f"""
Match English transcript segments with Hindi translations
from the same narration.

English and Hindi sentence boundaries may differ.

Return JSON only:
{{
  "groups": [
    {{"en": [0], "hi": [0, 1]}}
  ]
}}

Rules:
- Use only the supplied indices.
- Preserve chronological order.
- Include every supplied index exactly once.
- Do not invent or duplicate indices.
- Match only segments that express the same content.
- A group may contain multiple indices from either language.
- If a segment has no matching content in this window,
  place it in a group with an empty opposite list.
- Do not match unrelated speech just to fill a group.

ENGLISH:
{json.dumps(en_items, ensure_ascii=False)}

HINDI:
{json.dumps(hi_items, ensure_ascii=False)}
"""

    response = client.chat.completions.create(
        model=ALIGNMENT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Return valid JSON only. No markdown."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=2500
    )

    if not response.choices:
        raise RuntimeError("AI returned no alignment choices.")

    choice = response.choices[0]

    if not choice.message.content:
        raise RuntimeError(
            "AI returned empty alignment content. "
            f"Finish reason: {choice.finish_reason}"
        )

    return parse_json_response(choice.message.content).get(
        "groups", []
    )


def validate_window(groups, en_items, hi_items):
    expected_en = [x["i"] for x in en_items]
    expected_hi = [x["i"] for x in hi_items]

    seen_en = []
    seen_hi = []

    for group in groups:
        if not isinstance(group, dict):
            raise RuntimeError("Invalid alignment group.")

        en_ids = group.get("en", [])
        hi_ids = group.get("hi", [])

        if not isinstance(en_ids, list):
            raise RuntimeError("English indices must be a list.")

        if not isinstance(hi_ids, list):
            raise RuntimeError("Hindi indices must be a list.")

        if not en_ids and not hi_ids:
            raise RuntimeError("An alignment group is empty.")

        for index in en_ids:
            if type(index) is not int or index not in expected_en:
                raise RuntimeError("Invalid English segment index.")

        for index in hi_ids:
            if type(index) is not int or index not in expected_hi:
                raise RuntimeError("Invalid Hindi segment index.")

        seen_en.extend(en_ids)
        seen_hi.extend(hi_ids)

    if sorted(seen_en) != sorted(expected_en):
        raise RuntimeError(
            "AI omitted or duplicated English segments in a window."
        )

    if sorted(seen_hi) != sorted(expected_hi):
        raise RuntimeError(
            "AI omitted or duplicated Hindi segments in a window."
        )

    if seen_en != expected_en or seen_hi != expected_hi:
        raise RuntimeError(
            "AI changed transcript order in a window."
        )

    return groups


def align_segments(client, english_segments, hindi_segments):
    """
    Align short windows based on approximate timeline position.

    This avoids asking the model to return one huge mapping.
    Timeline windows are only an initial guide: translations
    can have different pacing, so the result still needs review.
    """

    groups = []

    en_count = len(english_segments)
    hi_count = len(hindi_segments)

    if not en_count or not hi_count:
        raise RuntimeError("One transcript contains no speech.")

    # Use common timeline windows instead of pairing arbitrary
    # groups of four English with four Hindi segments.
    window_seconds = 20.0
    max_time = max(
        english_segments[-1]["end"],
        hindi_segments[-1]["end"]
    )

    window_start = 0.0

    while window_start < max_time:
        window_end = window_start + window_seconds

        en_window = [
            {"i": i, "text": seg["text"]}
            for i, seg in enumerate(english_segments)
            if (
                seg["start"] >= window_start
                and seg["start"] < window_end
            )
        ]

        hi_window = [
            {"i": i, "text": seg["text"]}
            for i, seg in enumerate(hindi_segments)
            if (
                seg["start"] >= window_start
                and seg["start"] < window_end
            )
        ]

        # Include speech that crosses a window boundary.
        if not en_window and not hi_window:
            window_start = window_end
            continue

        # Large windows are subdivided to keep responses short.
        en_chunks = [
            en_window[i:i + ALIGNMENT_BATCH_SIZE]
            for i in range(0, len(en_window), ALIGNMENT_BATCH_SIZE)
        ]

        hi_chunks = [
            hi_window[i:i + ALIGNMENT_BATCH_SIZE]
            for i in range(0, len(hi_window), ALIGNMENT_BATCH_SIZE)
        ]

        # When both sides have speech, use the smallest common
        # chunks. This is approximate, not forced alignment.
        if en_window and hi_window:
            count = max(len(en_chunks), len(hi_chunks))

            for chunk_index in range(count):
                en_items = (
                    en_chunks[chunk_index]
                    if chunk_index < len(en_chunks)
                    else []
                )

                hi_items = (
                    hi_chunks[chunk_index]
                    if chunk_index < len(hi_chunks)
                    else []
                )

                if not en_items or not hi_items:
                    groups.append({
                        "en": [x["i"] for x in en_items],
                        "hi": [x["i"] for x in hi_items]
                    })
                    continue

                matched = match_small_window(
                    client,
                    en_items,
                    hi_items
                )

                matched = validate_window(
                    matched,
                    en_items,
                    hi_items
                )

                groups.extend(matched)

                # Small delay helps avoid rapid repeated requests.
                time.sleep(0.15)

        elif en_window:
            groups.append({
                "en": [x["i"] for x in en_window],
                "hi": []
            })

        else:
            groups.append({
                "en": [],
                "hi": [x["i"] for x in hi_window]
            })

        window_start = window_end

    # Validate global coverage.
    all_en = [
        i
        for group in groups
        for i in group["en"]
    ]

    all_hi = [
        i
        for group in groups
        for i in group["hi"]
    ]

    if sorted(all_en) != list(range(en_count)):
        raise RuntimeError(
            "Final alignment omitted or duplicated English segments."
        )

    if sorted(all_hi) != list(range(hi_count)):
        raise RuntimeError(
            "Final alignment omitted or duplicated Hindi segments."
        )

    return groups


# =====================================================
# AUDIO SYNCHRONIZATION
# =====================================================

def synchronize_audio(
    english_audio,
    hindi_audio,
    progress=gr.Progress()
):
    if not API_KEY:
        raise gr.Error(
            "GROQ_API_KEY is missing in Render Environment."
        )

    if not english_audio or not hindi_audio:
        raise gr.Error("Upload both audio files.")

    client = Groq(api_key=API_KEY)
    work_dir = tempfile.mkdtemp(prefix="audiosync_")

    try:
        progress(0.05, desc="Reading audio durations")

        english_duration = get_duration(english_audio)
        hindi_duration = get_duration(hindi_audio)

        progress(0.15, desc="Transcribing English")

        en_segments = transcribe(
            client,
            english_audio,
            "en"
        )

        progress(0.35, desc="Transcribing Hindi")

        hi_segments = transcribe(
            client,
            hindi_audio,
            "hi"
        )

        progress(0.55, desc="Matching transcript windows")

        groups = align_segments(
            client,
            en_segments,
            hi_segments
        )

        placements = []
        warnings = []

        for group_number, group in enumerate(groups, start=1):
            en_ids = group["en"]
            hi_ids = group["hi"]

            if not hi_ids:
                warnings.append(
                    f"Group {group_number}: no Hindi speech matched."
                )
                continue

            hi_ids = sorted(hi_ids)

            source_start = hi_segments[hi_ids[0]]["start"]
            source_end = hi_segments[hi_ids[-1]]["end"]

            if en_ids:
                en_ids = sorted(en_ids)

                target_start = en_segments[en_ids[0]]["start"]
                target_end = en_segments[en_ids[-1]]["end"]
            else:
                target_start = source_start
                target_end = source_end

                warnings.append(
                    f"Group {group_number}: Hindi-only speech "
                    "kept near its original timestamp."
                )

            source_start = max(0.0, source_start)
            source_end = min(hindi_duration, source_end)

            target_start = max(0.0, target_start)
            target_end = min(english_duration, target_end)

            source_length = source_end - source_start
            target_length = target_end - target_start

            if source_length <= 0 or target_length <= 0:
                warnings.append(
                    f"Group {group_number}: invalid timing."
                )
                continue

            speed = source_length / target_length

            # Preserve natural speed when matching would require
            # extreme time stretching.
            if not MIN_SPEED <= speed <= MAX_SPEED:
                warnings.append(
                    f"Group {group_number}: speed {speed:.2f}x "
                    "was outside the safe range; kept at natural speed."
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
                "start": target_start
            })

        if not placements:
            raise RuntimeError("No Hindi audio segments were placed.")

        progress(0.75, desc="Building the output timeline")

        inputs = []
        filters = []
        labels = []

        for i, piece in enumerate(placements):
            inputs.extend(["-i", piece["path"]])

            delay_ms = max(
                0,
                int(piece["start"] * 1000)
            )

            label = f"a{i}"

            filters.append(
                f"[{i}:a]adelay={delay_ms}|{delay_ms}[{label}]"
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
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[mixed]",
                "-t",
                f"{english_duration:.4f}",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "192k",
                output_path
            ],
            timeout=900
        )

        final_duration = get_duration(output_path)

        status = (
            "Processing completed.\n"
            f"English duration: {english_duration:.2f}s\n"
            f"Hindi duration: {hindi_duration:.2f}s\n"
            f"Output duration: {final_duration:.2f}s\n"
            f"English segments: {len(en_segments)}\n"
            f"Hindi segments: {len(hi_segments)}\n"
            f"Alignment groups: {len(groups)}\n\n"
            "IMPORTANT: This is approximate automatic alignment. "
            "Please review the output against the English audio."
        )

        if warnings:
            status += "\n\nWarnings:\n- " + "\n- ".join(warnings[:12])

        progress(1.0, desc="Finished")

        return output_path, status

    except gr.Error:
        raise

    except Exception as exc:
        raise gr.Error(
            f"Processing failed: {str(exc)[:900]}"
        )


# =====================================================
# USER INTERFACE
# =====================================================

with gr.Blocks(title="AudioSync AI") as app:
    gr.Markdown(
        "# AudioSync AI\n"
        "Upload English and Hindi audio. "
        "Transcripts and timestamps are generated automatically."
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
