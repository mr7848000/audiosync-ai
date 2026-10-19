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
BATCH_SIZE = 8

MIN_SPEED = 0.80
MAX_SPEED = 1.25


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
            "FFmpeg failed: " + result.stderr[-1200:]
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


# =====================================================
# TRANSCRIPTION
# =====================================================

def transcribe(client, audio_path, language):
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    if size_mb > MAX_FILE_MB:
        raise RuntimeError(
            f"{Path(audio_path).name} is {size_mb:.1f} MB. "
            "Please upload an MP3 smaller than 24 MB."
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
            f"No timestamped speech found for {language}."
        )

    return segments


# =====================================================
# ALIGNMENT
# =====================================================

def ask_alignment(client, english_items, hindi_items):
    """
    Align short chronological batches.
    The model returns local batch positions, not global indices.
    """

    prompt = f"""
Match the English and Hindi speech segments from the same video.

The Hindi is a translation of the English.
Both lists are in chronological order.
Sentence boundaries can differ.

For each group:
- Match speech with the same meaning.
- Preserve chronological order.
- Every item must be used exactly once.
- Do not invent or repeat indices.
- A group may contain multiple items from either list.
- If a segment has no counterpart, use an empty list
  on the other side.

Use ONLY the local integer positions shown below.

Return JSON only in this format:
{{"groups":[{{"en":[0],"hi":[0,1]}}]}}

ENGLISH:
{json.dumps(english_items, ensure_ascii=False)}

HINDI:
{json.dumps(hindi_items, ensure_ascii=False)}
"""

    response = client.chat.completions.create(
        model=ALIGNMENT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return only a compact valid JSON object. "
                    "No markdown or explanations."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1800
    )

    if not response.choices:
        raise RuntimeError("Alignment model returned no choices.")

    choice = response.choices[0]
    content = choice.message.content

    if not content or not content.strip():
        raise RuntimeError(
            "Alignment model returned empty content. "
            f"Finish reason: {choice.finish_reason}. "
            "Please retry."
        )

    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    start = content.find("{")
    end = content.rfind("}")

    if start < 0 or end < start:
        raise RuntimeError(
            "Alignment response was incomplete. "
            f"Finish reason: {choice.finish_reason}. "
            "Try again."
        )

    data = json.loads(content[start:end + 1])
    groups = data.get("groups")

    if not isinstance(groups, list) or not groups:
        raise RuntimeError("No alignment groups were returned.")

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
            raise RuntimeError("An empty alignment group was returned.")

        for index in en_ids:
            if type(index) is not int or not 0 <= index < len(english_items):
                raise RuntimeError("Invalid English segment index.")

        for index in hi_ids:
            if type(index) is not int or not 0 <= index < len(hindi_items):
                raise RuntimeError("Invalid Hindi segment index.")

        seen_en.extend(en_ids)
        seen_hi.extend(hi_ids)

    if sorted(seen_en) != list(range(len(english_items))):
        raise RuntimeError(
            "AI omitted or duplicated English segments in a batch."
        )

    if sorted(seen_hi) != list(range(len(hindi_items))):
        raise RuntimeError(
            "AI omitted or duplicated Hindi segments in a batch."
        )

    if seen_en != list(range(len(english_items))):
        raise RuntimeError("English segment order is invalid.")

    if seen_hi != list(range(len(hindi_items))):
        raise RuntimeError("Hindi segment order is invalid.")

    return groups


def align_segments(client, english_segments, hindi_segments):
    """
    Match short chronological windows.
    Windows are chosen by approximate relative timeline position,
    rather than blindly pairing equal numbers of segments.
    """

    groups = []

    en_count = len(english_segments)
    hi_count = len(hindi_segments)

    en_cursor = 0
    hi_cursor = 0

    while en_cursor < en_count or hi_cursor < hi_count:

        if en_cursor >= en_count:
            groups.append({
                "en": [],
                "hi": list(range(hi_cursor, hi_count))
            })
            break

        if hi_cursor >= hi_count:
            groups.append({
                "en": list(range(en_cursor, en_count)),
                "hi": []
            })
            break

        # Keep batches small so the model has room to respond.
        en_end = min(en_cursor + BATCH_SIZE, en_count)

        en_batch = english_segments[en_cursor:en_end]

        # Estimate the matching Hindi time window from the
        # relative position of the English batch.
        en_start_time = en_batch[0]["start"]
        en_end_time = en_batch[-1]["end"]

        # Approximate expected Hindi position using relative time.
        # This is only a windowing heuristic, not exact alignment.
        en_total_duration = max(
            0.1,
            english_segments[-1]["end"]
        )

        hi_total_duration = max(
            0.1,
            hindi_segments[-1]["end"]
        )

        expected_hi_start_time = (
            en_start_time / en_total_duration
        ) * hi_total_duration

        expected_hi_end_time = (
            en_end_time / en_total_duration
        ) * hi_total_duration

        # Include context on both sides to tolerate translation
        # timing differences and different sentence segmentation.
        lower_time = max(
            0.0,
            expected_hi_start_time - 12.0
        )

        upper_time = expected_hi_end_time + 18.0

        candidates = [
            i for i in range(hi_cursor, hi_count)
            if hindi_segments[i]["end"] >= lower_time
            and hindi_segments[i]["start"] <= upper_time
        ]

        if not candidates:
            candidates = list(
                range(
                    hi_cursor,
                    min(hi_cursor + BATCH_SIZE, hi_count)
                )
            )

        hi_start = candidates[0]
        hi_end = min(candidates[-1] + 1, hi_count)

        # Limit each batch to avoid excessively long model responses.
        hi_end = min(hi_end, hi_start + BATCH_SIZE)

        hi_batch = hindi_segments[hi_start:hi_end]

        en_items = [
            {"i": i, "text": english_segments[i]["text"]}
            for i in range(en_cursor, en_end)
        ]

        hi_items = [
            {"i": i, "text": hindi_segments[i]["text"]}
            for i in range(hi_start, hi_end)
        ]

        raw_groups = ask_alignment(
            client,
            [
                {"i": j, "text": item["text"]}
                for j, item in enumerate(en_items)
            ],
            [
                {"i": j, "text": item["text"]}
                for j, item in enumerate(hi_items)
            ]
        )

        # Convert local positions back to global segment indices.
        for group in raw_groups:
            global_en = [
                en_cursor + i for i in group["en"]
            ]

            global_hi = [
                hi_start + i for i in group["hi"]
            ]

            groups.append({
                "en": global_en,
                "hi": global_hi
            })

        en_cursor = en_end
        hi_cursor = hi_end

    return groups


# =====================================================
# SPEED ADJUSTMENT
# =====================================================

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


# =====================================================
# SYNCHRONIZATION
# =====================================================

def synchronize_audio(english_audio, hindi_audio, progress=gr.Progress()):
    if not API_KEY:
        raise gr.Error(
            "GROQ_API_KEY is missing. Add it in Render Environment."
        )

    if not english_audio or not hindi_audio:
        raise gr.Error("Upload both English and Hindi audio files.")

    client = Groq(api_key=API_KEY)
    work_dir = tempfile.mkdtemp(prefix="audiosync_")

    try:
        progress(0.05, desc="Reading audio durations")

        english_duration = get_duration(english_audio)
        hindi_duration = get_duration(hindi_audio)

        progress(0.12, desc="Transcribing English")

        english_segments = transcribe(
            client, english_audio, "en"
        )

        progress(0.32, desc="Transcribing Hindi")

        hindi_segments = transcribe(
            client, hindi_audio, "hi"
        )

        progress(0.52, desc="Matching transcript batches")

        groups = align_segments(
            client,
            english_segments,
            hindi_segments
        )

        placements = []
        warnings = []

        for number, group in enumerate(groups, start=1):
            en_ids = group["en"]
            hi_ids = group["hi"]

            if not hi_ids:
                warnings.append(
                    f"Group {number}: no matching Hindi speech."
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
                # No corresponding English segment.
                target_start = source_start
                target_end = source_end

                warnings.append(
                    f"Group {number}: Hindi-only speech retained "
                    "near its original timestamp."
                )

            source_start = max(0.0, source_start)
            source_end = min(hindi_duration, source_end)

            target_start = max(0.0, target_start)
            target_end = min(english_duration, target_end)

            source_length = source_end - source_start
            target_length = target_end - target_start

            if source_length <= 0 or target_length <= 0:
                warnings.append(
                    f"Group {number}: invalid timing."
                )
                continue

            speed = source_length / target_length

            # Keep speed changes within a limited range.
            if speed < MIN_SPEED or speed > MAX_SPEED:
                warnings.append(
                    f"Group {number}: required speed {speed:.2f}x "
                    "is outside the natural range; kept original speed."
                )
                speed = 1.0

            piece_path = os.path.join(
                work_dir,
                f"piece_{number:04d}.wav"
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
            raise RuntimeError("No Hindi audio segments could be placed.")

        progress(0.75, desc="Building timeline")

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
                "-filter_complex", ";".join(filters),
                "-map", "[mixed]",
                "-t", f"{english_duration:.4f}",
                "-codec:a", "libmp3lame",
                "-b:a", "192k",
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
            f"English segments: {len(english_segments)}\n"
            f"Hindi segments: {len(hindi_segments)}\n"
            f"Alignment groups: {len(groups)}\n\n"
            "IMPORTANT: Matching is approximate. Listen to verify "
            "that the translated sentences align correctly."
        )

        if warnings:
            status += (
                "\n\nWarnings:\n- "
                + "\n- ".join(warnings[:12])
            )

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
