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
ALIGNMENT_MODEL = "openai/gpt-oss-20b"

API_KEY = os.environ.get("GROQ_API_KEY")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

MAX_FILE_MB = 24
BATCH_SIZE = 8


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
            f"No timestamped speech returned for language '{language}'."
        )

    # Keep segments chronological.
    segments.sort(key=lambda x: (x["start"], x["end"]))

    return segments


# ---------------- ALIGNMENT HELPERS ----------------

def parse_json_response(content):
    if not content or not content.strip():
        raise RuntimeError("Alignment model returned empty content.")

    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    start = content.find("{")
    end = content.rfind("}")

    if start < 0 or end < start:
        raise RuntimeError(
            "Alignment model did not return a complete JSON object."
        )

    try:
        return json.loads(content[start:end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Alignment JSON is incomplete or invalid: {exc}"
        ) from exc


def align_one_batch(client, english_batch, hindi_batch, batch_number):
    en_text = [
        {"i": i, "text": seg["text"]}
        for i, seg in enumerate(english_batch)
    ]

    hi_text = [
        {"i": i, "text": seg["text"]}
        for i, seg in enumerate(hindi_batch)
    ]

    prompt = f"""
Align these English and Hindi transcript segments from the same video.

The two languages may divide sentences differently.

Rules:
1. Group consecutive segments that express the same content.
2. Preserve the original order.
3. Include every supplied segment index exactly once.
4. Never invent an index.
5. If speech has no equivalent in the other language, use an empty
   list on that side.
6. Do not group unrelated speech merely to make counts match.

Return compact JSON only, in this format:
{{"groups":[{{"en":[0],"hi":[0,1]}}]}}

ENGLISH SEGMENTS:
{json.dumps(en_text, ensure_ascii=False)}

HINDI SEGMENTS:
{json.dumps(hi_text, ensure_ascii=False)}
"""

    try:
        response = client.chat.completions.create(
            model=ALIGNMENT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only a compact, valid JSON object. "
                        "No markdown or explanation. "
                        "Include every supplied index exactly once."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_completion_tokens=3000
        )
    except Exception as exc:
        raise RuntimeError(
            f"Alignment batch {batch_number} API request failed: "
            f"{str(exc)[:500]}"
        ) from exc

    choice = response.choices[0]
    content = choice.message.content

    if not content or not content.strip():
        raise RuntimeError(
            f"Alignment batch {batch_number} returned empty content. "
            f"Finish reason: {choice.finish_reason}. "
            "The model may have reached its output limit."
        )

    data = parse_json_response(content)
    groups = data.get("groups")

    if not isinstance(groups, list) or not groups:
        raise RuntimeError(
            f"Alignment batch {batch_number} returned no groups."
        )

    en_seen = []
    hi_seen = []
    validated_groups = []

    for group in groups:
        if not isinstance(group, dict):
            raise RuntimeError(
                f"Invalid group in alignment batch {batch_number}."
            )

        en_ids = group.get("en", [])
        hi_ids = group.get("hi", [])

        if not isinstance(en_ids, list) or not isinstance(hi_ids, list):
            raise RuntimeError("Alignment indices must be lists.")

        if not en_ids and not hi_ids:
            raise RuntimeError("An alignment group is empty.")

        for idx in en_ids:
            if type(idx) is not int or not 0 <= idx < len(english_batch):
                raise RuntimeError(
                    f"Invalid English index in batch {batch_number}."
                )

        for idx in hi_ids:
            if type(idx) is not int or not 0 <= idx < len(hindi_batch):
                raise RuntimeError(
                    f"Invalid Hindi index in batch {batch_number}."
                )

        en_seen.extend(en_ids)
        hi_seen.extend(hi_ids)

        validated_groups.append({
            "en": en_ids,
            "hi": hi_ids
        })

    if sorted(en_seen) != list(range(len(english_batch))):
        raise RuntimeError(
            f"Batch {batch_number} omitted or duplicated English segments."
        )

    if sorted(hi_seen) != list(range(len(hindi_batch))):
        raise RuntimeError(
            f"Batch {batch_number} omitted or duplicated Hindi segments."
        )

    if en_seen != list(range(len(english_batch))):
        raise RuntimeError(
            f"English order is invalid in batch {batch_number}."
        )

    if hi_seen != list(range(len(hindi_batch))):
        raise RuntimeError(
            f"Hindi order is invalid in batch {batch_number}."
        )

    return validated_groups


def split_into_batches(items, batch_count):
    if batch_count <= 1:
        return [items]

    result = []
    total = len(items)

    for i in range(batch_count):
        start = round(i * total / batch_count)
        end = round((i + 1) * total / batch_count)
        result.append(items[start:end])

    return result


def align_segments(client, english_segments, hindi_segments):
    """
    Align short transcript batches and translate local indices
    back to global transcript indices.

    Important: corresponding batches are approximated by their
    relative positions. This is not a full forced aligner.
    """

    if not english_segments or not hindi_segments:
        raise RuntimeError("One transcript contains no speech segments.")

    batch_count = max(
        1,
        (max(len(english_segments), len(hindi_segments))
         + BATCH_SIZE - 1) // BATCH_SIZE
    )

    en_batches = split_into_batches(english_segments, batch_count)
    hi_batches = split_into_batches(hindi_segments, batch_count)

    all_groups = []
    en_offset = 0
    hi_offset = 0

    for batch_index, (en_batch, hi_batch) in enumerate(
        zip(en_batches, hi_batches),
        start=1
    ):
        if not en_batch and not hi_batch:
            continue

        # The model can identify speech missing from one side.
        local_groups = align_one_batch(
            client,
            en_batch,
            hi_batch,
            batch_index
        )

        for group in local_groups:
            global_en = [
                en_offset + idx for idx in group["en"]
            ]
            global_hi = [
                hi_offset + idx for idx in group["hi"]
            ]

            all_groups.append({
                "en": global_en,
                "hi": global_hi
            })

        en_offset += len(en_batch)
        hi_offset += len(hi_batch)

    # Final validation: every segment appears once.
    all_en = [
        idx for group in all_groups for idx in group["en"]
    ]
    all_hi = [
        idx for group in all_groups for idx in group["hi"]
    ]

    if sorted(all_en) != list(range(len(english_segments))):
        raise RuntimeError(
            "Final alignment omitted or duplicated English segments."
        )

    if sorted(all_hi) != list(range(len(hindi_segments))):
        raise RuntimeError(
            "Final alignment omitted or duplicated Hindi segments."
        )

    if all_en != list(range(len(english_segments))):
        raise RuntimeError("Final English segment order is invalid.")

    if all_hi != list(range(len(hindi_segments))):
        raise RuntimeError("Final Hindi segment order is invalid.")

    return all_groups


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

        groups = align_segments(
            client,
            en_segments,
            hi_segments
        )

        pieces = []
        warnings = []

        for index, group in enumerate(groups):
            en_ids = group["en"]
            hi_ids = group["hi"]

            if not hi_ids:
                warnings.append(
                    f"Group {index + 1}: no matching Hindi speech."
                )
                continue

            source_start = hi_segments[hi_ids[0]]["start"]
            source_end = hi_segments[hi_ids[-1]]["end"]

            if en_ids:
                target_start = en_segments[en_ids[0]]["start"]
                target_end = en_segments[en_ids[-1]]["end"]
            else:
                # Hindi-only material has no reliable English timing.
                target_start = source_start
                target_end = source_end

                warnings.append(
                    f"Group {index + 1}: Hindi-only speech "
                    "kept near its original timestamp."
                )

            source_start = max(0.0, source_start)
            source_end = min(hi_duration, source_end)

            target_start = max(0.0, target_start)
            target_end = min(en_duration, target_end)

            source_len = source_end - source_start
            target_len = target_end - target_start

            if source_len <= 0 or target_len <= 0:
                warnings.append(
                    f"Group {index + 1}: invalid timing; omitted."
                )
                continue

            speed = source_len / target_len

            # Do not force extreme time stretching.
            if speed < 0.65 or speed > 1.55:
                warnings.append(
                    f"Group {index + 1}: required speed "
                    f"{speed:.2f}x is outside the safe range; "
                    "kept at original speed."
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
            raise RuntimeError(
                "No usable Hindi segments were created. "
                "No output was exported."
            )

        # Place each adjusted Hindi clip on the English timeline.
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
            f"Hindi source duration: {hi_duration:.2f}s\n"
            f"Output duration: {final_duration:.2f}s\n"
            f"English transcript segments: {len(en_segments)}\n"
            f"Hindi transcript segments: {len(hi_segments)}\n"
            f"Matched groups: {len(groups)}\n"
            "Please listen to the output to verify alignment."
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
        "No scripts need to be pasted."
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
