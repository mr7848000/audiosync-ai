import os
import re
import json
import tempfile
import subprocess
from pathlib import Path

import gradio as gr
from groq import Groq
import imageio_ffmpeg

TRANSCRIPTION_MODEL = "whisper-large-v3"
ALIGNMENT_MODEL = "openai/gpt-oss-20b"

API_KEY = os.environ.get("GROQ_API_KEY")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


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
    if os.path.getsize(audio_path) > 24 * 1024 * 1024:
        raise RuntimeError(
            "Audio exceeds 24 MB. Please use a smaller file."
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

    return segments


def align_segments(client, english_segments, hindi_segments):
    english_text = [
        {"i": i, "text": seg["text"]}
        for i, seg in enumerate(english_segments)
    ]
    hindi_text = [
        {"i": i, "text": seg["text"]}
        for i, seg in enumerate(hindi_segments)
    ]

    prompt = f"""
Align transcripts of the same video. English and Hindi segments
may be grouped differently. Group consecutive segments with the
same meaning, preserving order. Include every index exactly once.
Return only JSON in this format:
{{"groups":[{{"en":[0],"hi":[0,1]}}]}}

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
                "content": "Return only a valid JSON object. No markdown."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    if not content or not content.strip():
        choice = response.choices[0]
        raise RuntimeError(
            "Alignment model returned empty content. "
            f"Finish reason: {choice.finish_reason}"
        )

    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    start = content.find("{")
    end = content.rfind("}")

    if start < 0 or end < start:
        raise RuntimeError(
            "Alignment model did not return valid JSON."
        )

    try:
        data = json.loads(content[start:end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid alignment JSON: {exc}"
        ) from exc

    groups = data.get("groups", [])

    if not isinstance(groups, list) or not groups:
        raise RuntimeError("No alignment groups returned.")

    en_indices = []
    hi_indices = []

    for group in groups:
        if not isinstance(group, dict):
            raise RuntimeError("Invalid alignment group.")

        en_ids = group.get("en", [])
        hi_ids = group.get("hi", [])

        if not isinstance(en_ids, list) or not isinstance(hi_ids, list):
            raise RuntimeError("Alignment indices must be lists.")

        if not en_ids and not hi_ids:
            raise RuntimeError("An alignment group is empty.")

        en_indices.extend(en_ids)
        hi_indices.extend(hi_ids)

    if sorted(en_indices) != list(range(len(english_segments))):
        raise RuntimeError("Not all English segments were aligned.")

    if sorted(hi_indices) != list(range(len(hindi_segments))):
        raise RuntimeError("Not all Hindi segments were aligned.")

    if (
        en_indices != list(range(len(english_segments)))
        or hi_indices != list(range(len(hindi_segments)))
    ):
        raise RuntimeError("Alignment order is invalid.")

    for group in groups:
        for i in group["en"]:
            if not isinstance(i, int) or not 0 <= i < len(english_segments):
                raise RuntimeError("Invalid English segment index.")

        for i in group["hi"]:
            if not isinstance(i, int) or not 0 <= i < len(hindi_segments):
                raise RuntimeError("Invalid Hindi segment index.")

    return groups


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


def synchronize(english_audio, hindi_audio):
    if not API_KEY:
        raise gr.Error("GROQ_API_KEY missing in Render Environment.")

    if not english_audio or not hindi_audio:
        raise gr.Error("Upload both English and Hindi audio.")

    client = Groq(api_key=API_KEY)
    work_dir = tempfile.mkdtemp(prefix="audiosync_")

    try:
        en_segments = transcribe(client, english_audio, "en")
        hi_segments = transcribe(client, hindi_audio, "hi")

        en_duration = get_duration(english_audio)

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
                target_start = source_start
                target_end = source_end

            source_len = source_end - source_start
            target_len = target_end - target_start

            if source_len <= 0 or target_len <= 0:
                warnings.append(f"Group {index + 1}: invalid timing.")
                continue

            speed = source_len / target_len

            if speed < 0.5 or speed > 2.0:
                warnings.append(
                    f"Group {index + 1}: extreme speed required "
                    f"({speed:.2f}x); kept at original speed."
                )
                speed = 1.0

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
                "start": max(0.0, target_start)
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
            f"Completed. English: {en_duration:.2f}s; "
            f"output: {final_duration:.2f}s."
        )

        if warnings:
            status += "\n\nWarnings:\n- " + "\n- ".join(warnings[:12])

        return output_path, status

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Processing failed: {str(exc)[:900]}")


with gr.Blocks(title="AudioSync AI") as app:
    gr.Markdown(
        "# AudioSync AI\n"
        "Upload English and Hindi audio. No scripts required."
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

    run_button = gr.Button("Synchronize Audio", variant="primary")
    output_audio = gr.Audio(
        label="Synchronized Hindi MP3",
        type="filepath"
    )
    status = gr.Textbox(label="Processing Status", lines=8)

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
