import os
import re
import time
import tempfile
import subprocess
import requests

from pathlib import Path

import gradio as gr
import imageio_ffmpeg
from groq import Groq


# =====================================================
# CONFIGURATION
# =====================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
AI33_API_KEY = os.environ.get("AI33_API_KEY")

GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3"

# Updated translation model
GROQ_TRANSLATION_MODEL = "openai/gpt-oss-120b"

AI33_TTS_URL = "https://api.ai33.pro/v3/text-to-speech"
AI33_TASK_URL = "https://api.ai33.pro/v1/task"

VOICE_ID = "minimax_2618345"
VOICE_MODEL = "speech-02-hd"

MAX_FILE_MB = 24

MIN_SPEED = 1.0
MAX_SPEED = 1.5

TRANSLATION_MAX_ATTEMPTS = 3

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


TRANSLATION_INSTRUCTIONS = """
Translate the script into a natural mix of Hindi and English.

Keep English words in English and write Hindi words in Devanagari.
Use conversational Hindi, not a traditional or literal translation.

Additional rules:
- Do not skip, summarize, or shorten any sentence.
- Preserve the original meaning and order.
- Keep English technical terms in English.
- Do not add headings, bullet points, explanations, or labels.
- Return only the translated text.
- Do not add quotation marks.
"""


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
            "FFmpeg error: " + result.stderr[-1500:]
        )


def get_duration(path):
    result = subprocess.run(
        [FFMPEG, "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        timeout=120
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        result.stderr
    )

    if not match:
        raise RuntimeError(
            f"Could not read audio duration: {path}"
        )

    hours, minutes, seconds = match.groups()

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + float(seconds)
    )


def prepare_audio(input_path, output_path):
    run_ffmpeg([
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-codec:a", "libmp3lame",
        "-b:a", "64k",
        str(output_path)
    ])

    return str(output_path)


def make_atempo(speed):
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
# TRANSCRIPTION
# =====================================================

def transcribe_english(client, audio_path):
    size_mb = (
        os.path.getsize(audio_path) / (1024 * 1024)
    )

    if size_mb > MAX_FILE_MB:
        raise RuntimeError(
            f"Audio file is {size_mb:.1f} MB. "
            "Please upload a smaller MP3 under 24 MB."
        )

    with open(audio_path, "rb") as audio:
        result = client.audio.transcriptions.create(
            file=(Path(audio_path).name, audio),
            model=GROQ_TRANSCRIPTION_MODEL,
            language="en",
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

        text = str(text).strip()
        start = float(start)
        end = float(end)

        if text and end > start:
            segments.append({
                "start": start,
                "end": end,
                "text": text
            })

    if not segments:
        raise RuntimeError(
            "No timestamped English speech was detected."
        )

    return segments


# =====================================================
# SENTENCE GROUPING
# =====================================================

def group_sentences(segments):
    groups = []
    current = None

    for segment in segments:
        if current is None:
            current = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            }
        else:
            current["end"] = segment["end"]
            current["text"] += " " + segment["text"].strip()

        sentence_end = bool(
            re.search(
                r"""[.!?]["')\]]*$""",
                current["text"]
            )
        )

        too_long = len(current["text"]) >= 180

        if sentence_end or too_long:
            groups.append(current)
            current = None

    if current is not None:
        groups.append(current)

    return [
        item for item in groups
        if item["text"].strip()
        and item["end"] > item["start"]
    ]


# =====================================================
# TRANSLATION
# =====================================================

def clean_translation(text):
    text = text.strip()

    text = re.sub(
        r"^```[^\n]*\n?",
        "",
        text
    )

    text = re.sub(
        r"\n?```$",
        "",
        text
    )

    text = re.sub(
        r"^(Translation|Hindi translation|हिंदी अनुवाद)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip().strip('"“”')


def translate_one_sentence(client, sentence, sentence_number):
    prompt = f"""
{TRANSLATION_INSTRUCTIONS}

Translate this English sentence now.
Return only the translated text.

English sentence:
{sentence}
"""

    last_error = "Unknown translation error"

    for attempt in range(
        1,
        TRANSLATION_MAX_ATTEMPTS + 1
    ):
        try:
            response = client.chat.completions.create(
                model=GROQ_TRANSLATION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You translate English into natural "
                            "conversational Hindi-English mix. "
                            "Follow the instructions exactly. "
                            "Return plain translated text only."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_completion_tokens=2048,
                reasoning_effort="low",
                include_reasoning=False
            )

            if not response.choices:
                raise RuntimeError(
                    "Translation model returned no choices."
                )

            choice = response.choices[0]
            content = choice.message.content

            if not content or not content.strip():
                raise RuntimeError(
                    "Translation response was empty. "
                    f"Finish reason: {choice.finish_reason}. "
                    "Check the Render logs for details."
                )

            translated = clean_translation(content)

            if not translated:
                raise RuntimeError(
                    "Translation was empty after cleanup."
                )

            return translated

        except Exception as exc:
            last_error = str(exc)

            print(
                f"Translation sentence {sentence_number}, "
                f"attempt {attempt}/{TRANSLATION_MAX_ATTEMPTS}: "
                f"{last_error[:500]}",
                flush=True
            )

            if attempt < TRANSLATION_MAX_ATTEMPTS:
                time.sleep(attempt * 2)

    raise RuntimeError(
        f"Translation failed for sentence {sentence_number}. "
        f"English text: {sentence[:160]!r}. "
        f"Last error: {last_error[:600]}"
    )


def translate_sentences(client, sentences, progress=None):
    translated = []
    total = len(sentences)

    for index, sentence in enumerate(sentences):
        hindi_text = translate_one_sentence(
            client,
            sentence["text"],
            index + 1
        )

        translated.append({
            **sentence,
            "hindi": hindi_text
        })

        if progress:
            progress(
                (index + 1) / max(total, 1),
                desc=(
                    f"Translating sentence "
                    f"{index + 1}/{total}"
                )
            )

    if len(translated) != len(sentences):
        raise RuntimeError(
            "Translation count does not match sentence count."
        )

    return translated


# =====================================================
# AI33 PRO TASK MANAGEMENT
# =====================================================

def get_task(task_id):
    response = requests.get(
        f"{AI33_TASK_URL}/{task_id}",
        headers={
            "xi-api-key": AI33_API_KEY,
            "Content-Type": "application/json"
        },
        timeout=45
    )

    response.raise_for_status()

    return response.json()


def generate_hindi_audio(text, output_path):
    response = requests.post(
        AI33_TTS_URL,
        headers={
            "xi-api-key": AI33_API_KEY
        },
        data={
            "text": text,
            "voice_id": VOICE_ID,
            "speed": "1",
            "with_transcript": "false"
        },
        timeout=90
    )

    if not response.ok:
        raise RuntimeError(
            "AI33 TTS request failed: "
            + response.text[:1000]
        )

    data = response.json()

    if data.get("success") is False:
        raise RuntimeError(
            "AI33 rejected TTS request: "
            + str(data)[:1000]
        )

    task_id = data.get("task_id") or data.get("id")

    if not task_id:
        raise RuntimeError(
            "AI33 response did not contain task_id: "
            + str(data)[:800]
        )

    deadline = time.time() + 300
    audio_url = None

    while time.time() < deadline:
        task = get_task(task_id)

        status = str(
            task.get("status", "")
        ).lower()

        if status in (
            "error",
            "failed",
            "cancelled"
        ):
            raise RuntimeError(
                "AI33 voice generation failed: "
                + str(
                    task.get("error_message") or task
                )[:800]
            )

        metadata = task.get("metadata") or {}
        audio_url = metadata.get("audio_url")

        if status == "done" and audio_url:
            break

        time.sleep(3)

    if not audio_url:
        raise RuntimeError(
            "AI33 did not return audio_url within 5 minutes."
        )

    audio_response = requests.get(
        audio_url,
        timeout=120
    )

    audio_response.raise_for_status()

    with open(output_path, "wb") as audio_file:
        audio_file.write(audio_response.content)

    if os.path.getsize(output_path) < 100:
        raise RuntimeError(
            "Generated audio file is unexpectedly small."
        )

    return output_path


# =====================================================
# AUDIO TIMELINE
# =====================================================

def build_dubbed_audio(
    sentences,
    english_duration,
    work_dir
):
    pieces = []
    warnings = []
    previous_end = 0.0

    for index, sentence in enumerate(sentences):
        hindi_text = sentence["hindi"].strip()

        if not hindi_text:
            warnings.append(
                f"Sentence {index + 1}: empty translation."
            )
            continue

        source_path = os.path.join(
            work_dir,
            f"hindi_{index:04d}.mp3"
        )

        generate_hindi_audio(
            hindi_text,
            source_path
        )

        original_duration = get_duration(source_path)

        if original_duration <= 0:
            warnings.append(
                f"Sentence {index + 1}: invalid audio duration."
            )
            continue

        desired_start = float(
            sentence["start"]
        )

        next_start = (
            float(sentences[index + 1]["start"])
            if index + 1 < len(sentences)
            else english_duration
        )

        target_start = max(
            desired_start,
            previous_end
        )

        available = max(
            0.10,
            next_start - target_start
        )

        speed = 1.0

        if original_duration > available:
            required_speed = (
                original_duration / available
            )

            if required_speed <= MAX_SPEED:
                speed = max(
                    MIN_SPEED,
                    required_speed
                )
            else:
                speed = MAX_SPEED

                warnings.append(
                    f"Sentence {index + 1}: Hindi speech exceeds "
                    "the available time window."
                )

        processed_path = os.path.join(
            work_dir,
            f"processed_{index:04d}.wav"
        )

        run_ffmpeg([
            "-i", source_path,
            "-af", make_atempo(speed),
            "-ar", "44100",
            "-ac", "2",
            "-codec:a", "pcm_s16le",
            processed_path
        ])

        actual_duration = get_duration(
            processed_path
        )

        if (
            target_start + actual_duration
            > english_duration
        ):
            warnings.append(
                f"Sentence {index + 1}: audio extends beyond "
                "the English duration."
            )

        pieces.append({
            "path": processed_path,
            "start": target_start,
            "duration": actual_duration
        })

        previous_end = (
            target_start + actual_duration
        )

    if not pieces:
        raise RuntimeError(
            "No Hindi audio was successfully generated."
        )

    timeline_parts = []
    cursor = 0.0

    for index, piece in enumerate(pieces):
        start = piece["start"]

        if start > cursor:
            silence_path = os.path.join(
                work_dir,
                f"silence_{index:04d}.wav"
            )

            silence_duration = start - cursor

            run_ffmpeg([
                "-f", "lavfi",
                "-i", "anullsrc=r=44100:cl=stereo",
                "-t", f"{silence_duration:.4f}",
                "-ar", "44100",
                "-ac", "2",
                "-codec:a", "pcm_s16le",
                silence_path
            ])

            timeline_parts.append(silence_path)
            cursor = start

        timeline_parts.append(
            piece["path"]
        )

        cursor = (
            start + piece["duration"]
        )

    manifest_path = os.path.join(
        work_dir,
        "concat_list.txt"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as manifest:
        for path in timeline_parts:
            safe_path = (
                os.path.abspath(path)
                .replace("'", "'\\''")
            )

            manifest.write(
                f"file '{safe_path}'\n"
            )

    output_path = os.path.join(
        work_dir,
        "hindi_dubbed.mp3"
    )

    run_ffmpeg([
        "-f", "concat",
        "-safe", "0",
        "-i", manifest_path,
        "-vn",
        "-codec:a", "libmp3lame",
        "-b:a", "192k",
        output_path
    ], timeout=900)

    return output_path, warnings


# =====================================================
# MAIN WORKFLOW
# =====================================================

def create_dubbing(
    english_audio,
    progress=gr.Progress()
):
    if not GROQ_API_KEY:
        raise gr.Error(
            "GROQ_API_KEY is missing in Render Environment."
        )

    if not AI33_API_KEY:
        raise gr.Error(
            "AI33_API_KEY is missing in Render Environment."
        )

    if not english_audio:
        raise gr.Error(
            "Please upload an English audio file."
        )

    work_dir = tempfile.mkdtemp(
        prefix="ai_dubbing_"
    )

    try:
        progress(
            0.03,
            desc="Preparing audio"
        )

        prepared_audio = os.path.join(
            work_dir,
            "english.mp3"
        )

        prepare_audio(
            english_audio,
            prepared_audio
        )

        duration = get_duration(
            prepared_audio
        )

        size_mb = (
            os.path.getsize(prepared_audio)
            / (1024 * 1024)
        )

        if size_mb > MAX_FILE_MB:
            raise RuntimeError(
                f"Prepared audio is {size_mb:.1f} MB. "
                "Please upload a shorter or lower-bitrate audio."
            )

        client = Groq(
            api_key=GROQ_API_KEY
        )

        progress(
            0.10,
            desc="Transcribing English speech"
        )

        english_segments = transcribe_english(
            client,
            prepared_audio
        )

        progress(
            0.25,
            desc="Grouping sentences"
        )

        sentences = group_sentences(
            english_segments
        )

        if not sentences:
            raise RuntimeError(
                "No sentences were detected."
            )

        progress(
            0.35,
            desc="Translating into Hinglish"
        )

        translated_sentences = translate_sentences(
            client,
            sentences
        )

        progress(
            0.50,
            desc="Generating Hindi cloned voice"
        )

        dubbed_path, warnings = build_dubbed_audio(
            translated_sentences,
            duration,
            work_dir
        )

        progress(
            0.95,
            desc="Preparing download"
        )

        final_duration = get_duration(
            dubbed_path
        )

        status_text = (
            "Dubbing completed.\n\n"
            f"English duration: {duration:.2f} seconds\n"
            f"Dubbed duration: {final_duration:.2f} seconds\n"
            f"English transcript segments: "
            f"{len(english_segments)}\n"
            f"Sentences translated: "
            f"{len(translated_sentences)}\n"
            f"Voice ID: {VOICE_ID}\n"
            f"Translation model: "
            f"{GROQ_TRANSLATION_MODEL}\n"
        )

        if warnings:
            status_text += (
                "\nWarnings:\n- "
                + "\n- ".join(warnings[:15])
            )

        status_text += (
            "\n\nPlease listen to verify pronunciation, "
            "translation and synchronization."
        )

        progress(
            1.0,
            desc="Finished"
        )

        return dubbed_path, status_text

    except gr.Error:
        raise

    except Exception as exc:
        print(
            "DUBBING ERROR:",
            repr(exc),
            flush=True
        )

        raise gr.Error(
            "Dubbing failed: "
            + str(exc)[:1500]
        )


# =====================================================
# USER INTERFACE
# =====================================================

with gr.Blocks(
    title="AudioSync AI - Hindi Dubbing"
) as app:

    gr.Markdown(
        """
# 🎙️ AudioSync AI — Hindi Dubbing Studio

Upload English audio to generate a conversational
Hindi/Hinglish dub using your configured cloned voice.

**Automatic workflow:** Transcription → Translation →
Voice Generation → Timeline Synchronization
"""
    )

    english_audio = gr.Audio(
        label="Upload English Audio",
        type="filepath"
    )

    run_button = gr.Button(
        "Generate Hindi Dub",
        variant="primary"
    )

    output_audio = gr.Audio(
        label="Generated Hindi Dub",
        type="filepath"
    )

    status = gr.Textbox(
        label="Processing Status",
        lines=12
    )

    run_button.click(
        fn=create_dubbing,
        inputs=[english_audio],
        outputs=[output_audio, status]
    )


if __name__ == "__main__":
    app.queue(
        default_concurrency_limit=1
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 10000)
        )
)
