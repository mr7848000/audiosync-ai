import os
import re
import json
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
GROQ_TRANSLATION_MODEL = "openai/gpt-oss-20b"

AI33_TTS_URL = "https://api.ai33.pro/v3/text-to-speech"
AI33_TASK_URL = "https://api.ai33.pro/v1/task"

VOICE_ID = "minimax_2618345"
VOICE_MODEL = "speech-02-hd"

MAX_FILE_MB = 24

MIN_SPEED = 1.0
MAX_SPEED = 1.5

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

TRANSLATION_INSTRUCTIONS = """
Translate the script in mix of hindi and english.
Keep the words in english while rest of the sentence
in devanagri hindi language.
Script should be conversational, not traditional translation.
Don't make any headings or bullet points, keep simple text.
Translate sentence by sentence, but only show the translated text.

Additional rules:
- Do not skip any sentence.
- Do not summarize or shorten the script.
- Preserve the original meaning and sentence order.
- Keep English technical terms in English.
- Write Hindi words in Devanagari.
- Return only the translated text.
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
        "-b:a", "96k",
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
            "Please upload a smaller MP3, under 24 MB."
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
# GROUP ENGLISH SEGMENTS INTO SENTENCES
# =====================================================

def group_sentences(segments):
    """
    Combine Whisper segments into sentence-sized groups.
    Preserve the start and end times of the original speech.
    """

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

        # Close a group at sentence-ending punctuation.
        # Also prevent excessively long groups.
        sentence_end = bool(
            re.search(r'[.!?]["\')\]]*$', current["text"])
        )

        too_long = len(current["text"]) >= 280

        if sentence_end or too_long:
            groups.append(current)
            current = None

    if current is not None:
        groups.append(current)

    # Remove empty groups.
    return [
        group for group in groups
        if group["text"].strip()
        and group["end"] > group["start"]
    ]


# =====================================================
# TRANSLATION
# =====================================================

def translate_sentences(client, sentences):
    """
    Translate each English sentence independently.
    The response must contain one Hindi/Hinglish string
    for every input sentence.
    """

    translated = []

    batch_size = 3

    for offset in range(0, len(sentences), batch_size):
        batch = sentences[offset:offset + batch_size]

        payload = [
            {
                "index": index,
                "text": sentence["text"]
            }
            for index, sentence in enumerate(batch)
        ]

        prompt = f"""
{TRANSLATION_INSTRUCTIONS}

Translate each item independently.

IMPORTANT:
- Return valid JSON only.
- Use the exact input indices.
- Return exactly one translation for each input item.
- Do not merge sentences.
- Do not omit sentences.

Required format:
{{"translations":[
    {{"index":0,"text":"Hindi translation"}},
    {{"index":1,"text":"Hindi translation"}}
]}}

INPUT:
{json.dumps(payload, ensure_ascii=False)}
"""

        response = client.chat.completions.create(
            model=GROQ_TRANSLATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional conversational "
                        "Hindi-Hinglish translator. Follow the "
                        "requested format exactly."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=3000
        )

        if not response.choices:
            raise RuntimeError(
                "Translation model returned no response."
            )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Translation response was empty."
            )

        content = content.strip()
        content = re.sub(
            r"^```(?:json)?\s*", "", content
        )
        content = re.sub(r"\s*```$", "", content)

        start = content.find("{")
        end = content.rfind("}")

        if start < 0 or end < start:
            raise RuntimeError(
                "Translation response was not valid JSON."
            )

        data = json.loads(content[start:end + 1])
        items = data.get("translations")

        if not isinstance(items, list):
            raise RuntimeError(
                "Translation list is missing."
            )

        by_index = {}

        for item in items:
            if not isinstance(item, dict):
                continue

            index = item.get("index")
            text = item.get("text")

            if (
                type(index) is int
                and 0 <= index < len(batch)
                and isinstance(text, str)
                and text.strip()
            ):
                if index in by_index:
                    raise RuntimeError(
                        "Duplicate translation index returned."
                    )

                by_index[index] = text.strip()

        if len(by_index) != len(batch):
            raise RuntimeError(
                "AI did not translate every sentence. "
                "Please retry the operation."
            )

        for index, sentence in enumerate(batch):
            translated.append({
                **sentence,
                "hindi": by_index[index]
            })

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


def generate_hindi_audio(text, output_path, progress=None):
    """
    Submit a MiniMax TTS task and wait for its audio URL.
    """

    response = requests.post(
        AI33_TTS_URL,
        headers={
            "xi-api-key": AI33_API_KEY
        },
        data={
            "text": text,
            "voice_id": VOICE_ID,
            "speed": "1",
            "with_transcript": "false",
            "model": VOICE_MODEL
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
            "AI33 rejected the TTS request: "
            + json.dumps(data, ensure_ascii=False)[:1000]
        )

    task_id = data.get("task_id") or data.get("id")

    if not task_id:
        raise RuntimeError(
            "AI33 response did not contain task_id. "
            + json.dumps(data, ensure_ascii=False)[:800]
        )

    audio_url = None
    last_status = None

    # Allow up to 5 minutes for each voice generation task.
    deadline = time.time() + 300

    while time.time() < deadline:
        task = get_task(task_id)

        status = str(task.get("status", "")).lower()

        if status != last_status:
            last_status = status

        if status in ("error", "failed", "cancelled"):
            raise RuntimeError(
                "AI33 voice generation failed: "
                + str(task.get("error_message") or task)
            )

        metadata = task.get("metadata") or {}
        audio_url = metadata.get("audio_url")

        if status == "done" and audio_url:
            break

        time.sleep(3)

    if not audio_url:
        raise RuntimeError(
            "AI33 task did not provide audio_url within 5 minutes."
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
# TIMELINE SYNCHRONIZATION
# =====================================================

def build_dubbed_audio(sentences, english_duration, work_dir):
    """
    Generate each Hindi sentence separately.
    Place sentences chronologically.
    Try to fit each sentence before the next English
    sentence while avoiding overlapping clips.
    """

    pieces = []
    warnings = []

    previous_end = 0.0

    total = len(sentences)

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

        desired_start = float(sentence["start"])
        next_start = (
            float(sentences[index + 1]["start"])
            if index + 1 < total
            else english_duration
        )

        # Never place a clip on top of the previous clip.
        target_start = max(desired_start, previous_end)

        available = max(
            0.05,
            next_start - target_start
        )

        speed = 1.0

        if original_duration > available:
            required_speed = original_duration / available

            if required_speed <= MAX_SPEED:
                speed = max(MIN_SPEED, required_speed)
            else:
                # Do not cut off speech just to force it
                # into an unrealistically short interval.
                speed = MAX_SPEED

                warnings.append(
                    f"Sentence {index + 1}: speech is longer "
                    "than its available timeline. Timing may "
                    "extend beyond the original sentence."
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

        actual_duration = get_duration(processed_path)

        if target_start + actual_duration > english_duration:
            warnings.append(
                f"Sentence {index + 1}: audio extends beyond "
                "the English audio duration."
            )

        pieces.append({
            "path": processed_path,
            "start": target_start,
            "duration": actual_duration
        })

        previous_end = target_start + actual_duration

    if not pieces:
        raise RuntimeError(
            "No Hindi audio was successfully generated."
        )

    # Create the final timeline by delaying each clip.
    inputs = []
    filters = []
    labels = []

    for index, piece in enumerate(pieces):
        inputs.extend(["-i", piece["path"]])

        delay_ms = max(
            0,
            int(piece["start"] * 1000)
        )

        label = f"clip{index}"

        filters.append(
            f"[{index}:a]"
            f"adelay={delay_ms}|{delay_ms}"
            f"[{label}]"
        )

        labels.append(f"[{label}]")

    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:"
          "duration=longest:normalize=0[mix]"
    )

    output_path = os.path.join(
        work_dir,
        "hindi_dubbed.mp3"
    )

    run_ffmpeg(
        inputs + [
            "-filter_complex", ";".join(filters),
            "-map", "[mix]",
            "-codec:a", "libmp3lame",
            "-b:a", "192k",
            output_path
        ],
        timeout=900
    )

    return output_path, warnings


# =====================================================
# MAIN WORKFLOW
# =====================================================

def create_dubbing(english_audio, progress=gr.Progress()):
    if not GROQ_API_KEY:
        raise gr.Error(
            "GROQ_API_KEY is missing in Render Environment."
        )

    if not AI33_API_KEY:
        raise gr.Error(
            "AI33_API_KEY is missing in Render Environment."
        )

    if not english_audio:
        raise gr.Error("Please upload an English audio file.")

    work_dir = tempfile.mkdtemp(
        prefix="ai_dubbing_"
    )

    try:
        progress(0.03, desc="Preparing audio")

        prepared_audio = os.path.join(
            work_dir,
            "english.mp3"
        )

        prepare_audio(
            english_audio,
            prepared_audio
        )

        duration = get_duration(prepared_audio)

        # Groq's transcription API has a file-size limit.
        size_mb = (
            os.path.getsize(prepared_audio)
            / (1024 * 1024)
        )

        if size_mb > MAX_FILE_MB:
            raise RuntimeError(
                f"Prepared audio is {size_mb:.1f} MB. "
                "Please upload a shorter or lower-bitrate audio."
            )

        client = Groq(api_key=GROQ_API_KEY)

        progress(0.10, desc="Transcribing English speech")

        english_segments = transcribe_english(
            client,
            prepared_audio
        )

        progress(0.25, desc="Grouping sentences")

        sentences = group_sentences(
            english_segments
        )

        if not sentences:
            raise RuntimeError(
                "No sentences were detected."
            )

        progress(0.35, desc="Translating into Hinglish")

        translated_sentences = translate_sentences(
            client,
            sentences
        )

        progress(0.45, desc="Generating Hindi cloned voice")

        # Voice generation can take time and use API credits.
        # Update progress after each completed sentence.
        generated_count = 0

        def voice_progress(current, total):
            nonlocal generated_count
            generated_count = current

        dubbed_path, warnings = build_dubbed_audio(
            translated_sentences,
            duration,
            work_dir
        )

        progress(0.95, desc="Preparing download")

        final_duration = get_duration(dubbed_path)

        status = (
            "Dubbing completed.\n\n"
            f"English duration: {duration:.2f} seconds\n"
            f"Dubbed duration: {final_duration:.2f} seconds\n"
            f"English transcript segments: "
            f"{len(english_segments)}\n"
            f"Sentences translated: {len(translated_sentences)}\n"
            f"Voice: MiniMax cloned voice\n"
            f"Voice ID: {VOICE_ID}\n"
            f"Model: {VOICE_MODEL}\n"
        )

        if warnings:
            status += (
                "\nWarnings:\n- "
                + "\n- ".join(warnings[:15])
            )

        status += (
            "\n\nPlease listen to the output and check "
            "pronunciation, sentence timing and translation."
        )

        progress(1.0, desc="Finished")

        return dubbed_path, status

    except gr.Error:
        raise

    except Exception as exc:
        raise gr.Error(
            "Dubbing failed: " + str(exc)[:1200]
        )

    finally:
        # Do not delete files here. Gradio needs the output
        # file to remain available for playback/download.
        pass


# =====================================================
# USER INTERFACE
# =====================================================

with gr.Blocks(title="AudioSync AI - Hindi Dubbing") as app:
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
    app.queue(default_concurrency_limit=1)

    app.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 10000)
        )
    )
