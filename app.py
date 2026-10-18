import os
import re
import subprocess
import tempfile
import gradio as gr

def run_cmd(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1500:])
    return result.stdout.strip()

def duration(path):
    return float(run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]))

def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    return [
        s.strip() for s in
        re.split(r"(?<=[.!?।])\s+", text)
        if s.strip()
    ]

def synchronize(en_audio, hi_audio, en_script, hi_script):
    if not all([en_audio, hi_audio, en_script, hi_script]):
        raise gr.Error(
            "Upload both audio files and both script TXT files."
        )

    try:
        with open(en_script, "r", encoding="utf-8-sig") as f:
            en_text = f.read()

        with open(hi_script, "r", encoding="utf-8-sig") as f:
            hi_text = f.read()

        en_sentences = split_sentences(en_text)
        hi_sentences = split_sentences(hi_text)

        if not en_sentences or not hi_sentences:
            raise gr.Error("One of the scripts is empty.")

        en_duration = duration(en_audio)
        hi_duration = duration(hi_audio)

        report = f"""
Audio files received successfully.

English audio: {en_duration:.2f} seconds
Hindi audio: {hi_duration:.2f} seconds

English script sentences: {len(en_sentences)}
Hindi script sentences: {len(hi_sentences)}

IMPORTANT:
This prototype has checked the inputs only.
It has NOT synchronized the audio yet.
Sentence counts alone cannot determine sentence timings.
"""

        return report, None

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Processing failed: {str(e)}")

with gr.Blocks(title="AudioSync AI") as app:
    gr.Markdown("# 🎧 AudioSync AI")
    gr.Markdown(
        "Prepare existing English and Hindi audio "
        "for timeline synchronization."
    )

    with gr.Row():
        en_audio = gr.Audio(
            label="English Audio",
            type="filepath"
        )
        hi_audio = gr.Audio(
            label="Hindi Audio",
            type="filepath"
        )

    with gr.Row():
        en_script = gr.File(
            label="English Script TXT",
            file_types=[".txt"],
            type="filepath"
        )
        hi_script = gr.File(
            label="Hindi Script TXT",
            file_types=[".txt"],
            type="filepath"
        )

    button = gr.Button("Check Inputs", variant="primary")
    output = gr.Textbox(label="Analysis", lines=8)
    download = gr.File(label="Synchronized Hindi Audio")

    button.click(
        fn=synchronize,
        inputs=[en_audio, hi_audio, en_script, hi_script],
        outputs=[output, download]
    )

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
