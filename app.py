import gradio as gr

def check_audio(english, hindi, script):
    if not english or not hindi:
        return "Please upload both audio files."

    return (
        "Both audio files uploaded successfully!\n"
        "Next step: automatic sentence synchronization."
    )

with gr.Blocks(title="AudioSync AI") as app:
    gr.Markdown("# 🎧 AudioSync AI")
    gr.Markdown("Synchronize your existing English and Hindi audio.")

    english = gr.Audio(
        label="English Audio",
        type="filepath"
    )

    hindi = gr.Audio(
        label="Hindi Audio",
        type="filepath"
    )

    script = gr.File(
        label="English Script TXT",
        file_types=[".txt"],
        type="filepath"
    )

    button = gr.Button("Check Audio")
    output = gr.Textbox(label="Status")

    button.click(
        check_audio,
        inputs=[english, hindi, script],
        outputs=output
    )

app.launch()
