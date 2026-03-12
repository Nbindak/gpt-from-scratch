# src/demo/app.py
# =================
# Gradio web interface for the GPT From Scratch model.
# Provides story generation, text completion, and model info tabs.
#
# Usage:
#   pip install gradio
#   python src/demo/app.py
#   python src/demo/app.py --checkpoint checkpoints/finetuned/latest.pt
#   python src/demo/app.py --share   # creates a public link

import argparse
import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import gradio as gr
except ImportError:
    raise ImportError("pip install gradio")

from src.demo.interface import GenerationInterface

_iface: GenerationInterface | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────

def handle_instruct(instruction, context, max_tokens, temperature, top_k):
    if not instruction.strip():
        return "", "⚠ Please enter an instruction."
    response, stats = _iface.instruct(instruction, context, max_tokens, temperature, top_k)
    info = f"Generated {stats['new_tokens']} tokens in {stats['elapsed_s']}s ({stats['tok_per_sec']} tok/s)"
    return response, info


def handle_complete(prompt, max_tokens, temperature, top_k):
    if not prompt.strip():
        return "", "⚠ Please enter a prompt."
    full, stats = _iface.complete(prompt, max_tokens, temperature, top_k)
    info = f"Generated {stats['new_tokens']} tokens in {stats['elapsed_s']}s ({stats['tok_per_sec']} tok/s)"
    return full, info


def handle_model_info():
    return _iface.info() if _iface else "Model not loaded."


# ─────────────────────────────────────────────────────────────────────────────
# Static content
# ─────────────────────────────────────────────────────────────────────────────

HEADER = """
# 🧠 GPT From Scratch

A GPT-style decoder-only Transformer built entirely from scratch using PyTorch.
**Phase 1**: Pretrained on English short stories (next-token prediction).  
**Phase 2**: Fine-tuned on 200+ instruction-response pairs for story writing.
"""

STORY_EXAMPLES = [
    ["Write a short story about a lighthouse keeper who saves a child from a shipwreck.", ""],
    ["Write a story about two old friends reuniting after many years apart.", ""],
    ["Write a mystery story about a detective and a disappeared painting.", ""],
    ["Write a story about someone who discovers an unexpected letter.", ""],
    ["Write a story with the theme: small acts of kindness change lives.", ""],
    ["Write a story about a musician teaching their last lesson.", ""],
    ["Write a poem about the sea at dawn.", ""],
    ["Write a haiku about an old library.", ""],
]

COMPLETION_EXAMPLES = [
    ["The lighthouse stood at the edge of the world,"],
    ["She found the letter in the lining of her mother's coat,"],
    ["The detective examined the empty frame on the wall and"],
    ["The garden had belonged to the house for three generations."],
    ["After thirty years, he returned to the village where"],
]


# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="GPT From Scratch",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:

        gr.Markdown(HEADER)

        # Shared generation controls
        with gr.Accordion("⚙️ Generation Settings", open=False):
            with gr.Row():
                max_tokens  = gr.Slider(50, 500, value=200, step=10,   label="Max New Tokens")
                temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature (0=greedy, 1=creative)")
                top_k       = gr.Slider(1, 100,  value=40,  step=1,    label="Top-K sampling")

        with gr.Tabs():

            # ── Story / Instruction tab ──────────────────────────────────
            with gr.TabItem("📖 Story Generator"):
                gr.Markdown("Give an instruction and the model will write a story or poem.")
                with gr.Row():
                    with gr.Column(scale=2):
                        inst_box = gr.Textbox(label="Instruction", placeholder="Write a story about…", lines=3)
                        ctx_box  = gr.Textbox(label="Context (optional)", placeholder="Set in a lighthouse…", lines=2)
                        with gr.Row():
                            gen_btn   = gr.Button("✍️ Generate", variant="primary")
                            clr_btn   = gr.Button("Clear")
                    with gr.Column(scale=3):
                        out_box  = gr.Textbox(label="Output", lines=18)
                        info_box = gr.Textbox(label="", lines=1, interactive=False)

                gr.Examples(examples=STORY_EXAMPLES, inputs=[inst_box, ctx_box])

                gen_btn.click(handle_instruct,
                    inputs=[inst_box, ctx_box, max_tokens, temperature, top_k],
                    outputs=[out_box, info_box])
                clr_btn.click(lambda: ("", "", "", ""),
                    outputs=[inst_box, ctx_box, out_box, info_box])

            # ── Text completion tab ──────────────────────────────────────
            with gr.TabItem("✏️ Text Completion"):
                gr.Markdown("Type the beginning of a story and the model will continue it.")
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt_box = gr.Textbox(label="Prompt", placeholder="The lighthouse keeper…", lines=5)
                        with gr.Row():
                            comp_btn = gr.Button("🚀 Complete", variant="primary")
                            comp_clr = gr.Button("Clear")
                    with gr.Column(scale=3):
                        comp_out  = gr.Textbox(label="Completed Text", lines=18)
                        comp_info = gr.Textbox(label="", lines=1, interactive=False)

                gr.Examples(examples=COMPLETION_EXAMPLES, inputs=[prompt_box])

                comp_btn.click(handle_complete,
                    inputs=[prompt_box, max_tokens, temperature, top_k],
                    outputs=[comp_out, comp_info])
                comp_clr.click(lambda: ("", "", ""),
                    outputs=[prompt_box, comp_out, comp_info])

            # ── Model info tab ────────────────────────────────────────────
            with gr.TabItem("ℹ️ Model Info"):
                info_btn = gr.Button("Refresh")
                info_md  = gr.Markdown()
                info_btn.click(handle_model_info, outputs=[info_md])
                demo.load(handle_model_info, outputs=[info_md])

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global _iface
    parser = argparse.ArgumentParser(description="Launch GPT From Scratch Gradio demo")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    parser.add_argument("--port",       default=7860, type=int)
    parser.add_argument("--share",      action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    _iface = GenerationInterface(checkpoint_path=args.checkpoint)
    app    = build_app()
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, show_api=False)


if __name__ == "__main__":
    main()
