#!/usr/bin/env python3
"""
Minimal Gradio test - helps debug issues
"""

import gradio as gr

def test_function(text):
    return f"You entered: {text}"

with gr.Blocks() as demo:
    gr.Markdown("# Minimal Test")

    input_box = gr.Textbox(label="Test Input")
    output_box = gr.Textbox(label="Test Output")
    btn = gr.Button("Test")

    btn.click(fn=test_function, inputs=input_box, outputs=output_box)

if __name__ == "__main__":
    print("Starting minimal Gradio test...")
    demo.launch(server_name="127.0.0.1", server_port=7860)
