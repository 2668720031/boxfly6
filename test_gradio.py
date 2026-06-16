import gradio as gr

def process_instruction(instruction):
    return "111"

def get_last_assisant_message():
    return "222"


with gr.Blocks() as ui:
    gr.Interface(
            process_instruction,
            'textbox',
            None
        )
    gr.Textbox(
        label="Assistant Message List",
        value=get_last_assisant_message,
        every=0.01,
    )

ui.launch(server_name="0.0.0.0", server_port=7861)