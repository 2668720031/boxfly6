

import gradio as gr

# 模拟的无人机 ID 列表
drone_id_list = ["drone_1", "drone_2", "drone_3"]
mode_list = [
    (0, "多无人机分别进行动作"),
    (1, "多无人机同时进行某个动作"),
    (2, "指定某一架无人机"),
]

description_to_id = {desc: idx for idx, desc in mode_list}

mode_descriptions = {
    0: "多无人机分别进行动作:对系统默认无人机发送指令。",
    1: "多无人机同时进行某个动作:同时对多架无人机广播指令，适用于集群控制。",
    2: "指定某一架无人机:选择一台具体的无人机进行单独控制，适用于定向任务。",
}


# 执行逻辑
def process_instruction_mode(instruction, mode_label, drone_id):
    mode_id = description_to_id[mode_label]
    if mode_id == 0:
        return f"{mode_list[mode_id][1]} ID={mode_id} 执行: {instruction}"
    elif mode_id == 1:
        return f"{mode_list[mode_id][1]} ID={mode_id} 对所有无人机执行: {instruction}"
    elif mode_id == 2:
        if not drone_id:
            return f"[错误] 未选择无人机 ID"
        return f"{mode_list[mode_id][1]} ID={mode_id}] 指令: {instruction} -> 无人机: {drone_id}"
    else:
        return "[错误] 未知模式"


def toggle_dropdown_visibility(mode):
    mode_id = description_to_id[mode]
    return gr.update(visible=(mode_id == 2))

def get_last_assistant_message():
    return "sadsad"

# UI 构建
with gr.Blocks() as ui:
    gr.Markdown("### 无人机指令控制界面")

    mode_selector = gr.Radio(
        choices=[desc for _, desc in mode_list],
        value=[desc for _, desc in mode_list][0],
        label="选择控制模式"
    )

    drone_id_dropdown = gr.Dropdown(
        choices=drone_id_list,
        label="选择无人机 ID（仅在指定模式下使用）",
        visible=False  # 初始隐藏
    )

    instruction_input = gr.Textbox(
        label="输入指令",
        placeholder="请输入你对无人机的指令"
    )



    mode_select_box = gr.Textbox(
        label="Executing Instruction:",
    )

    assistant_box = gr.Textbox(
        label="Assistant Message List",
        value=get_last_assistant_message(),
        every=1
    )

    submit_btn = gr.Button("提交指令")

    # 控制 Dropdown 显示与否


    mode_selector.change(
        toggle_dropdown_visibility,
        inputs=mode_selector,
        outputs=drone_id_dropdown
    )

    # mode_introduce = gr.Markdown(value=mode_descriptions[], label="模式介绍")

    # 点击提交按钮处理
    submit_btn.click(
        process_instruction_mode,
        inputs=[instruction_input, mode_selector, drone_id_dropdown],
        outputs=mode_select_box
    )

ui.launch(server_name="0.0.0.0")
