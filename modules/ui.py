import importlib
import os
import gradio as gr


def load_tabs():
    tab_folder = os.path.join(os.path.dirname(__file__), "tabs")
    tabs = []
    for file in os.listdir(tab_folder):
        if file.endswith(".py"):
            module_name = file[:-3]
            tabs.append(importlib.import_module(f"modules.tabs.{module_name}"))
    return tabs


def create_ui():
    block = gr.Blocks()

    with block:
        with gr.Tabs():
            for tab in load_tabs():
                with gr.Tab(tab.title()):
                    tab.tab()

    return block
