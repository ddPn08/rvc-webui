import importlib
import os
from typing import *

import gradio as gr
import gradio.routes
import torch

from . import models, shared
from .core import preload
from .shared import ROOT_DIR


class Tab:
    TABS_DIR = os.path.join(ROOT_DIR, "modules", "tabs")

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def sort(self):
        return 1

    def title(self):
        return ""

    def ui(self, outlet: Callable):
        pass

    def __call__(self):
        children_dir = self.filepath[:-3]
        children = []

        if os.path.isdir(children_dir):
            for file in os.listdir(children_dir):
                if not file.endswith(".py"):
                    continue
                module_name = file[:-3]
                parent = os.path.relpath(Tab.TABS_DIR, Tab.TABS_DIR).replace("/", ".")

                if parent.startswith("."):
                    parent = parent[1:]
                if parent.endswith("."):
                    parent = parent[:-1]

                children.append(
                    importlib.import_module(f"modules.tabs.{parent}.{module_name}")
                )

        children = sorted(children, key=lambda x: x.sort())

        tabs = []

        for child in children:
            attrs = child.__dict__
            tab = [x for x in attrs.values() if issubclass(x, Tab)]
            if len(tab) > 0:
                tabs.append(tab[0])

        def outlet():
            with gr.Tabs():
                for tab in tabs:
                    with gr.Tab(tab.title()):
                        tab()

        return self.ui(outlet)


def load_tabs() -> List[Tab]:
    tabs = []
    files = os.listdir(os.path.join(ROOT_DIR, "modules", "tabs"))

    for file in files:
        if not file.endswith(".py"):
            continue
        module_name = file[:-3]
        module = importlib.import_module(f"modules.tabs.{module_name}")
        attrs = module.__dict__
        TabClass = [
            x
            for x in attrs.values()
            if type(x) == type and issubclass(x, Tab) and not x == Tab
        ]
        if len(TabClass) > 0:
            tabs.append((file, TabClass[0]))

    tabs = sorted([TabClass(file) for file, TabClass in tabs], key=lambda x: x.sort())
    return tabs


def webpath(fn):
    if fn.startswith(ROOT_DIR):
        web_path = os.path.relpath(fn, ROOT_DIR).replace("\\", "/")
    else:
        web_path = os.path.abspath(fn)

    return f"file={web_path}?{os.path.getmtime(fn)}"


def javascript_html():
    script_js = os.path.join(ROOT_DIR, "script.js")
    head = f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'

    return head


def css_html():
    return f'<link rel="stylesheet" property="stylesheet" href="{webpath(os.path.join(ROOT_DIR, "styles.css"))}">'


def create_head():
    head = ""
    head += css_html()
    head += javascript_html()

    def template_response(*args, **kwargs):
        res = shared.gradio_template_response_original(*args, **kwargs)
        res.body = res.body.replace(b"</head>", f"{head}</head>".encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


def create_ui():
    preload()
    block = gr.Blocks()

    with block:
        with gr.Tabs():
            tabs = load_tabs()
            for tab in tabs:
                with gr.Tab(tab.title()):
                    tab()

    create_head()

    return block


def create_model_list_ui(speaker_id: bool = True, load: bool = True):
    speaker_id_info = {
        "visible": False,
        "maximum": 2333,
    }

    def reload_model(raw=False):
        model_list = models.get_models()
        if len(model_list) > 0:
            models.load_model(model_list[0])

        if models.vc_model is not None:
            speaker_id_info["visible"] = True
            speaker_id_info["maximum"] = models.vc_model.n_spk

        return model_list if raw else gr.Dropdown.update(choices=model_list)

    model_list = reload_model(raw=True)

    def load_model(model_name):
        if load:
            models.load_model(model_name)
            speaker_id_info["visible"] = True
            speaker_id_info["maximum"] = models.vc_model.n_spk
        else:
            model = models.get_vc_model(model_name)
            speaker_id_info["visible"] = True
            speaker_id_info["maximum"] = model.n_spk
            del model
            torch.cuda.empty_cache()
        return gr.Slider.update(
            maximum=speaker_id_info["maximum"], visible=speaker_id_info["visible"]
        )

    with gr.Row(equal_height=False):
        model = gr.Dropdown(
            choices=model_list,
            label="Model",
            value=model_list[0] if len(model_list) > 0 else None,
        )
        speaker_id = gr.Slider(
            minimum=0,
            maximum=speaker_id_info["maximum"],
            step=1,
            label="Speaker ID",
            value=0,
            visible=speaker_id and speaker_id_info["visible"],
            interactive=True,
        )
        reload_model_button = gr.Button("♻️")

        model.change(load_model, inputs=[model], outputs=[speaker_id])
        reload_model_button.click(reload_model, outputs=[model])

    return model, speaker_id


if not hasattr(shared, "gradio_template_response_original"):
    shared.gradio_template_response_original = gradio.routes.templates.TemplateResponse
