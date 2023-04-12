import importlib
import os

import gradio as gr
import gradio.routes

from . import shared
from .core import preload
from .shared import ROOT_DIR


def load_tabs():
    tabs = []
    for file in ["inference.py", "training.py", "merge.py"]:
        module_name = file[:-3]
        tabs.append(importlib.import_module(f"modules.tabs.{module_name}"))
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
            for tab in load_tabs():
                with gr.Tab(tab.title()):
                    tab.tab()

    create_head()

    return block


if not hasattr(shared, "gradio_template_response_original"):
    shared.gradio_template_response_original = gradio.routes.templates.TemplateResponse
