import json
import os
from typing import *

import gradio as gr
import torch

from modules import models
from modules.ui import Tab
from modules.merge import merge
from modules.tabs.inference import inference_options_ui

MERGE_METHODS = {
    "weight_sum": "Weight sum:A*(1-alpha)+B*alpha",
    "add_diff": "Add difference:A+(B-C)*alpha",
}


class Merge(Tab):
    def title(self):
        return "Merge"

    def sort(self):
        return 3

    def ui(self, outlet):
        def merge_ckpt(model_a, model_b, model_c, weight_text, alpha, each_key, method):
            model_a = model_a if type(model_a) != list and model_a != "" else None
            model_b = model_b if type(model_b) != list and model_b != "" else None
            model_c = model_c if type(model_c) != list and model_c != "" else None

            if each_key:
                weights = json.loads(weight_text)
            else:
                weights = {}

            method = [k for k, v in MERGE_METHODS.items() if v == method][0]
            return merge(
                os.path.join(models.MODELS_DIR, "checkpoints", model_a),
                os.path.join(models.MODELS_DIR, "checkpoints", model_b),
                os.path.join(models.MODELS_DIR, "checkpoints", model_c)
                if model_c
                else None,
                alpha,
                weights,
                method,
            )

        def merge_and_save(
            model_a, model_b, model_c, alpha, each_key, weight_text, method, out_name
        ):
            print(each_key)
            out_path = os.path.join(models.MODELS_DIR, "checkpoints", out_name)
            if os.path.exists(out_path):
                return "Model name already exists."
            merged = merge_ckpt(
                model_a, model_b, model_c, weight_text, alpha, each_key, method
            )
            torch.save(merged, os.path.join(models.MODELS_DIR, "checkpoints", out_name))
            return "Success"

        def merge_and_gen(
            model_a,
            model_b,
            model_c,
            alpha,
            each_key,
            weight_text,
            method,
            speaker_id,
            source_audio,
            transpose,
            fo_curve_file,
            pitch_extraction_algo,
            retrieval_feature_ratio,
        ):
            merged = merge_ckpt(
                model_a, model_b, model_c, weight_text, alpha, each_key, method
            )
            model = models.VC_MODEL("merge", merged)
            audio = model.single(
                speaker_id,
                source_audio,
                transpose,
                fo_curve_file,
                pitch_extraction_algo,
                retrieval_feature_ratio,
            )
            tgt_sr = model.tgt_sr
            del merged
            del model
            torch.cuda.empty_cache()
            return "Success", (tgt_sr, audio)

        def reload_model():
            model_list = models.get_models()
            return (
                gr.Dropdown.update(choices=model_list),
                gr.Dropdown.update(choices=model_list),
                gr.Dropdown.update(choices=model_list),
            )

        def update_speaker_ids(model):
            if model == "":
                return gr.Slider.update(
                    maximum=0,
                    visible=False,
                )
            model = torch.load(
                os.path.join(models.MODELS_DIR, "checkpoints", model),
                map_location="cpu",
            )
            vc_model = models.VC_MODEL("merge", model)
            max = vc_model.n_spk
            del model
            del vc_model
            return gr.Slider.update(
                maximum=max,
                visible=True,
            )

        with gr.Group():
            with gr.Column():
                with gr.Row().style(equal_height=False):
                    model_a = gr.Dropdown(choices=models.get_models(), label="Model A")
                    model_b = gr.Dropdown(choices=models.get_models(), label="Model B")
                    model_c = gr.Dropdown(choices=models.get_models(), label="Model C")
                    reload_model_button = gr.Button("♻️")
                    reload_model_button.click(
                        reload_model, outputs=[model_a, model_b, model_c]
                    )
                with gr.Row().style(equal_height=False):
                    method = gr.Radio(
                        label="Merge method",
                        choices=list(MERGE_METHODS.values()),
                        value="Weight sum:A*(1-alpha)+B*alpha",
                    )
                    output_name = gr.Textbox(label="Output name")
                    each_key = gr.Checkbox(label="Each key merge")
                with gr.Row().style(equal_height=False):
                    base_alpha = gr.Slider(
                        label="Base alpha", minimum=0, maximum=1, value=0.5, step=0.01
                    )

                default_weights = {}
                weights = {}

                def create_weight_ui(name: str, *keys_list: List[List[str]]):
                    with gr.Accordion(label=name, open=False):
                        with gr.Row().style(equal_height=False):
                            for keys in keys_list:
                                with gr.Column():
                                    for key in keys:
                                        default_weights[key] = 0.5
                                        weights[key] = gr.Slider(
                                            label=key,
                                            minimum=0,
                                            maximum=1,
                                            step=0.01,
                                            value=0.5,
                                        )

                with gr.Box(visible=False) as each_key_ui:
                    with gr.Column():
                        create_weight_ui(
                            "enc_p",
                            [
                                "enc_p.encoder.attn_layers.0",
                                "enc_p.encoder.attn_layers.1",
                                "enc_p.encoder.attn_layers.2",
                                "enc_p.encoder.attn_layers.3",
                                "enc_p.encoder.attn_layers.4",
                                "enc_p.encoder.attn_layers.5",
                                "enc_p.encoder.norm_layers_1.0",
                                "enc_p.encoder.norm_layers_1.1",
                                "enc_p.encoder.norm_layers_1.2",
                                "enc_p.encoder.norm_layers_1.3",
                                "enc_p.encoder.norm_layers_1.4",
                                "enc_p.encoder.norm_layers_1.5",
                            ],
                            [
                                "enc_p.encoder.ffn_layers.0",
                                "enc_p.encoder.ffn_layers.1",
                                "enc_p.encoder.ffn_layers.2",
                                "enc_p.encoder.ffn_layers.3",
                                "enc_p.encoder.ffn_layers.4",
                                "enc_p.encoder.ffn_layers.5",
                                "enc_p.encoder.norm_layers_2.0",
                                "enc_p.encoder.norm_layers_2.1",
                                "enc_p.encoder.norm_layers_2.2",
                                "enc_p.encoder.norm_layers_2.3",
                                "enc_p.encoder.norm_layers_2.4",
                                "enc_p.encoder.norm_layers_2.5",
                            ],
                            [
                                "enc_p.emb_phone",
                                "enc_p.emb_pitch",
                            ],
                        )

                        create_weight_ui(
                            "dec",
                            [
                                "dec.noise_convs.0",
                                "dec.noise_convs.1",
                                "dec.noise_convs.2",
                                "dec.noise_convs.3",
                                "dec.noise_convs.4",
                                "dec.noise_convs.5",
                                "dec.ups.0",
                                "dec.ups.1",
                                "dec.ups.2",
                                "dec.ups.3",
                            ],
                            [
                                "dec.resblocks.0",
                                "dec.resblocks.1",
                                "dec.resblocks.2",
                                "dec.resblocks.3",
                                "dec.resblocks.4",
                                "dec.resblocks.5",
                                "dec.resblocks.6",
                                "dec.resblocks.7",
                                "dec.resblocks.8",
                                "dec.resblocks.9",
                                "dec.resblocks.10",
                                "dec.resblocks.11",
                            ],
                            [
                                "dec.m_source.l_linear",
                                "dec.conv_pre",
                                "dec.conv_post",
                                "dec.cond",
                            ],
                        )

                        create_weight_ui(
                            "flow",
                            [
                                "flow.flows.0",
                                "flow.flows.1",
                                "flow.flows.2",
                                "flow.flows.3",
                                "flow.flows.4",
                                "flow.flows.5",
                                "flow.flows.6",
                                "emb_g.weight",
                            ],
                        )

                        with gr.Accordion(label="JSON", open=False):
                            weights_text = gr.TextArea(
                                value=json.dumps(default_weights),
                            )

                with gr.Accordion(label="Inference options", open=False):
                    with gr.Row().style(equal_height=False):
                        speaker_id = gr.Slider(
                            minimum=0,
                            maximum=2333,
                            step=1,
                            label="Speaker ID",
                            value=0,
                            visible=True,
                            interactive=True,
                        )
                    (
                        source_audio,
                        transpose,
                        pitch_extraction_algo,
                        retrieval_feature_ratio,
                        fo_curve_file,
                    ) = inference_options_ui()

                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        status = gr.Textbox(value="", label="Status")
                        audio_output = gr.Audio(label="Output", interactive=False)

                with gr.Row().style(equal_height=False):
                    merge_and_save_button = gr.Button(
                        "Merge and save", variant="primary"
                    )
                    merge_and_gen_button = gr.Button("Merge and gen", variant="primary")

                def each_key_on_change(each_key):
                    return gr.update(visible=each_key)

                each_key.change(
                    fn=each_key_on_change,
                    inputs=[each_key],
                    outputs=[each_key_ui],
                )

                def update_weights_text(data):
                    d = {}
                    for key in weights.keys():
                        d[key] = data[weights[key]]
                    return json.dumps(d)

                for w in weights.values():
                    w.change(
                        fn=update_weights_text,
                        inputs={*weights.values()},
                        outputs=[weights_text],
                    )

                merge_data = [
                    model_a,
                    model_b,
                    model_c,
                    base_alpha,
                    each_key,
                    weights_text,
                    method,
                ]

                inference_opts = [
                    speaker_id,
                    source_audio,
                    transpose,
                    fo_curve_file,
                    pitch_extraction_algo,
                    retrieval_feature_ratio,
                ]

                merge_and_save_button.click(
                    fn=merge_and_save,
                    inputs=[
                        *merge_data,
                        output_name,
                    ],
                    outputs=[status],
                )
                merge_and_gen_button.click(
                    fn=merge_and_gen,
                    inputs=[
                        *merge_data,
                        *inference_opts,
                    ],
                    outputs=[status, audio_output],
                )

                model_a.change(
                    update_speaker_ids, inputs=[model_a], outputs=[speaker_id]
                )
