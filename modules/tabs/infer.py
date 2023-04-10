import traceback

import gradio as gr

from modules import models


def title():
    return "Infer"


def tab():
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
        models.load_model(model_name)
        speaker_id_info["visible"] = True
        speaker_id_info["maximum"] = models.vc_model.n_spk
        return gr.Slider.update(
            maximum=speaker_id_info["maximum"], visible=speaker_id_info["visible"]
        )

    def infer(
        sid,
        input_audio,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_big_npy,
        index_rate,
    ):
        model = models.vc_model
        try:
            audio = model.single(
                sid,
                input_audio,
                f0_up_key,
                f0_file,
                f0_method,
                file_index,
                file_big_npy,
                index_rate,
            )
            return "Success", (model.tgt_sr, audio)
        except:
            info = traceback.format_exc()
            return "Error: " + info, None

    with gr.Group():
        with gr.Box():
            with gr.Column():
                with gr.Row().style(equal_height=False):
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
                        visible=speaker_id_info["visible"],
                        interactive=True,
                    )
                    reload_model_button = gr.Button("♻️")

                    model.change(load_model, inputs=[model], outputs=[speaker_id])
                    reload_model_button.click(reload_model, outputs=[model])

                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        source_audio = gr.Textbox(label="Source Audio")
                    with gr.Column():
                        transpose = gr.Slider(
                            minimum=-20, maximum=20, value=0, step=1, label="Transpose"
                        )
                        pitch_extraction_algo = gr.Radio(
                            choices=["pm", "harvest"],
                            value="pm",
                            label="Pitch Extraction Algorithm",
                        )
                    with gr.Column():
                        feature_retrieval_lib = gr.Textbox(
                            value="", label="Feature Retrieval Library"
                        )
                        feature_file_path = gr.Textbox(
                            value="", label="Feature File Path"
                        )
                        retrieval_feature_ratio = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=1,
                            step=0.01,
                            label="Retrieval Feature Ratio",
                        )
                    with gr.Column():
                        fo_curve_file = gr.File(label="F0 Curve File")

                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        status = gr.Textbox(value="", label="Status")
                        output = gr.Audio(label="Output", interactive=False)

                with gr.Row():
                    infer_button = gr.Button("Infer", variant="primary")

    infer_button.click(
        infer,
        inputs=[
            speaker_id,
            source_audio,
            transpose,
            fo_curve_file,
            pitch_extraction_algo,
            feature_retrieval_lib,
            feature_file_path,
            retrieval_feature_ratio,
        ],
        outputs=[status, output],
    )
