import traceback

import gradio as gr

from modules import models, ui
from modules.ui import Tab


def inference_options_ui():
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
            embedder_model = gr.Radio(
                choices=[
                    "hubert_base",
                    "contentvec",
                    "distilhubert",
                    # "distilhubert-ja",
                    # "distilhubert-ja_dev",
                ],
                value="contentvec",
                label="Embedder Model",
            )
        with gr.Column():
            auto_load_index = gr.Checkbox(value=False, label="Auto Load Index")
            faiss_index_file = gr.Textbox(value="", label="Faiss Index File Path")
            big_npy_file = gr.Textbox(value="", label="Big NPY File Path")
            retrieval_feature_ratio = gr.Slider(
                minimum=0,
                maximum=1,
                value=1,
                step=0.01,
                label="Retrieval Feature Ratio",
            )
        with gr.Column():
            fo_curve_file = gr.File(label="F0 Curve File")

    return (
        source_audio,
        transpose,
        embedder_model,
        pitch_extraction_algo,
        auto_load_index,
        faiss_index_file,
        big_npy_file,
        retrieval_feature_ratio,
        fo_curve_file,
    )


class Inference(Tab):
    def title(self):
        return "Inference"

    def sort(self):
        return 1

    def ui(self, outlet):
        def infer(
            sid,
            input_audio,
            embedder_model,
            f0_up_key,
            f0_file,
            f0_method,
            auto_load_index,
            faiss_index_file,
            big_npy_file,
            index_rate,
        ):
            model = models.vc_model
            try:
                yield "Infering...", None
                audio = model.single(
                    sid,
                    input_audio,
                    embedder_model,
                    f0_up_key,
                    f0_file,
                    f0_method,
                    auto_load_index,
                    faiss_index_file,
                    big_npy_file,
                    index_rate,
                )
                yield "Success", (model.tgt_sr, audio)
            except:
                yield "Error: " + traceback.format_exc(), None

        with gr.Group():
            with gr.Box():
                with gr.Column():
                    _, speaker_id = ui.create_model_list_ui()

                    (
                        source_audio,
                        transpose,
                        embedder_model,
                        pitch_extraction_algo,
                        auto_load_index,
                        faiss_index_file,
                        big_npy_file,
                        retrieval_feature_ratio,
                        f0_curve_file,
                    ) = inference_options_ui()

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
                embedder_model,
                transpose,
                f0_curve_file,
                pitch_extraction_algo,
                auto_load_index,
                faiss_index_file,
                big_npy_file,
                retrieval_feature_ratio,
            ],
            outputs=[status, output],
            queue=True,
        )
