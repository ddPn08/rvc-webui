import gradio as gr

from modules.separate import separate_audio
from modules.ui import Tab


class Split(Tab):
    def title(self):
        return "Split Audio"

    def sort(self):
        return 5

    def ui(self, outlet):
        def separate(
            input_audio,
            output_dir,
            silence_thresh,
            min_silence_len,
            keep_silence,
            margin,
            padding,
            min,
            max,
        ):
            min = None if min == 0 else min
            max = None if max == 0 else max
            separate_audio(
                input_audio,
                output_dir,
                int(silence_thresh),
                int(min_silence_len),
                int(keep_silence),
                int(margin),
                padding,
                int(min),
                int(max),
            )
            return "Success"

        with gr.Group():
            with gr.Column():
                with gr.Row().style(equal_height=False):
                    input_audio = gr.Textbox(label="Input Audio (File or Directory)")
                    output_dir = gr.Textbox(label="Output Directory")

                with gr.Row().style(equal_height=False):
                    silence_thresh = gr.Number(value=-40, label="Silence Threshold")
                    min_silence_len = gr.Number(
                        value=750, label="Minimum Silence Length"
                    )
                    keep_silence = gr.Number(value=750, label="Keep Silence")
                    margin = gr.Number(value=0, label="Margin")
                    padding = gr.Checkbox(value=True, label="Padding")

                with gr.Row().style(equal_height=False):
                    min = gr.Number(value=1000, label="Minimum audio length")
                    max = gr.Number(value=5000, label="Maximum audio length")

                with gr.Row().style(equal_height=False):
                    status = gr.Textbox(value="", label="Status")
                with gr.Row().style(equal_height=False):
                    separate_button = gr.Button("Separate", variant="primary")

        separate_button.click(
            separate,
            inputs=[
                input_audio,
                output_dir,
                silence_thresh,
                min_silence_len,
                keep_silence,
                margin,
                padding,
                min,
                max,
            ],
            outputs=[status],
        )
