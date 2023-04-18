import os
import shutil
from multiprocessing import cpu_count

import gradio as gr

from modules import utils
from modules.shared import MODELS_DIR
from modules.training.extract import extract_f0, extract_feature
from modules.training.preprocess import preprocess_dataset
from modules.training.train import (
    create_dataset_meta,
    glob_dataset,
    train_index,
    train_model,
)
from modules.ui import Tab

SR_DICT = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


class Training(Tab):
    def title(self):
        return "Training"

    def sort(self):
        return 2

    def ui(self, outlet):
        def train_index_only(
            model_name,
            target_sr,
            f0,
            dataset_glob,
            speaker_id,
            num_cpu_process,
            norm_audio_when_preprocess,
            pitch_extraction_algo,
            embedder_name,
            ignore_cache,
        ):
            f0 = f0 == "Yes"
            norm_audio_when_preprocess = norm_audio_when_preprocess == "Yes"
            training_dir = os.path.join(MODELS_DIR, "training", "models", model_name)
            yield f"Training directory: {training_dir}"

            if os.path.exists(training_dir) and ignore_cache:
                shutil.rmtree(training_dir)

            os.makedirs(training_dir, exist_ok=True)

            datasets = glob_dataset(dataset_glob, speaker_id)

            yield "Preprocessing..."
            preprocess_dataset(
                datasets,
                SR_DICT[target_sr],
                num_cpu_process,
                training_dir,
                norm_audio_when_preprocess,
            )

            if f0:
                yield "Extracting f0..."
                extract_f0(training_dir, num_cpu_process, pitch_extraction_algo)

            yield "Extracting features..."
            extract_feature(training_dir, embedder_name)

            yield "Training index..."
            train_index(
                training_dir,
                model_name,
                256 if not embedder_name.endswith("768") else 768,
            )

            yield "Training complete"

        def train_all(
            model_name,
            target_sr,
            f0,
            dataset_glob,
            speaker_id,
            gpu_id,
            num_cpu_process,
            norm_audio_when_preprocess,
            pitch_extraction_algo,
            batch_size,
            cache_batch,
            num_epochs,
            save_every_epoch,
            pre_trained_bottom_model_g,
            pre_trained_bottom_model_d,
            embedder_name,
            ignore_cache,
        ):
            f0 = f0 == "Yes"
            norm_audio_when_preprocess = norm_audio_when_preprocess == "Yes"
            training_dir = os.path.join(MODELS_DIR, "training", "models", model_name)
            yield f"Training directory: {training_dir}"

            if os.path.exists(training_dir) and ignore_cache:
                shutil.rmtree(training_dir)

            os.makedirs(training_dir, exist_ok=True)

            datasets = glob_dataset(dataset_glob, speaker_id)

            yield "Preprocessing..."
            preprocess_dataset(
                datasets,
                SR_DICT[target_sr],
                num_cpu_process,
                training_dir,
                norm_audio_when_preprocess,
            )

            if f0:
                yield "Extracting f0..."
                extract_f0(training_dir, num_cpu_process, pitch_extraction_algo)

            yield "Extracting features..."
            extract_feature(training_dir, embedder_name)

            create_dataset_meta(training_dir, target_sr, f0)

            yield "Training model..."

            print(f"train_all: emb_name: {embedder_name}")

            train_model(
                gpu_id.split(","),
                training_dir,
                model_name,
                target_sr,
                1 if f0 else 0,
                batch_size,
                cache_batch,
                num_epochs,
                save_every_epoch,
                pre_trained_bottom_model_g,
                pre_trained_bottom_model_d,
                embedder_name,
            )

            yield "Training index..."

            train_index(
                training_dir,
                model_name,
                256 if not embedder_name.endswith("768") else 768,
            )

            yield "Training completed"

        with gr.Group():
            with gr.Box():
                with gr.Column():
                    with gr.Row().style(equal_height=False):
                        model_name = gr.Textbox(label="Model Name")
                        ignore_cache = gr.Checkbox(label="Ignore cache")
                        target_sr = gr.Radio(
                            choices=["32k", "40k", "48k"],
                            value="40k",
                            label="Target sampling rate",
                        )
                        f0 = gr.Radio(
                            choices=["Yes", "No"],
                            value="Yes",
                            label="f0 Model",
                        )

                    with gr.Row().style(equal_height=False):
                        dataset_glob = gr.Textbox(
                            label="Dataset glob", placeholder="data/**/*.wav"
                        )
                        speaker_id = gr.Slider(
                            maximum=4, minimum=0, value=0, step=1, label="Speaker ID"
                        )
                        embedder_name = gr.Radio(
                            choices=[
                                "hubert_base",
                                "contentvec",
                                "hubert_base768",
                                "contentvec768",
                            ],
                            value="hubert_base",
                            label="Using phone embedder",
                        )
                        norm_audio_when_preprocess = gr.Radio(
                            choices=["Yes", "No"],
                            value="Yes",
                            label="Normalize audio volume when preprocess",
                        )
                    with gr.Row().style(equal_height=False):
                        gpu_id = gr.Textbox(
                            label="GPU ID",
                            value=", ".join([f"{x.index}" for x in utils.get_gpus()]),
                        )
                        num_cpu_process = gr.Slider(
                            minimum=0,
                            maximum=cpu_count(),
                            step=1,
                            value=cpu_count(),
                            label="Number of CPU processes",
                        )
                        pitch_extraction_algo = gr.Radio(
                            choices=["pm", "harvest", "dio"],
                            value="harvest",
                            label="Pitch extraction algorithm",
                        )
                    with gr.Row().style(equal_height=False):
                        batch_size = gr.Slider(
                            minimum=1, maximum=64, value=4, step=1, label="Batch size"
                        )
                        num_epochs = gr.Slider(
                            minimum=1,
                            maximum=1000,
                            value=100,
                            step=1,
                            label="Number of epochs",
                        )
                        save_every_epoch = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            value=10,
                            step=1,
                            label="Save every epoch",
                        )
                        cache_batch = gr.Checkbox(label="Cache batch", value=True)
                    with gr.Row().style(equal_height=False):
                        pre_trained_bottom_model_g = gr.Textbox(
                            label="Pre-trained bottom model G path",
                            value=os.path.join(MODELS_DIR, "pretrained", "f0G40k.pth"),
                        )
                        pre_trained_bottom_model_d = gr.Textbox(
                            label="Pre-trained bottom model D path",
                            value=os.path.join(MODELS_DIR, "pretrained", "f0D40k.pth"),
                        )

                    with gr.Row().style(equal_height=False):
                        status = gr.Textbox(value="", label="Status")
                    with gr.Row().style(equal_height=False):
                        train_index_button = gr.Button("Train Index", variant="primary")
                        train_all_button = gr.Button("Train", variant="primary")

        train_index_button.click(
            train_index_only,
            inputs=[
                model_name,
                target_sr,
                f0,
                dataset_glob,
                speaker_id,
                num_cpu_process,
                norm_audio_when_preprocess,
                pitch_extraction_algo,
                embedder_name,
                ignore_cache,
            ],
            outputs=[status],
        )

        train_all_button.click(
            train_all,
            inputs=[
                model_name,
                target_sr,
                f0,
                dataset_glob,
                speaker_id,
                gpu_id,
                num_cpu_process,
                norm_audio_when_preprocess,
                pitch_extraction_algo,
                batch_size,
                cache_batch,
                num_epochs,
                save_every_epoch,
                pre_trained_bottom_model_g,
                pre_trained_bottom_model_d,
                embedder_name,
                ignore_cache,
            ],
            outputs=[status],
        )
