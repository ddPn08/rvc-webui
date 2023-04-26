import os
import shutil
from multiprocessing import cpu_count

import gradio as gr

from lib.rvc.preprocessing import extract_f0, extract_feature, split
from lib.rvc.train import create_dataset_meta, glob_dataset, train_index, train_model
from modules import models, utils
from modules.shared import MODELS_DIR
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
            gpu_id,
            num_cpu_process,
            norm_audio_when_preprocess,
            pitch_extraction_algo,
            embedder_name,
            embedding_channels,
            ignore_cache,
        ):
            embedding_channels = int(embedding_channels)
            f0 = f0 == "Yes"
            norm_audio_when_preprocess = norm_audio_when_preprocess == "Yes"
            training_dir = os.path.join(MODELS_DIR, "training", "models", model_name)
            gpu_ids = [int(x.strip()) for x in gpu_id.split(",")]
            yield f"Training directory: {training_dir}"

            if os.path.exists(training_dir) and ignore_cache:
                shutil.rmtree(training_dir)

            os.makedirs(training_dir, exist_ok=True)

            datasets = glob_dataset(dataset_glob, speaker_id)

            yield "Preprocessing..."
            split.preprocess_audio(
                datasets,
                SR_DICT[target_sr],
                num_cpu_process,
                training_dir,
                norm_audio_when_preprocess,
            )

            if f0:
                yield "Extracting f0..."
                extract_f0.run(training_dir, num_cpu_process, pitch_extraction_algo)

            yield "Extracting features..."

            embedder_filepath, _, embedder_load_from = models.get_embedder(
                embedder_name
            )

            if embedder_load_from == "local":
                embedder_filepath = os.path.join(MODELS_DIR, embedder_filepath)

            extract_feature.run(
                training_dir,
                embedder_filepath,
                embedder_load_from,
                embedding_channels == 768,
                gpu_ids,
            )

            out_dir = os.path.join(MODELS_DIR, "checkpoints")

            yield "Training index..."
            train_index(
                training_dir,
                model_name,
                out_dir,
                embedding_channels,
            )

            yield "Training complete"

        def train_all(
            model_name,
            sampling_rate_str,
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
            embedding_channels,
            ignore_cache,
        ):
            batch_size = int(batch_size)
            num_epochs = int(num_epochs)
            embedding_channels = int(embedding_channels)
            f0 = f0 == "Yes"
            norm_audio_when_preprocess = norm_audio_when_preprocess == "Yes"
            training_dir = os.path.join(MODELS_DIR, "training", "models", model_name)
            gpu_ids = [int(x.strip()) for x in gpu_id.split(",")]

            if os.path.exists(training_dir) and ignore_cache:
                shutil.rmtree(training_dir)

            os.makedirs(training_dir, exist_ok=True)

            yield f"Training directory: {training_dir}"

            datasets = glob_dataset(dataset_glob, speaker_id)

            yield "Preprocessing..."
            split.preprocess_audio(
                datasets,
                SR_DICT[sampling_rate_str],
                num_cpu_process,
                training_dir,
                norm_audio_when_preprocess,
            )

            if f0:
                yield "Extracting f0..."
                extract_f0.run(training_dir, num_cpu_process, pitch_extraction_algo)

            yield "Extracting features..."

            embedder_filepath, _, embedder_load_from = models.get_embedder(
                embedder_name
            )

            if embedder_load_from == "local":
                embedder_filepath = os.path.join(
                    MODELS_DIR, "embeddings", embedder_filepath
                )

            extract_feature.run(
                training_dir,
                embedder_filepath,
                embedder_load_from,
                embedding_channels == 768,
                gpu_ids,
            )

            create_dataset_meta(training_dir, sampling_rate_str, f0)

            yield "Training model..."

            print(f"train_all: emb_name: {embedder_name}")

            config = utils.load_config(
                training_dir, sampling_rate_str, embedding_channels
            )
            out_dir = os.path.join(MODELS_DIR, "checkpoints")

            train_model(
                gpu_ids,
                config,
                training_dir,
                model_name,
                out_dir,
                sampling_rate_str,
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

            train_index(training_dir, model_name, out_dir, embedding_channels)

            yield "Training completed"

        with gr.Group():
            with gr.Box():
                with gr.Column():
                    with gr.Row().style(equal_height=False):
                        model_name = gr.Textbox(label="Model Name")
                        ignore_cache = gr.Checkbox(label="Ignore cache")
                        dataset_glob = gr.Textbox(
                            label="Dataset glob", placeholder="data/**/*.wav"
                        )
                        speaker_id = gr.Slider(
                            maximum=4, minimum=0, value=0, step=1, label="Speaker ID"
                        )

                    with gr.Row().style(equal_height=False):
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
                        embedder_name = gr.Radio(
                            choices=[
                                "hubert_base",
                                "contentvec",
                                "distilhubert",
                                # "distilhubert-ja",    # temporary
                                # "distilhubert-ja_dev",
                            ],
                            value="hubert_base",
                            label="Using phone embedder",
                        )
                        embedding_channels = gr.Radio(
                            choices=["256", "768"],
                            value="256",
                            label="Embedding channels",
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
                        norm_audio_when_preprocess = gr.Radio(
                            choices=["Yes", "No"],
                            value="Yes",
                            label="Normalize audio volume when preprocess",
                        )
                        pitch_extraction_algo = gr.Radio(
                            choices=["pm", "harvest", "dio"],
                            value="harvest",
                            label="Pitch extraction algorithm",
                        )
                    with gr.Row().style(equal_height=False):
                        batch_size = gr.Number(value=4, label="Batch size")
                        num_epochs = gr.Number(
                            value=30,
                            label="Number of epochs",
                        )
                        save_every_epoch = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=10,
                            step=1,
                            label="Save every epoch",
                        )
                        cache_batch = gr.Checkbox(label="Cache batch", value=True)
                    with gr.Row().style(equal_height=False):
                        pre_trained_generator = gr.Textbox(
                            label="Pre trained generator path",
                            value=os.path.join(
                                MODELS_DIR, "pretrained", "f0G40k256.pth"
                            ),
                        )
                        pre_trained_discriminator = gr.Textbox(
                            label="Pre trained discriminator path",
                            value=os.path.join(
                                MODELS_DIR, "pretrained", "f0D40k256.pth"
                            ),
                        )

                    with gr.Row().style(equal_height=False):
                        status = gr.Textbox(value="", label="Status")
                    with gr.Row().style(equal_height=False):
                        train_index_button = gr.Button("Train Index", variant="primary")
                        train_all_button = gr.Button("Train", variant="primary")

        def change_pretrained(sr, f0, emb_channels):
            f0 = f0 == "Yes"
            g = f"f0G{sr}{emb_channels}.pth" if f0 else f"G{sr}{emb_channels}.pth"
            d = f"f0D{sr}{emb_channels}.pth" if f0 else f"D{sr}{emb_channels}.pth"

            return gr.Textbox.update(
                value=os.path.join(MODELS_DIR, "pretrained", g)
            ), gr.Textbox.update(value=os.path.join(MODELS_DIR, "pretrained", d))

        change_pretrained_options = {
            "fn": change_pretrained,
            "inputs": [
                target_sr,
                f0,
                embedding_channels,
            ],
            "outputs": [
                pre_trained_generator,
                pre_trained_discriminator,
            ],
        }

        target_sr.change(**change_pretrained_options)
        f0.change(**change_pretrained_options)
        embedding_channels.change(**change_pretrained_options)

        train_index_button.click(
            train_index_only,
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
                embedder_name,
                embedding_channels,
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
                pre_trained_generator,
                pre_trained_discriminator,
                embedder_name,
                embedding_channels,
                ignore_cache,
            ],
            outputs=[status],
        )
