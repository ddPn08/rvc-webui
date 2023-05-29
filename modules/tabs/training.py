import math
import os
import shutil
from multiprocessing import cpu_count

import gradio as gr

from lib.rvc.preprocessing import extract_f0, extract_feature, split
from lib.rvc.train import (create_dataset_meta, glob_dataset, train_index,
                           train_model)
from modules import models, utils
from modules.shared import MODELS_DIR, device, half_support
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
            recursive,
            multiple_speakers,
            speaker_id,
            gpu_id,
            num_cpu_process,
            norm_audio_when_preprocess,
            pitch_extraction_algo,
            run_train_index,
            reduce_index_size,
            maximum_index_size,
            embedder_name,
            embedding_channels,
            embedding_output_layer,
            ignore_cache,
        ):
            maximum_index_size = int(maximum_index_size)
            f0 = f0 == "Yes"
            norm_audio_when_preprocess = norm_audio_when_preprocess == "Yes"
            run_train_index = run_train_index == "Yes"
            reduce_index_size = reduce_index_size == "Yes"
            training_dir = os.path.join(MODELS_DIR, "training", "models", model_name)
            gpu_ids = [int(x.strip()) for x in gpu_id.split(",")] if gpu_id else []
            yield f"Training directory: {training_dir}"

            if os.path.exists(training_dir) and ignore_cache:
                shutil.rmtree(training_dir)

            os.makedirs(training_dir, exist_ok=True)

            datasets = glob_dataset(
                dataset_glob,
                speaker_id,
                multiple_speakers=multiple_speakers,
                recursive=recursive,
            )

            if len(datasets) == 0:
                raise Exception("No audio files found")

            yield "Preprocessing..."
            split.preprocess_audio(
                datasets,
                SR_DICT[target_sr],
                num_cpu_process,
                training_dir,
                norm_audio_when_preprocess,
                os.path.join(
                    MODELS_DIR,
                    "training",
                    "mute",
                    "0_gt_wavs",
                    f"mute{target_sr}.wav",
                ),
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
                int(embedding_channels),
                int(embedding_output_layer),
                gpu_ids,
            )

            out_dir = os.path.join(MODELS_DIR, "checkpoints")

            yield "Training index..."
            if run_train_index:
                if not reduce_index_size:
                    maximum_index_size = None
                train_index(
                    training_dir,
                    model_name,
                    out_dir,
                    int(embedding_channels),
                    num_cpu_process,
                    maximum_index_size,
                )

            yield "Training complete"

        def train_all(
            model_name,
            version,
            sampling_rate_str,
            f0,
            dataset_glob,
            recursive,
            multiple_speakers,
            speaker_id,
            gpu_id,
            num_cpu_process,
            norm_audio_when_preprocess,
            pitch_extraction_algo,
            batch_size,
            augment,
            augment_from_pretrain,
            augment_path,
            speaker_info_path,
            cache_batch,
            num_epochs,
            save_every_epoch,
            fp16,
            pre_trained_bottom_model_g,
            pre_trained_bottom_model_d,
            run_train_index,
            reduce_index_size,
            maximum_index_size,
            embedder_name,
            embedding_channels,
            embedding_output_layer,
            ignore_cache,
        ):
            batch_size = int(batch_size)
            num_epochs = int(num_epochs)
            maximum_index_size = int(maximum_index_size)
            f0 = f0 == "Yes"
            norm_audio_when_preprocess = norm_audio_when_preprocess == "Yes"
            run_train_index = run_train_index == "Yes"
            reduce_index_size = reduce_index_size == "Yes"
            training_dir = os.path.join(MODELS_DIR, "training", "models", model_name)
            gpu_ids = [int(x.strip()) for x in gpu_id.split(",")] if gpu_id else []

            if os.path.exists(training_dir) and ignore_cache:
                shutil.rmtree(training_dir)

            os.makedirs(training_dir, exist_ok=True)

            yield f"Training directory: {training_dir}"

            datasets = glob_dataset(
                dataset_glob,
                speaker_id,
                multiple_speakers=multiple_speakers,
                recursive=recursive,
            )

            if len(datasets) == 0:
                raise Exception("No audio files found")

            yield "Preprocessing..."
            split.preprocess_audio(
                datasets,
                SR_DICT[sampling_rate_str],
                num_cpu_process,
                training_dir,
                norm_audio_when_preprocess,
                os.path.join(
                    MODELS_DIR,
                    "training",
                    "mute",
                    "0_gt_wavs",
                    f"mute{sampling_rate_str}.wav",
                ),
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
                int(embedding_channels),
                int(embedding_output_layer),
                gpu_ids,
                None if len(gpu_ids) > 1 else device,
            )

            create_dataset_meta(training_dir, f0)

            yield "Training model..."

            print(f"train_all: emb_name: {embedder_name}")

            config = utils.load_config(
                version, training_dir, sampling_rate_str, embedding_channels, fp16
            )
            out_dir = os.path.join(MODELS_DIR, "checkpoints")

            if not augment_from_pretrain:
                augment_path = None
                speaker_info_path = None

            train_model(
                gpu_ids,
                config,
                training_dir,
                model_name,
                out_dir,
                sampling_rate_str,
                f0,
                batch_size,
                augment,
                augment_path,
                speaker_info_path,
                cache_batch,
                num_epochs,
                save_every_epoch,
                pre_trained_bottom_model_g,
                pre_trained_bottom_model_d,
                embedder_name,
                int(embedding_output_layer),
                False,
                None if len(gpu_ids) > 1 else device,
            )

            yield "Training index..."
            if run_train_index:
                if not reduce_index_size:
                    maximum_index_size = None
                train_index(
                    training_dir,
                    model_name,
                    out_dir,
                    int(embedding_channels),
                    num_cpu_process,
                    maximum_index_size,
                )

            yield "Training completed"

        with gr.Group():
            with gr.Box():
                with gr.Column():
                    with gr.Row().style():
                        with gr.Column():
                            model_name = gr.Textbox(label="Model Name")
                            ignore_cache = gr.Checkbox(label="Ignore cache")
                        with gr.Column():
                            dataset_glob = gr.Textbox(
                                label="Dataset glob", placeholder="data/**/*.wav"
                            )
                            recursive = gr.Checkbox(label="Recursive", value=True)
                            multiple_speakers = gr.Checkbox(
                                label="Multiple speakers", value=False
                            )
                            speaker_id = gr.Slider(
                                maximum=4,
                                minimum=0,
                                value=0,
                                step=1,
                                label="Speaker ID",
                            )

                    with gr.Row().style(equal_height=False):
                        version = gr.Radio(
                            choices=["v1", "v2"],
                            value="v2",
                            label="Model version",
                        )
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
                        embedding_name = gr.Radio(
                            choices=list(models.EMBEDDINGS_LIST.keys()),
                            value="contentvec",
                            label="Using phone embedder",
                        )
                        embedding_channels = gr.Radio(
                            choices=["256", "768"],
                            value="768",
                            label="Embedding channels",
                        )
                        embedding_output_layer = gr.Radio(
                            choices=["9", "12"],
                            value="12",
                            label="Embedding output layer",
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
                            value=math.ceil(cpu_count() / 2),
                            label="Number of CPU processes",
                        )
                        norm_audio_when_preprocess = gr.Radio(
                            choices=["Yes", "No"],
                            value="Yes",
                            label="Normalize audio volume when preprocess",
                        )
                        pitch_extraction_algo = gr.Radio(
                            choices=["dio", "harvest"],
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
                        fp16 = gr.Checkbox(
                            label="FP16", value=half_support, disabled=not half_support
                        )
                    with gr.Row().style(equal_height=False):
                        augment = gr.Checkbox(label="Augment", value=False)
                        augment_from_pretrain = gr.Checkbox(label="Augment From Pretrain", value=False)
                        augment_path = gr.Textbox(
                            label="Pre trained generator path (pth)",
                            value="file is not prepared"
                        )
                        speaker_info_path = gr.Textbox(
                            label="speaker info path (npy)",
                            value="file is not prepared"
                        )
                    with gr.Row().style(equal_height=False):
                        pre_trained_generator = gr.Textbox(
                            label="Pre trained generator path",
                            value=os.path.join(
                                MODELS_DIR, "pretrained", "v2", "f0G40k.pth"
                            ),
                        )
                        pre_trained_discriminator = gr.Textbox(
                            label="Pre trained discriminator path",
                            value=os.path.join(
                                MODELS_DIR, "pretrained", "v2", "f0D40k.pth"
                            ),
                        )
                    with gr.Row().style(equal_height=False):
                        run_train_index = gr.Radio(
                            choices=["Yes", "No"],
                            value="Yes",
                            label="Train Index",
                        )
                        reduce_index_size = gr.Radio(
                            choices=["Yes", "No"],
                            value="No",
                            label="Reduce index size with kmeans",
                        )
                        maximum_index_size = gr.Number(
                            value=10000, label="maximum index size"
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
                recursive,
                multiple_speakers,
                speaker_id,
                gpu_id,
                num_cpu_process,
                norm_audio_when_preprocess,
                pitch_extraction_algo,
                run_train_index,
                reduce_index_size,
                maximum_index_size,
                embedding_name,
                embedding_channels,
                embedding_output_layer,
                ignore_cache,
            ],
            outputs=[status],
        )

        train_all_button.click(
            train_all,
            inputs=[
                model_name,
                version,
                target_sr,
                f0,
                dataset_glob,
                recursive,
                multiple_speakers,
                speaker_id,
                gpu_id,
                num_cpu_process,
                norm_audio_when_preprocess,
                pitch_extraction_algo,
                batch_size,
                augment,
                augment_from_pretrain,
                augment_path,
                speaker_info_path,
                cache_batch,
                num_epochs,
                save_every_epoch,
                fp16,
                pre_trained_generator,
                pre_trained_discriminator,
                run_train_index,
                reduce_index_size,
                maximum_index_size,
                embedding_name,
                embedding_channels,
                embedding_output_layer,
                ignore_cache,
            ],
            outputs=[status],
        )
