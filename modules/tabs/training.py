import os
from multiprocessing import cpu_count

import faiss
import gradio as gr
import numpy as np

from modules import utils
from modules.shared import MODELS_DIR
from modules.training.extract import extract_f0, extract_feature
from modules.training.preprocess import preprocess_trainset
from modules.training.train import run_training

SR_DICT = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def title():
    return "Training"


def tab():
    def train(
        model_name,
        target_sr,
        has_pitch_guidance,
        dataset_dir,
        speaker_id,
        gpu_id,
        num_cpu_process,
        pitch_extraction_algo,
        batch_size,
        num_epochs,
        save_every_epoch,
        pre_trained_bottom_model_g,
        pre_trained_bottom_model_d,
    ):
        has_pitch_guidance = has_pitch_guidance == "Yes"
        training_dir = os.path.join(MODELS_DIR, "training", model_name)
        preprocess_trainset(
            dataset_dir, SR_DICT[target_sr], num_cpu_process, model_name
        )
        extract_f0(model_name, num_cpu_process, pitch_extraction_algo)
        extract_feature(model_name)

        gt_wavs_dir = os.path.join(training_dir, "0_gt_wavs")
        co256_dir = os.path.join(training_dir, "3_feature256")

        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(co256_dir)]
        )

        if has_pitch_guidance:
            f0_dir = os.path.join(training_dir, "2a_f0")
            f0nsf_dir = os.path.join(training_dir, "2b_f0nsf")
            names = (
                names
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
            )

        opt = []

        for name in names:
            if has_pitch_guidance:
                gt_wav_path = os.path.join(gt_wavs_dir, f"{name}.wav")
                co256_path = os.path.join(co256_dir, f"{name}.npy")
                f0_path = os.path.join(f0_dir, f"{name}.wav.npy")
                f0nsf_path = os.path.join(f0nsf_dir, f"{name}.wav.npy")
                opt.append(
                    f"{gt_wav_path}|{co256_path}|{f0_path}|{f0nsf_path}|{speaker_id}"
                )
            else:
                gt_wav_path = os.path.join(gt_wavs_dir, f"{name}.wav")
                co256_path = os.path.join(co256_dir, f"{name}.npy")
                opt.append(f"{gt_wav_path}|{co256_path}|{speaker_id}")

        if has_pitch_guidance:
            mute_gt_wav = os.path.join(
                MODELS_DIR, "training", "mute", "0_gt_wavs", f"mute{target_sr}.wav"
            )
            mute_co256 = os.path.join(
                MODELS_DIR, "training", "mute", "3_feature256", "mute.npy"
            )
            mute_f0 = os.path.join(
                MODELS_DIR, "training", "mute", "2a_f0", f"mute.wav.npy"
            )
            mute_f0nsf = os.path.join(
                MODELS_DIR, "training", "mute", "2b_f0nsf", f"mute.wav.npy"
            )
            opt.append(
                f"{mute_gt_wav}|{mute_co256}|{mute_f0}|{mute_f0nsf}|{speaker_id}"
            )
        else:
            mute_gt_wav = os.path.join(
                MODELS_DIR, "training", "mute", "0_gt_wavs", f"mute{target_sr}.wav"
            )
            mute_co256 = os.path.join(
                MODELS_DIR, "training", "mute", "3_feature256", "mute.npy"
            )
            opt.append(f"{mute_gt_wav}|{mute_co256}|{speaker_id}")
        with open(os.path.join(training_dir, "filelist.txt"), "w") as f:
            f.write("\n".join(opt))

        run_training(
            gpu_id.split(","),
            model_name,
            target_sr,
            1 if has_pitch_guidance else 0,
            batch_size,
            num_epochs,
            save_every_epoch,
            pre_trained_bottom_model_g,
            pre_trained_bottom_model_d,
        )

        feature_dir = os.path.join(training_dir, "3_feature256")
        npys = []
        listdir_res = list(os.listdir(feature_dir))
        for name in sorted(listdir_res):
            phone = np.load("%s/%s" % (feature_dir, name))
            npys.append(phone)
        big_npy = np.concatenate(npys, 0)
        np.save(os.path.join(training_dir, "total_fea.npy"), big_npy)
        n_ivf = big_npy.shape[0] // 39
        index = faiss.index_factory(256, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = int(np.power(n_ivf, 0.3))
        index.train(big_npy)
        index.add(big_npy)
        faiss.write_index(
            index,
            os.path.join(
                training_dir, f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}.index"
            ),
        )

        return "Training completed"

    with gr.Group():
        with gr.Box():
            with gr.Column():
                with gr.Row().style(equal_height=False):
                    model_name = gr.Textbox(label="Model Name")
                    target_sr = gr.Radio(
                        choices=["32k", "40k", "48k"],
                        value="40k",
                        label="Target sampling rate",
                    )
                    has_pitch_guidance = gr.Radio(
                        choices=["Yes", "No"],
                        value="Yes",
                        label="Model has pitch guidance",
                    )

                with gr.Row().style(equal_height=False):
                    dataset_dir = gr.Textbox(label="Dataset directory")
                    speaker_id = gr.Slider(
                        maximum=4, minimum=0, value=0, step=1, label="Speaker ID"
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
                        minimum=1, maximum=64, value=16, step=1, label="Batch size"
                    )
                    num_epochs = gr.Slider(
                        minimum=1,
                        maximum=1000,
                        value=100,
                        step=1,
                        label="Number of epochs",
                    )
                    save_every_epoch = gr.Slider(
                        minimum=1,
                        maximum=1000,
                        value=10,
                        step=1,
                        label="Save every epoch",
                    )
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
                    train_button = gr.Button("Train")

    train_button.click(
        train,
        inputs=[
            model_name,
            target_sr,
            has_pitch_guidance,
            dataset_dir,
            speaker_id,
            gpu_id,
            num_cpu_process,
            pitch_extraction_algo,
            batch_size,
            num_epochs,
            save_every_epoch,
            pre_trained_bottom_model_g,
            pre_trained_bottom_model_d,
        ],
        outputs=[status],
    )
