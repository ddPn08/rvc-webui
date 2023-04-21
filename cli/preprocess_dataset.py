import sys
import os

sys.path.append(os.curdir)

from typing import *
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor
import argparse
import glob
import operator
from modules.training.preprocess import PreProcess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_glob", type=str, required=True)
    parser.add_argument("--training_dir", type=str, required=True)
    parser.add_argument("--sampling_rate", type=int, required=True)
    parser.add_argument("--num_processes", type=int, required=True)
    parser.add_argument("--is_normalize", action="store_true")
    return parser.parse_args()


def create_dataset(dataset_glob: str):
    files = glob.glob(dataset_glob, recursive=True)
    return [(file, 0) for file in files]


def process_data(sr, dir, datasets: List[Tuple[str, int]], is_normalize: bool):
    pp = PreProcess(sr, dir)
    print(f"Process {os.getpid()} started")
    for index, file in enumerate(sorted(datasets, key=operator.itemgetter(0))):
        pp.pipeline(file[1], file[0], index, is_normalize)


def main(args: argparse.Namespace):
    pp = PreProcess(args.sampling_rate, args.training_dir)
    os.makedirs(os.path.join(pp.gt_wavs_dir, f"{0:05}"), exist_ok=True)
    os.makedirs(os.path.join(pp.wavs16k_dir, f"{0:05}"), exist_ok=True)

    datasets = create_dataset(args.dataset_glob)

    with ProcessPoolExecutor() as thread:
        print(f"Number of processes: {args.num_processes}")
        thread.map(
            process_data,
            [
                (
                    args.sampling_rate,
                    args.training_dir,
                    datasets[i :: args.num_processes],
                    args.is_normalize,
                )
                for i in range(args.num_processes)
            ],
        )

    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
