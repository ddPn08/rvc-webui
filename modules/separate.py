import os
from typing import *

import tqdm
from pydub import AudioSegment
from pydub.silence import split_on_silence


def separate_audio(
    input: str,
    output: str,
    silence_thresh: int,
    min_silence_len: int = 1000,
    keep_silence: int = 100,
    margin: int = 0,
    padding: bool = False,
    min: Optional[int] = None,
    max: Optional[int] = None,
):
    if os.path.isfile(input):
        input = [input]
    elif os.path.isdir(input):
        input = [os.path.join(input, f) for f in os.listdir(input)]
    else:
        raise ValueError("input must be a file or directory")

    os.makedirs(output, exist_ok=True)

    for file in input:
        if os.path.splitext(file)[1] == ".mp3":
            audio = AudioSegment.from_mp3(file)
        elif os.path.splitext(file)[1] == ".wav":
            audio = AudioSegment.from_wav(file)
        elif os.path.splitext(file)[1] == ".flac":
            audio = AudioSegment.from_file(file, "flac")
        else:
            raise ValueError(
                "Invalid file format. Only MP3 and WAV files are supported."
            )

        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
        )

        output_chunks: List[AudioSegment] = []

        so_short = None

        for chunk in tqdm.tqdm(chunks):
            if so_short is not None:
                chunk = so_short + chunk
                so_short = None
            if min is None or len(chunk) > min:
                if max is not None and len(chunk) > max:
                    sub_chunks = [
                        chunk[i : i + max + margin]
                        for i in range(0, len(chunk) - margin, max)
                    ]

                    if len(sub_chunks[-1]) < min:
                        if padding and len(sub_chunks) > 2:
                            output_chunks.extend(sub_chunks[0:-2])
                            output_chunks.append(sub_chunks[-2] + sub_chunks[-1])
                        else:
                            output_chunks.extend(sub_chunks[0:-1])
                    else:
                        output_chunks.extend(sub_chunks)
                else:
                    output_chunks.append(chunk)
            else:
                if so_short is None:
                    so_short = chunk
                else:
                    so_short += chunk
        basename = os.path.splitext(os.path.basename(file))[0]

        for i, chunk in enumerate(output_chunks):
            filepath = os.path.join(output, f"{basename}_{i}.wav")
            chunk.export(filepath, format="wav")
