#!/usr/bin/env python3
"""Precompute Wav2Vec features."""

import os
import json
from pathlib import Path
from tempfile import mkstemp
from multiprocessing import cpu_count

import tqdm
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from copy import deepcopy

from models import load_pretrained_wav2vec
from data import PreprocessDataset
from data.feature_extract import FeatureExtractor

def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dirs", type=str, nargs="+")
    parser.add_argument("feature_name", type=str)
    parser.add_argument("wav2vec_path", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--trim_method", choices=["librosa", "vad"], default="vad")
    parser.add_argument("--n_workers", type=int, default=cpu_count())

    parser.add_argument("--sample_rate", type=int, default=16000)

    return vars(parser.parse_args())


def main(
    data_dirs,
    feature_name,
    wav2vec_path,
    out_dir,
    trim_method,
    n_workers,
    sample_rate,
    **kwargs,
):
    """Main function."""

    out_dir_path = Path(out_dir)

    if out_dir_path.exists():
        assert out_dir_path.is_dir()
    else:
        out_dir_path.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PreprocessDataset(
        data_dirs,
        trim_method,
        sample_rate
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=n_workers
    )


    speaker_infos = {}
    speaker_infos['feature_name'] = feature_name

    pbar = tqdm.tqdm(total=len(dataset), ncols=0)
    mapping = {'apc': 'fbank', 'timit_posteriorgram': 'fbank', 'cpc': 'cpc_mel', 'wav2vec2': 'wav2vec2_mel'}
    feat_extractor = FeatureExtractor(feature_name, wav2vec_path, device)
    mel_extractor = FeatureExtractor(mapping[feature_name], wav2vec_path, device)
    for speaker_name, audio_path, wav in dataloader:
        if wav.size(-1) < 10:
            continue

        wav = wav.to(device)
        speaker_name = speaker_name[0]
        audio_path = audio_path[0]

        
        with torch.no_grad():
            feat = feat_extractor.get_feature(wav)[0]
            mel = mel_extractor.get_feature(wav)[0]
        fd, temp_file = mkstemp(suffix=".tar", prefix="utterance-", dir=out_dir_path)
        torch.save({"feat": feat.detach().cpu(), "mel": mel.detach().cpu()}, temp_file)
        os.close(fd)

        if speaker_name not in speaker_infos.keys():
            speaker_infos[speaker_name] = []

        speaker_infos[speaker_name].append(
            {
                "feature_path": Path(temp_file).name,
                "audio_path": audio_path,
                "mel_len": len(mel),
            }
        )

        pbar.update(dataloader.batch_size)

    with open(out_dir_path / "metadata.json", "w") as f:
        json.dump(speaker_infos, f, indent=2)


if __name__ == "__main__":
    main(**parse_args())
