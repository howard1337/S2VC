import torch
from functools import partial
from multiprocessing import Pool, cpu_count
from models import load_pretrained_wav2vec
from data import log_mel_spectrogram


class FeatureExtractor:
    def __init__(self, feature_name, wav2vec2_path=None):
        if (
            feature_name == "apc"
            or feature_name == "cpc"
            or feature_name == "timit_posteriorgram"
            or feature_name == "fbank"
        ):
            self.extractor = (
                torch.hub.load("s3prl/s3prl", feature_name).eval().cuda()
            )
            self.mode = 1
        elif feature_name == "wav2vec2":
            self.extractor = load_pretrained_wav2vec(wav2vec2_path).eval().cuda()
            self.mode = 2
        elif feature_name == "wav2vec2_mel":
            self.extractor = partial(
                log_mel_spectrogram,
                preemph=0.97,
                sample_rate=16000,
                n_mels=80,
                n_fft=400,
                hop_length=320,
                win_length=400,
                f_min=0,
                center=False,
            )
            self.mode = 3
        elif feature_name == "cpc_mel":
            self.extractor = partial(
                log_mel_spectrogram,
                preemph=0.97,
                sample_rate=16000,
                n_mels=80,
                n_fft=465,
                hop_length=160,
                win_length=465,
                f_min=80,
                center=True,
            )
            self.mode = 3
        else:
            print(feature_name)
            print(
                "Please use timit_posteriorgram, apc, wav2vec2, cpc, wav2vec2_mel, cpc_mel, or fbank"
            )
            exit()

    def get_feature(self, wavs):
        if self.mode == 1:
            return self.extractor(wavs)
        elif self.mode == 2:
            feats = []
            for wav in wavs:
                feat = self.extractor.extract_features(wav.unsqueeze(0), None)[0].squeeze(0)
                feats.append(feat)
        elif self.mode == 3:
            wavs = [wav.cpu().numpy() for wav in wavs]
            feats = [self.extractor(wav) for wav in wavs]
            feats = [torch.FloatTensor(feat).cuda() for feat in feats]
            return feats

        return feats
