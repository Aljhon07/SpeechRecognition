import os
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pydantic.v1.parse import load_file
from tools import language_corpus as lc
import config
from tools import audio
from tools.utils import double_vad, rms_normalize, normalize_text, get_bucket_duration, plot_spectrogram, \
    plot_waveforms, mean_norm
import pandas as pd

class LogMelSpectrogram(nn.Module):
    def __init__(self, n_fft=400, hop_length=160, win_length=400, n_mels=80, sr=16000):
        super(LogMelSpectrogram, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sr = sr
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
    def forward(self, x):
        x = rms_normalize(x)
        spec = self.mel_spectrogram(x)
        spec = np.log10(spec + 1e-8)
        spec = mean_norm(spec)
        return spec

class AudioInfo():
    def __init__(self, tsv_file = config.OUTPUT_PATH / f'{config.LANGUAGE}.tsv', output_path = config.OUTPUT_PATH / 'spectrogram', sr = 16000, log_mel_spec = LogMelSpectrogram()):
        self.sr = sr
        self.log_mel_spec = log_mel_spec
        self.tsv_file = tsv_file
        self.output_path = output_path
        self.metadata = {
            'durations': [],
            'duration_buckets': [],
            'num_frames': []
        }
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def preprocess(self):
        if not os.path.exists(self.tsv_file):
            raise FileNotFoundError(f"{self.tsv_file} does not exist")

        df = pd.read_csv(self.tsv_file, sep='\t')
        for idx, rows in df.iterrows():
            file_name = rows['file_name']
            audio_file = config.WAVS_PATH / f"{file_name}.wav"
            if not audio_file.exists():
                print(f"{audio_file.relative_to(Path.cwd())} does not exist")
                metadata.append((0, 0, 0))
                continue

            spec, trimmed_duration, bucket, num_frames = self.load_audio(audio_file, file_name)
            self.metadata['durations'].append(trimmed_duration)
            self.metadata['duration_buckets'].append(bucket)
            self.metadata['num_frames'].append(num_frames)

        df['duration'] = self.metadata['durations']
        df['duration_bucket'] = self.metadata['duration_buckets']
        df['num_frames'] = self.metadata['num_frames']
        df.to_csv(self.tsv_file, sep='\t',index=False)

    def load_audio(self, audio_file, file_name):
        info = torchaudio.info(audio_file)
        waveform, sr = torchaudio.load(audio_file)
        if sr != self.sr:
            waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)

        trimmed_waveform = double_vad(waveform)

        orig_duration = info.num_frames / self.sr
        trimmed_duration = trimmed_waveform.shape[1] / self.sr
        bucket = get_bucket_duration(trimmed_waveform)
        spec = self.log_mel_spec(trimmed_waveform)
        print(
            f"Spec Info: {spec.shape} | Min: {spec.min()} | Max: {spec.max()} | Mean: {spec.mean()} | Std: {spec.std()}")
        print(f"File: {file_name} | Trimmed Duration: {trimmed_duration} | Bucket: {bucket}")

        torch.save(spec, self.output_path / f"{file_name}.pt")

        # plot_waveforms(waveform, trimmed_waveform)
        # plot_spectrogram(spec, spec)
        return spec, trimmed_duration, bucket, info.num_frames

class AudioTranscriptionTSV():
    def __init__(self, valid_files = ['validated', 'clean']):
        self.valid_files = valid_files
        self.save_file = config.OUTPUT_PATH / f'{config.LANGUAGE}.tsv'
        self.data = {
            'file_name': [],
            'transcription': []
        }

        if not os.path.exists(config.OUTPUT_PATH):
            os.makedirs(config.OUTPUT_PATH)

        if not os.path.exists(config.WAVS_PATH):
            os.makedirs(config.WAVS_PATH)

        columns = ['file_name', 'duration', 'duration_bucket', 'num_frames', 'transcription']
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.save_file, sep='\t',index=False)

    def preprocess_file(self):
        for file in self.valid_files:
            file_path = config.COMMON_VOICE_PATH / f"{file}.tsv"
            if os.path.exists(file_path):
                self.load_file(file_path)

        self.save_tsv()
        self.generate_transcription_list()
        lc.train()

    def save_tsv(self):
        df = pd.read_csv(self.save_file, sep='\t')
        df['file_name'] = self.data['file_name']
        df['transcription'] = self.data['transcription']
        df.to_csv(self.save_file, sep='\t',index=False)

    def load_file(self, tsv_file):
        df = pd.read_csv(tsv_file, sep='\t')
        for idx, rows in df.iterrows():
            file_name = rows['path']

            audio_file = config.COMMON_VOICE_PATH / 'clips' / file_name
            if not audio_file.exists():
                print(f"{audio_file.relative_to(Path.cwd())} does not exist")
                continue

            file_name = file_name.replace('.mp3', '')
            audio.to_wav(audio_file, config.WAVS_PATH / f"{file_name}.wav")
            transcription = normalize_text(rows['sentence'])

            self.data['file_name'].append(file_name)
            self.data['transcription'].append(transcription)
            if idx == 2:
                break

    def generate_transcription_list(self):
        transcription_filename = config.OUTPUT_PATH / f'{config.LANGUAGE}_sentences.txt'
        os.makedirs(transcription_filename.parent, exist_ok=True)
        with open(transcription_filename, 'w') as f:
            for transcription in self.data['transcription']:
                f.write(transcription + '\n')

        print(f"Generated {transcription_filename}")

def prepocess():
    AudioTranscriptionTSV().preprocess_file()
    AudioInfo().preprocess()

if __name__ == '__main__':
    preprocess()