import os
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from tools import language_corpus as lc
import config
from tools import audio
from tools.utils import double_vad, rms_normalize, normalize_text, get_bucket_duration, plot_spectrogram, plot_waveforms, mean_norm
import pandas as pd
import json

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
        if x is None:
            return None
        
        spec = self.mel_spectrogram(x)
        spec = np.log10(spec + 1e-8)
        spec = mean_norm(spec)
        return spec

class AudioInfo():
    def __init__(self, tsv_file = config.OUTPUT_DIR / f'{config.LANGUAGE}.tsv', output_dir = config.OUTPUT_DIR / 'spectrograms', sr = 16000, log_mel_spec = LogMelSpectrogram()):
        self.sr = sr
        self.log_mel_spec = log_mel_spec
        self.tsv_file = tsv_file
        self.output_dir = output_dir
        self.metadata = {
            'durations': [],
            'bucket_duration': [],
            'num_frames': [],
            'orig_durations': []
        }
        self.processed_files = {
            'success': 0,
            'fail': 0
        }
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def preprocess(self):
        if not os.path.exists(self.tsv_file):
            raise FileNotFoundError(f"{self.tsv_file} does not exist")

        df = pd.read_csv(self.tsv_file, sep='\t')
        total_rows = len(df)
        for idx, rows in df.iterrows():
            current_row = idx + 1  
            print(f"Processing audio {current_row}/{total_rows}", end='\r')
            file_name = rows['file_name']
            audio_file = config.WAVS_PATH / f"{file_name}.wav"

            loaded = self.load_audio(audio_file, file_name)
            if not loaded:
                self.processed_files['fail'] += 1
                for key in self.metadata.keys():
                    self.metadata[key].append(0)
                continue

            self.processed_files['success'] += 1
            spec, orig_dur, trimmed_duration, bucket, num_frames = loaded
            self.metadata['durations'].append(trimmed_duration)
            self.metadata['bucket_duration'].append(bucket)
            self.metadata['num_frames'].append(num_frames) 
            self.metadata['orig_durations'].append(orig_dur)
        print(f"Processed {self.processed_files['success']} rows | Failed {self.processed_files['fail']} rows")

        df['orig_duration'] = self.metadata['orig_durations']
        df['duration'] = self.metadata['durations']
        df['bucket_duration'] = self.metadata['bucket_duration']
        df['num_frames'] = self.metadata['num_frames']
        df.to_csv(self.tsv_file, sep='\t',index=False)

    def load_audio(self, audio_file, file_name):
        info = torchaudio.info(audio_file)
        waveform, sr = torchaudio.load(audio_file)

        if waveform is None or waveform.numel() == 0:
            print(f"Skipping file {audio_file} due to invalid or empty waveform.")
            return False

        if sr != self.sr:
            waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)
        # trimmed_waveform = double_vad(waveform)

        # if trimmed_waveform is None or trimmed_waveform.shape[1] <= 0 or trimmed_waveform.numel() == 0:
        #     print(f"Skipping trimmed audio due to invalid or empty waveform.")
        #     return False

        orig_duration = info.num_frames / self.sr
        # trimmed_duration = trimmed_waveform.shape[1] / self.sr
        bucket = get_bucket_duration(waveform)
    
        spec = self.log_mel_spec(waveform)
        if spec is None:
            return False
        
        torch.save(spec, self.output_dir / f"{file_name}.pt")
        return spec, orig_duration, orig_duration, bucket, info.num_frames

class AudioTranscriptionTSV():
    def __init__(self, valid_files = ['clean', 'train', 'dev', 'test']):
        self.valid_files = valid_files
        self.save_file = config.OUTPUT_DIR / f'{config.LANGUAGE}.tsv'
        self.data = {
            'file_name': [],
            'transcription': []
        }
        self.preprocess_count = {
            'success': 0,
            'fail': 0
        }
        if not os.path.exists(config.OUTPUT_DIR):
            os.makedirs(config.OUTPUT_DIR)

        if not os.path.exists(config.WAVS_PATH):
            os.makedirs(config.WAVS_PATH)

        if not os.path.exists(self.save_file):
            columns = ['file_name', 'orig_duration', 'duration', 'bucket_duration', 'num_frames', 'transcription']
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.save_file, sep='\t',index=False)

    def preprocess_tsv(self):
        for file in self.valid_files:
            file_path = config.COMMON_VOICE_PATH / f"{file}.tsv"
            if os.path.exists(file_path):
                self.load_file(file_path)


        self.save_tsv()
        print(f"Saved TSV file to {self.save_file} | Success: {self.preprocess_count['success']} | Fail: {self.preprocess_count['fail']}")

        print(f"Generating transcription list")
        self.generate_transcription_list()
        print(f"Generated transcription list")

        print(f"Training language corpus")
        lc.train()

    def save_tsv(self):
        print(f"Saving TSV file to {self.save_file}")
        df = pd.read_csv(self.save_file, sep='\t')
        df['file_name'] = self.data['file_name']
        df['transcription'] = self.data['transcription']
        df.to_csv(self.save_file, sep='\t',index=False)

    def load_file(self, tsv_file):
        df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
        row_preprocessed = 0
        total_rows = len(df)

        for idx, rows in df.iterrows():
            self.preprocess_count['fail'] += 1
            file_name = rows['path'].replace('.mp3', '')
            transcription = rows['sentence']
            print(f"Processing row {idx + 1}/{total_rows}", end='\r')

            if not isinstance(transcription, str):
                df.drop(idx)
                print(f"Skipping file {file_name} due to invalid transcription.")
                continue

            if len(transcription) < 3:
                print(f"Skipping file {file_name} due to short transcription.")
                continue

            transcription = normalize_text(transcription)
            audio_file = config.COMMON_VOICE_PATH / 'clips' / f"{file_name}.mp3"
            wav_file = config.WAVS_PATH / f"{file_name}.wav"

            if not os.path.exists(audio_file):
                print(f"{audio_file.relative_to(Path.cwd())} does not exist")
                continue
            
            if not os.path.exists(wav_file):
                audio.to_wav(audio_file, wav_file)
            
            self.data['file_name'].append(file_name)
            self.data['transcription'].append(transcription)
            self.preprocess_count['fail'] -= 1
            self.preprocess_count['success'] += 1
            row_preprocessed += 1

            if row_preprocessed >= 75000:
                break

        print(f"Processed {self.preprocess_count['success']} rows | Failed {self.preprocess_count['fail']} rows")

    def generate_transcription_list(self):
        print(f"Generating transcription list")

        df = pd.read_csv(self.save_file, sep='\t')
        transcription_list = df['transcription'].tolist()

        transcription_filename = config.OUTPUT_DIR / f'{config.LANGUAGE}_sentences.txt'
        os.makedirs(transcription_filename.parent, exist_ok=True)
        with open(transcription_filename,'w', encoding="utf-8") as f:
            for transcription in transcription_list:
                f.write(transcription + '\n')

        print(f"Generated transcription list - {transcription_filename}")

class BucketAudio():
    def __init__(self):
        self.output_dir = config.OUTPUT_DIR / 'buckets'
        self.tsv_file = config.OUTPUT_DIR / f'{config.LANGUAGE}.tsv'
        self.df = pd.read_csv(self.tsv_file, sep='\t')
        self.buckets = {}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def group_duration(self):
        df = pd.read_csv(self.tsv_file, sep='\t')
        for idx, rows in df.iterrows():
            
            bucket_duration = rows['bucket_duration']
            
            if bucket_duration not in self.buckets:
                self.buckets[bucket_duration] = []

            self.buckets[bucket_duration].append({
                "file_name": rows["file_name"],
                "transcription": rows['transcription'],
                "bucket_duration": rows['bucket_duration'],
                "orig_duration": rows['orig_duration'],
                "duration": rows['duration']
            })

    def save_buckets(self):
        for keys, item in self.buckets.items():
            bucket_file = self.output_dir / f"bucket_{keys}.json"

            with open(bucket_file, 'w') as f:
                json.dump(item, f, indent=2)

        print(f"[✓] Saved {len(self.buckets.keys())} buckets to: {self.output_dir}")
    
    def load_buckets(self):
        data = {}
        for bucket_file in os.listdir(self.output_dir):
            if bucket_file.endswith('.json'):
                with open(os.path.join(self.output_dir, bucket_file), 'r') as f:
                    data[bucket_file.replace('.json', '').replace('bucket_', '')] = json.load(f)

        print(f"[✓] Loaded {len(data.keys())} buckets from: {self.output_dir}")
        return data

    def init(self):
        self.group_duration()
        self.save_buckets()

def preprocess():
    # AudioTranscriptionTSV().generate_transcription_list()
    # print(f"Preprocessing transcriptions")
    # AudioTranscriptionTSV().preprocess_tsv()
    # print(f"Preprocessing audio")
    # AudioInfo().preprocess()
    # print(f"Preprocessing buckets")
    # BucketAudio().init()

    lc.train()
    
if __name__ == '__main__':
    preprocess()

