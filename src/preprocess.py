import os
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from tools import language_corpus as lc
import config
from tools import audio
from tools.utils import rms_normalize, normalize_text, get_bucket_duration, mean_norm
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading

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
        self.resample = torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=self.sr)
    def forward(self, x):
        x = rms_normalize(x)
        spec = self.mel_spectrogram(x)
        spec = np.log10(spec + 1e-8)
        spec = mean_norm(spec)
        return spec

class  AudioInfo():
    def __init__(self, output_dir = config.OUTPUT_DIR / 'spectrograms', sr = 16000, log_mel_spec = LogMelSpectrogram()):
        self.sr = sr
        self.log_mel_spec = log_mel_spec
        self.tsv_file = None
        self.output_dir = output_dir
        self.processed_files = {
            'success': 0,
            'fail': 0
        }
        self.file_lock = threading.Lock()
        self.count_lock = threading.Lock()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def preprocess(self, tsv_file):
        self.tsv_file = config.OUTPUT_DIR / tsv_file + ".tsv"
        if not os.path.exists(self.tsv_file):
            raise FileNotFoundError(f"{self.tsv_file} does not exist")

        df = pd.read_csv(self.tsv_file, sep='\t')
        total_rows = len(df)
        progress = tqdm(total=total_rows, desc="Processing files")

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for idx, rows in df.iterrows():
                futures.append(executor.submit(self.process_row, idx, rows))

            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, result = future.result()
                    with self.file_lock:
                        df.at[idx, 'orig_duration'] = result['orig_dur']
                        df.at[idx, 'duration'] = result['trimmed_duration']
                        df.at[idx, 'bucket_duration'] = result['bucket']
                        df.at[idx, 'num_frames'] = result['num_frames']
                    with self.count_lock:
                        if result['success']:
                            self.processed_files['success'] += 1
                        else:
                            self.processed_files['fail'] += 1

                    progress.set_postfix({
                    "Success": self.processed_files['success'],
                    "Fail": self.processed_files['fail'],
                })
                    
                    progress.update(1)
                except Exception as e:
                    tqdm.write(f"Error processing file: {e}")

        df.to_csv(self.tsv_file, sep='\t', index=False)

    def process_row(self, idx, rows):
        file_name = rows['file_name']
        audio_file = config.WAVS_PATH / f"{file_name}.wav"
        metadata = self.load_audio(audio_file, file_name)

        if metadata is not None:
            orig_dur, trimmed_duration, bucket, num_frames = metadata

            return idx,  {"success": True, 'orig_dur': orig_dur, 'trimmed_duration': trimmed_duration, 'bucket': bucket, 'num_frames': num_frames}
        else:
            return idx,  {"success": False, 'orig_dur': 0, 'trimmed_duration': 0, 'bucket': 0, 'num_frames': 0}


    def load_audio(self, audio_file, file_name):
        info = torchaudio.info(audio_file)
        waveform, sr = torchaudio.load(audio_file)

        if waveform is None or waveform.numel() == 0:
            print(f"Skipping file {audio_file} due to invalid or empty waveform.")
            return None

        if sr != self.sr:
            waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)

        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            print(f"Skipping {audio_file}: invalid values.")
            return None
        
        orig_duration = info.num_frames / self.sr
        bucket = get_bucket_duration(waveform)
    
        rms = rms_normalize(waveform)
        if rms is None:
            return None
        
        return orig_duration, orig_duration, bucket, info.num_frames

class AudioTranscriptionTSV():
    def __init__(self ):
        self.save_file = config.OUTPUT_DIR / f'{config.LANGUAGE}.tsv'
        self.data = []
        self.preprocess_count = {
            'success': 0,
            'fail': 0,
            'missing': 0,
            'converted': 0,
            'error': 0,
            'skip': 0,
            'warning': 0
        }
        self.file_lock = threading.Lock()
        self.count_lock = threading.Lock()

        if not os.path.exists(config.OUTPUT_DIR):
            os.makedirs(config.OUTPUT_DIR)

        if not os.path.exists(config.WAVS_PATH):
            os.makedirs(config.WAVS_PATH)

        if not os.path.exists(self.save_file):
            columns = ['file_name', 'orig_duration', 'duration', 'bucket_duration', 'num_frames', 'transcription']
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.save_file, sep='\t',index=False)

    def preprocess_tsv(self, file_name = 'clean'):
        file_path = config.COMMON_VOICE_PATH / f"{file_name}.tsv"
        self.save_file = config.OUTPUT_DIR / f'{file_name}.tsv'
        self.load_file(file_path)

        self.save_tsv()
        print(f"Saved TSV file to {self.save_file} | Success: {self.preprocess_count['success']} | Fail: {self.preprocess_count['fail']}")


    def save_tsv(self):
        print(f"Saving TSV file to {self.save_file}")
        df = pd.read_csv(self.save_file, sep='\t')

        new_data_df = pd.DataFrame(self.data, columns=['file_name', 'transcription'])

        df[['file_name', 'transcription']] = new_data_df
        df.to_csv(self.save_file, sep='\t',index=False)

    def load_file(self, tsv_file):
        df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
        total_rows = len(df)
        progress = tqdm(total=total_rows, desc="Processing files")
        
        if not df['path'].duplicated().any():
            tqdm.write("All paths are unique!")
        else:
            raise ValueError("Duplicate paths found in the TSV file.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for idx, rows in df.iterrows():
                futures.append(executor.submit(self.process_row, idx, rows))

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        with self.count_lock:
                            self.preprocess_count['success'] += result['success']
                            self.preprocess_count['fail'] += result['fail']
                            self.preprocess_count['skip'] += result['skip']
                            self.preprocess_count['converted'] += result['converted']
                            self.preprocess_count['error'] += result['error']
                            self.preprocess_count['missing'] += result['missing']
                            self.preprocess_count['warning'] += result['warning']
                            
                        progress.set_postfix({
                        "Success": self.preprocess_count['success'],
                        "Fail": self.preprocess_count['fail'],
                        "Missing": self.preprocess_count['missing'],
                        "Skip": self.preprocess_count['skip'],
                        "Converted": self.preprocess_count['converted'],
                        "Error": self.preprocess_count['error'],
                        "Warning": self.preprocess_count['warning']
                    })
                    progress.update(1)
                except Exception as e:
                    tqdm.write(f"Error processing file: {e}")
        tqdm.write(f"Processed {self.preprocess_count['success']} rows | Failed {self.preprocess_count['fail']} rows")

    def process_row(self, idx, rows):
        file_name = rows['path'].replace('.mp3', '')
        audio_file = config.COMMON_VOICE_PATH / 'clips' / f"{file_name}.mp3"
        wav_file = config.WAVS_PATH / f"{file_name}.wav"
        transcription = rows['sentence']

        local_results = {'skip': 0, 'fail': 0, 'success': 0, 'converted': 0, 'error': 0, 'missing': 0, 'warning': 0}
        local_results['fail'] += 1
        try:
            if not isinstance(transcription, str):
                if os.path.exists(audio_file):
                    local_results['warning'] += 1
                return local_results

            if len(transcription) == 0:
                if os.path.exists(audio_file):
                    local_results['warning'] += 1
                return local_results

            transcription = normalize_text(transcription)

            if os.path.exists(wav_file):
                if os.path.exists(audio_file):
                    # os.remove(audio_file)
                    pass
                local_results['skip'] += 1

            elif os.path.exists(audio_file):
                output = audio.to_wav(audio_file, wav_file)
                if output is None or not os.path.exists(wav_file):
                    local_results['error'] += 1
                    return local_results
                local_results['converted'] += 1
            else:
                local_results['missing'] += 1
                return local_results
            
            with self.file_lock:
                self.data.append((file_name, transcription))

            local_results['fail'] -= 1
            local_results['success'] += 1

        except Exception as e:
            self.preprocess_count['error'] += 1
            tqdm.write(f"Error processing file {file_name}: {e}")
        return local_results
           
    def generate_transcription_list(self):
        print(f"Generating transcription list")

        # Read train file
        train_file = config.OUTPUT_DIR / f'train.tsv'
        dev_file = config.OUTPUT_DIR / f'dev.tsv'
        
        transcription_list = []
        
        # Read train file if it exists
        if os.path.exists(train_file):
            df_train = pd.read_csv(train_file, sep='\t')
            train_transcriptions = df_train['transcription'].tolist()
            transcription_list.extend(train_transcriptions)
            print(f"Added {len(train_transcriptions)} transcriptions from train.tsv")
        
        # Read dev file if it exists
        if os.path.exists(dev_file):
            df_dev = pd.read_csv(dev_file, sep='\t')
            dev_transcriptions = df_dev['transcription'].tolist()
            transcription_list.extend(dev_transcriptions)
            print(f"Added {len(dev_transcriptions)} transcriptions from dev.tsv")
        
        # Fallback to original save_file if train/dev files don't exist
        if not transcription_list:
            print(f"Train/dev files not found, using {self.save_file}")
            df = pd.read_csv(self.save_file, sep='\t')
            transcription_list = df['transcription'].tolist()

        transcription_filename = config.OUTPUT_DIR / f'{config.LANGUAGE}_sentences.txt'
        os.makedirs(transcription_filename.parent, exist_ok=True)
        with open(transcription_filename,'w', encoding="utf-8") as f:
            for transcription in transcription_list:
                f.write(transcription + '\n')

        print(f"Generated transcription list with {len(transcription_list)} total transcriptions - {transcription_filename}")

class BucketAudio():
    def __init__(self):
        self.output_dir = config.OUTPUT_DIR / 'buckets'
        self.train_output_dir = self.output_dir / 'train'
        self.dev_output_dir = self.output_dir / 'dev'
        self.train_tsv_file = config.OUTPUT_DIR / 'train.tsv'
        self.dev_tsv_file = config.OUTPUT_DIR / 'dev.tsv'
        self.train_buckets = {}
        self.dev_buckets = {}

        # Create directories for train and dev buckets
        if not os.path.exists(self.train_output_dir):
            os.makedirs(self.train_output_dir)
        if not os.path.exists(self.dev_output_dir):
            os.makedirs(self.dev_output_dir)
        
    def group_duration_for_file(self, tsv_file, buckets_dict, data_type):
        """Group duration for a specific TSV file (train or dev)"""
        if not os.path.exists(tsv_file):
            print(f"File {tsv_file} not found, skipping {data_type} bucketing")
            return
            
        df = pd.read_csv(tsv_file, sep='\t')
        print(f"Processing {len(df)} samples for {data_type} bucketing")
        progress = tqdm(total=len(df), desc=f"Processing {data_type} Buckets..")
        
        for idx, rows in df.iterrows():
            bucket_duration = rows['bucket_duration']
            if bucket_duration >= 30.0:
                bucket_duration = 30.0
            elif bucket_duration <= 0.0:
                bucket_duration = 0.0
            elif bucket_duration <= 5.0:
                bucket_duration = 5.0
            elif bucket_duration <= 10.0:
                bucket_duration = 10.0
            elif bucket_duration <= 15.0:
                bucket_duration = 15.0
            elif bucket_duration <= 20.0:
                bucket_duration = 20.0
            elif bucket_duration <= 25.0:
                bucket_duration = 25.0
            else:
                bucket_duration = 0.0
            
            if bucket_duration not in buckets_dict:
                buckets_dict[bucket_duration] = []

            buckets_dict[bucket_duration].append({
                "file_name": rows["file_name"],
                "transcription": rows['transcription'],
                "bucket_duration": rows['bucket_duration'],
                "orig_duration": rows['orig_duration'],
                "duration": rows['duration']
            })
            progress.update(1)
    
    def group_duration(self):
        # Process train data
        self.group_duration_for_file(self.train_tsv_file, self.train_buckets, "train")
        
        # Process dev data  
        self.group_duration_for_file(self.dev_tsv_file, self.dev_buckets, "dev")

    def save_buckets(self):
        # Save train buckets
        for keys, item in self.train_buckets.items():
            bucket_file = self.train_output_dir / f"bucket_{keys}.json"
            with open(bucket_file, 'w') as f:
                json.dump(item, f, indent=2)
        
        # Save dev buckets
        for keys, item in self.dev_buckets.items():
            bucket_file = self.dev_output_dir / f"bucket_{keys}.json"
            with open(bucket_file, 'w') as f:
                json.dump(item, f, indent=2)

        print(f"[✓] Saved {len(self.train_buckets.keys())} train buckets to: {self.train_output_dir}")
        print(f"[✓] Saved {len(self.dev_buckets.keys())} dev buckets to: {self.dev_output_dir}")
    
    def load_buckets(self, data_type='train'):
        """Load buckets for specific data type (train or dev)"""
        if data_type == 'train':
            bucket_dir = self.train_output_dir
        elif data_type == 'dev':
            bucket_dir = self.dev_output_dir
        else:
            # Fallback to original behavior for backward compatibility
            bucket_dir = self.output_dir
        
        data = {}
        if not os.path.exists(bucket_dir):
            print(f"Bucket directory {bucket_dir} does not exist")
            return data
            
        for bucket_file in os.listdir(bucket_dir):
            if bucket_file.endswith('.json'):
                with open(os.path.join(bucket_dir, bucket_file), 'r') as f:
                    data[bucket_file.replace('.json', '').replace('bucket_', '')] = json.load(f)

        print(f"[✓] Loaded {len(data.keys())} {data_type} buckets from: {bucket_dir}")
        return data

    def init(self):
        self.group_duration()
        self.save_buckets()


def preprocess():
    print(f"Preprocessing transcriptions")
    audio_transcription = AudioTranscriptionTSV()
    audio_transcription.preprocess_tsv('train')
    audio_transcription.preprocess_tsv('dev')
    audio_transcription.generate_transcription_list()
    lc.train()
    print(f"Preprocessing audio")
    audio_info = AudioInfo()
    audio_info.preprocess('train')
    audio_info.preprocess('dev')
    print(f"Preprocessing buckets")
    BucketAudio().init()

if __name__ == '__main__':
    preprocess()

