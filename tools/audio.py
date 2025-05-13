import os
import subprocess
import config
from pathlib import Path
import pandas as pd
import tqdm
def to_wav(input_file, output_file):
    try:
        if os.path.exists(output_file):
            # print(f"File {output_file.relative_to(Path.cwd())} already exists. Skipping conversion.", end="\r")
            print(f"File {output_file} already exists. Skipping conversion.")
            return

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        command = [
            'ffmpeg',
            '-i', input_file,
            '-acodec', 'pcm_s16le',  # WAV format, 16-bit PCM
            '-ac', '1',              # Mono audio
            '-ar', '16000',           # 16kHz sample rate
            output_file
        ]
        subprocess.run(command, check=True, stderr=subprocess.PIPE) #added stderr to help with debugging.
        return output_file
    except FileNotFoundError:
        print("Error: FFmpeg not found. Make sure it's installed and in your PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def to_wav_batch(file_names, input_dir, output_dir):
    for file_name in file_names:
        input_file = os.path.join(input_dir, f"{file_name}.mp3")
        output_file = os.path.join(output_dir, f"{file_name}.wav")
        
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping conversion.")
            continue
        
        print(f"Processing {input_file} to {output_file}")
        if os.path.exists(input_file): 
            to_wav(input_file, output_file)
        else:
            print(f"File {input_file} does not exist. Skipping.")



if __name__ == "__main__":
    tsv_file = config.COMMON_VOICE_PATH / 'combined.tsv'
    df = pd.read_csv(tsv_file, sep='\t')
    to_drop = []

    wav_exists = 0
    converted = 0
    not_exist = 0
    removed = 0
    mp3_count = 0
    progress_bar = tqdm.tqdm(total=len(df), desc="Processing files")

    for idx, row in df.iterrows():
        audio_file = row['path'].replace('.mp3', '')
        wav_file = config.COMMON_VOICE_PATH / 'wavs' / f"{audio_file}.wav"
        mp3_file = config.COMMON_VOICE_PATH / 'clips' / f"{audio_file}.mp3"

        progress_bar.set_postfix({
            'WAV Exists': wav_exists,
            'Converted': converted,
            'Not Exist': not_exist,
            'Removed': removed,
            'MP3': mp3_count
        })
        if os.path.exists(wav_file):
            wav_exists += 1
            if os.path.exists(mp3_file):
                os.remove(mp3_file)
                removed += 1
        elif os.path.exists(mp3_file):
            mp3_count += 1
            to_wav(mp3_file, wav_file)
            if os.path.exists(wav_file):
                os.remove(mp3_file)
                converted += 1
        else:
            not_exist += 1
            to_drop.append(idx)

        progress_bar.update(1)
    progress_bar.close()
    # df.drop(index=to_drop, inplace=True)
    # df.to_csv(tsv_file, sep='\t', index=False)