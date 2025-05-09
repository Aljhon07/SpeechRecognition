import os
import subprocess
import config
from pathlib import Path

def to_wav(input_file, output_file):
    try:
        if os.path.exists(output_file):
            print(f"File {output_file.relative_to(Path.cwd())} already exists. Skipping conversion.", end="\r")
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
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr.decode()}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

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


