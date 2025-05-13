import config
import pandas
import os
import tqdm
mp3_dir = config.COMMON_VOICE_PATH / 'clips'
wav_dir = config.COMMON_VOICE_PATH / 'wavs'

def remove_audio_files(input_tsv):
    df = pandas.read_csv(input_tsv, sep='\t')

    loader = tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing files")
    mp3_count = 0
    wav_count = 0
    no_match = 0
    both_match = 0
    for idx, row in df.iterrows():
        audio_file = row['path'].replace('.mp3', '')
        wav_file = wav_dir / f"{audio_file}.wav"
        mp3_file = mp3_dir / f"{audio_file}.mp3"
        if os.path.exists(wav_file):
            wav_count += 1
            if os.path.exists(mp3_file):
                both_match += 1
                os.remove(mp3_file)
                print(f"Removed {mp3_file}")
        else:
            if os.path.exists(mp3_file):
                mp3_count += mp3_count
                print(f"File {mp3_file} exists.")
            else:
                no_match += 1
                print(f"File {wav_file} does not exist.")
    
    print(f"WAV: {wav_count}, MP3: {mp3_count}, No match: {no_match}, Both match: {both_match}")


if __name__ == '__main__':
    input_tsv = config.COMMON_VOICE_PATH / 'combined.tsv'
    remove_audio_files(input_tsv)