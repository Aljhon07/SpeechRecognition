import os
import config
from src.preprocess import preprocess
import pandas as pd
from  src.neural_net.LightWeightModel import LightWeightModel
import torchaudio
import random
from tqdm import tqdm

# def sanity_check():
#     output_tsv = config.OUTPUT_DIR / f'{config.LANGUAGE}.tsv'
#     if not os.path.exists(output_tsv):
#         raise FileNotFoundError(f"{output_tsv} does not exist")
    
#     df = pd.read_csv(output_tsv, sep='\t')
#     total_rows = len(df)
#     progress_bar = tqdm(total=total_rows, desc="Sanity Check", unit="row")
#     aligned_rows = 0
#     misaligned_rows = 0
#     for idx, rows in df.iterrows():
#         file_name = rows['file_name']
#         num_frames = rows['num_frames']
        
#         audio_file = config.WAVS_PATH / f"{file_name}.wav"

#         info = torchaudio.info(audio_file)
#         if (info.num_frames != num_frames):
#             aligned_rows += 1
#         else:
#             misaligned_rows += 1

#         progress_bar.set_postfix({
#             'aligned_rows': aligned_rows,
#             'misaligned_rows': misaligned_rows,
#         })
#         progress_bar.update(1)

if __name__ == '__main__':
    # print(os.cpu_count())
    # file = config.COMMON_VOICE_PATH = config.COMMON_VOICE_PATH / 'clips' / 'common_voice_en_20273690.mp3'
    # print(file.exists())
    preprocess()

    # print(os.path.exists(config.COMMON_VOICE_PATH / 'clips' / 'common_voice_en_16759015.mp3'))
    # sanity_check()
    pass


