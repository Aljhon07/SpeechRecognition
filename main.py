from src import train
from src.preprocess import Preprocessor
import pandas as pd
from  src.neural_net.LightWeightModel import LightWeightModel
from tqdm import tqdm
from tools.dataset_cleaner import remove_audio_files


if __name__ == '__main__':
    # print(os.cpu_count())
    # file = config.COMMON_VOICE_PATH = config.COMMON_VOICE_PATH / 'clips' / 'common_voice_en_20273690.mp3'
    # print(file.exists())
    # remove_audio_files(config.COMMON_VOICE_PATH / 'invalidated.tsv')
    # preprocess()
    train.main()
    # print(os.path.exists(config.COMMON_VOICE_PATH / 'clips' / 'common_voice_en_16759015.mp3'))
    # sanity_check()
    pass


