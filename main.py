import os
import config
from src.preprocess import preprocess
import pandas as pd
import winsound

if __name__ == '__main__':

    preprocess()

    # print(os.path.exists(config.COMMON_VOICE_PATH / 'clips' / 'common_voice_en_16759015.mp3'))