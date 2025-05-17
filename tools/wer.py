import os
import jiwer
import pandas as pd
import config
from inference.inference import inference
from tools.utils import normalize_text

df = pd.read_csv(config.BASE_DIR / 'diff.tsv', sep='\t')

def wer(ref, hyp):
    wer = jiwer.wer(ref, hyp)
    return wer

total_sample = 0
total_wer = 0

for idx, row in df.iterrows():
    if total_sample > 1000:

        
        break
    ref = row['sentence']
    ref = normalize_text(ref)
    path = config.COMMON_VOICE_PATH / 'clips' / f"{row['path']}"

    if os.path.exists(path):
        result = inference(path)
        wer_result = wer(ref, result)
        print(f"Ref: {ref} | Hyp: {result} | WER: {wer_result}")
        total_wer += wer_result
        total_sample += 1

print("Total samples: ", total_sample)
print("Total WER: ", total_wer)
print(f"Mean WER: {total_wer / total_sample}")
