import os
import jiwer
from inference.inference import inference  # If inference.py is in the inference folder
import pandas as pd
import config
from tools.utils import normalize_text, rms_normalize
import torchaudio


test_tsv = config.BASE_DIR / 'inference' / 'test' / 'en' / 'validated.tsv'
test_clips = config.BASE_DIR / 'inference' / 'test' / 'en' / 'clips' 
log_file = config.BASE_DIR / 'inference' / 'test' / 'en' / 'wer.log'
if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        f.write("")
    
def evaluate_model(tsv_file):
    error_rate = 0.0
    step_count = 0

    for idx, row in pd.read_csv(test_tsv, sep='\t').iterrows():
        audio_file = test_clips / row['path']
        reference = row['sentence']
        reference = normalize_text(reference)


        if not os.path.exists(audio_file):
            continue

        wav, sr = torchaudio.load(audio_file)
        rms = rms_normalize(wav)
        if rms is None:
            continue

        hypothesis = inference(audio_file)

        if hypothesis == "" or hypothesis == None or reference == "" or reference == None:
            print(f"Reference: {reference} | Hypothesis: {hypothesis} | WER: 1.0")
            continue
        score = jiwer.wer(reference, hypothesis)
        if score > 1:
            continue

        with open(log_file, 'a') as f:
            f.write(f"Reference: {reference}\nPrediction: {hypothesis}\nScore: {100 - score * 100:.2f}\n\n")
        print(f"File Name: {audio_file} Reference: {reference} | Hypothesis: {hypothesis} | WER: {score}")
        error_rate += score
        step_count += 1
        
        if step_count >= 1000:
            print(f"Processed {step_count} samples, current WER: {error_rate / step_count}")
            break

    error_rate /= step_count

    print(f"Average WER ({step_count} samples): {error_rate}")

if __name__ == "__main__":
    evaluate_model(test_tsv)