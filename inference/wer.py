import os
import jiwer
from inference.inference import inference  # If inference.py is in the inference folder
import pandas as pd
import config
from tools.utils import normalize_text, rms_normalize
import torchaudio
import config
from tqdm import tqdm

def evaluate_model():
    error_rate = 0.0
    step_count = 0
    test_tsv =  config.OUTPUT_DIR / "test.tsv"
    precomputed_dir = config.PRECOMPUTED_DIR
    df = pd.read_csv(test_tsv, sep='\t').head(150)
    progress_bar = tqdm(df.iterrows(), total=len(df), desc="Evaluating WER")
    for idx, row in progress_bar:
        audio_file = row['audio_path']
        reference = row['transcript']
        reference = normalize_text(reference)

        if not os.path.exists(audio_file):
            tqdm.write(f"Audio file {audio_file} does not exist. Skipping.")
            continue

        hypothesis = inference(audio_file, use_post_processing=False)
        print(hypothesis)
        hypothesis = normalize_text(hypothesis[0])

        # tqdm.write(f"Reference: {reference}\nHypothesis: {hypothesis}")
        if hypothesis == "" or hypothesis == None or reference == "" or reference == None:
            tqdm.write(f"Reference: {reference} | Hypothesis: {hypothesis} | WER: 1.0")
            continue
        score = jiwer.cer(reference, hypothesis)
        log_file = config.MODEL_DIR / "wer_log.txt"

        with open(log_file, 'a') as f:
            f.write(f"Reference: {reference}\nPrediction: {hypothesis}\nScore: {100 - score * 100:.2f}\n\n")

        progress_bar.set_postfix({
            "WER": f"{score:.3f}",
            "Avg WER": f"{(error_rate + score) / (step_count + 1):.3f}" if step_count > 0 else f"{score:.3f}",
            "Accuracy": f"{(1 - score) * 100:.2f}%",
            "Avg Accuracy": f"{(1 - (error_rate + score) / (step_count + 1)) * 100:.2f}%" if step_count > 0 else f"{(1 - score) * 100:.2f}%"
        })
        error_rate += score
        step_count += 1
        
    if step_count == 0:
        print("No valid samples were evaluated. Please check your data and inference outputs.")
        return

    error_rate /= step_count

    print(f"Average WER ({step_count} samples): {(1 - error_rate) * 100:.2f}%")
    

if __name__ == "__main__":
    evaluate_model()