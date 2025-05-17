import pandas as pd
import os
import config
import tqdm

def unique_col_values(tsv_file, column_name='path'):
    df = pd.read_csv(tsv_file, sep='\t')
    duplicate_count = df[column_name].duplicated().sum()
    
    if duplicate_count == 0:
        print("✅ All paths are unique!")
    else:
        print(f"❌ Found {duplicate_count} duplicate paths")
        duplicates = df[column_name].value_counts()
        duplicates = duplicates[duplicates > 1]
    
    return duplicate_count
    
def combined_tsv(tsv_files):
    combined_df = pd.DataFrame()

    for tsv_file in tsv_files:
        tsv_file = config.COMMON_VOICE_PATH / f"{tsv_file}.tsv"
        if not os.path.exists(tsv_file):
            print(f"File {tsv_file} does not exist.")
            continue
        print(f"Loading {tsv_file}")
        df = pd.read_csv(tsv_file, sep='\t', low_memory=False)  
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(config.COMMON_VOICE_PATH / 'combined.tsv', sep='\t', index=False)

def keep_max_n_duplicates(input_tsv, output_tsv, column_name='sentence', max_count=2):
    df = pd.read_csv(input_tsv, sep='\t')

    df_limited = df.groupby(column_name).head(max_count).reset_index(drop=True)

    df_limited.to_csv(output_tsv, sep='\t', index=False)

    print(f"Saved filtered TSV with up to {max_count} duplicates per transcription to {output_tsv}")

def remove_audio_files(input_tsv):
    df = pd.read_csv(input_tsv, sep='\t')
    audio_files = df['path'].tolist()

    progress = tqdm.tqdm(audio_files, desc=f"Removing audio files from {input_tsv}", unit="file")
    removed = 0
    for audio_file in audio_files:
        mp3_file = config.COMMON_VOICE_PATH / 'clips' / f"{audio_file}"
        if os.path.exists(mp3_file):
            os.remove(mp3_file)
            removed += 1
        progress.set_postfix({'Removed': removed})
        progress.update(1)

def keep_unique_paths(input_tsv, output_tsv, column_name='path'):
    df = pd.read_csv(input_tsv, sep='\t')
    
    total_rows = len(df)
    duplicate_mask = df.duplicated(subset=[column_name], keep=False)
    duplicate_count = duplicate_mask.sum()
    
    if duplicate_count == 0:
        print("✅ No duplicates found - file copied unchanged")
        df.to_csv(output_tsv, sep='\t', index=False)
        return df
    
    print(f"🔍 Found {duplicate_count} duplicate rows (out of {total_rows} total)")
    
    cleaned_df = df.drop_duplicates(subset=[column_name], keep='first')
    
    remaining_duplicates = cleaned_df.duplicated(subset=[column_name]).sum()
    print(f"✅ Removed {duplicate_count - remaining_duplicates} duplicates")
    print(f"✅ Final count: {len(cleaned_df)} unique rows")
    
    cleaned_df.to_csv(output_tsv, sep='\t', index=False)
    print(f"📁 Cleaned data saved to: {output_tsv}")
    
    return cleaned_df

def tsv_diff(tsv1, tsv2):
    main_df = pd.read_csv(tsv1, sep='\t')
    subset_df = pd.read_csv(tsv2, sep='\t')

    filtered_df = main_df[~main_df['sentence'].isin(subset_df['sentence'])]

    filtered_df.to_csv(config.COMMON_VOICE_PATH / 'diff.tsv', sep='\t', index=False)

if __name__ == '__main__':
    # output_tsv = config.COMMON_VOICE_PATH / 'filtered.tsv'
    # combined_tsv( ['dev', 'test', 'train', 'validated'])
    # keep_unique_paths(config.COMMON_VOICE_PATH / 'combined.tsv', config.COMMON_VOICE_PATH / 'unique_audio.tsv')
    # keep_max_n_duplicates(config.COMMON_VOICE_PATH / 'unique_audio.tsv', config.COMMON_VOICE_PATH / 'clean_2.tsv', max_count=2)
    # remove_audio_files(config.COMMON_VOICE_PATH / 'diff.tsv')
    # combined_tsv(['clean_1', 'clean_2'])
    # tsv_diff(config.COMMON_VOICE_PATH / 'other.tsv', config.COMMON_VOICE_PATH / 'clean.tsv')
    pass
