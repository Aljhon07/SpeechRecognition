import os
import sentencepiece as spm
import config



def train(model_type='char', vocab_size=config.H_PARAMS["VOCAB_SIZE"],model_prefix = config.LANGUAGE):
    if vocab_size > 100:
        model_type = 'bpe'
        
    input_file = config.OUTPUT_DIR / (f"{config.LANGUAGE}_sentences.txt"
                                       f"")
    print(f"Training SentencePiece model with input file: {input_file}")
    model_path = config.OUTPUT_DIR / model_prefix

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_path,
        model_type=model_type,
        character_coverage=1.0,
        vocab_size=vocab_size,
        pad_id=0,        # <pad> = 0
        unk_id=1,        # <unk> = 1
        bos_id=2,        # <s> = 2
        eos_id=3,        # </s> = 3
        # Special token strings (must match the IDs above)
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>"
    )
    print(f"SentencePiece model trained successfully: {model_prefix}.model")


model_file = str(config.OUTPUT_DIR / f"{config.LANGUAGE}.model")

def encode(input_text, vocab_model=model_file):
    try:
        sp = spm.SentencePieceProcessor(vocab_model)
        encoded = sp.encode_as_ids(input_text) 
        return encoded
    
    except Exception as e:
        print(f"Error encoding with SentencePiece: {e}")
        return None

def decode(encoded_tokens, vocab_model=model_file):
    """Decodes tokens using a SentencePiece model."""
    try:
        sp = spm.SentencePieceProcessor(vocab_model)
        decoded = sp.decode(encoded_tokens)
        return decoded
    except Exception as e:
        print(f"Error decoding with SentencePiece: {e}")
        return None
    

if __name__ == "__main__":
    train()
    test_str = "Hello world"
    
    encoded = encode(test_str)
    print(f"Encoded: {encoded}")