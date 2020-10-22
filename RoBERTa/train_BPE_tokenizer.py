# use this when training BPE tokenizer from scratch
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = ['../../data/jw300.en-tw.tw','../../data/asante_twi_bible.txt'] # dataset location

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "[CLS]",
    "[PAD]",
    "[SEP]",
    "[UNK]",
    "[MASK]",
]) # which special tokens to use for start, padding, end, unknown and mask respectively

# Save files to disk - make sure these directories exist
tokenizer.save_model("distilbako-base-akuapem-twi-cased") # akuapem
