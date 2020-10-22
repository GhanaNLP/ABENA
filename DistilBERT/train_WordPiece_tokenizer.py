# use this when training BPE tokenizer from scratch
from pathlib import Path

from tokenizers import BertWordPieceTokenizer

paths = ['../../data/jw300.en-tw.tw','../../data/asante_twi_bible.txt'] # dataset location

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer()

# Customize training
tokenizer.train(
    paths,
    vocab_size=30000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    wordpieces_prefix="##",
)

# Save tokenizer to disk - make sure these directories exist
tokenizer.save_model("distilabena-base-v2-akuapem-twi-cased") # akuapem
