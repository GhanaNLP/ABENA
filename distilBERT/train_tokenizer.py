from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

#paths = [str(x) for x in Path("./eo_data/").glob("**/*.txt")]
paths = ['jw300.en-tw.tw']

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save("twibert")
