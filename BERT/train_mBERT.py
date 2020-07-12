from transformers import BertConfig
from transformers import BertTokenizerFast

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased") # use pretrained

tokenizer = BertTokenizerFast.from_pretrained("twibert", max_len=512) #  use the language-specific we trained with train_tokenizer.py

from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")

print("Number of parameters in mBERT model:")
print(model.num_parameters())

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="jw300.en-tw.tw",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="twimbert",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()

trainer.save_model("twimbert")


from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="twimbert",
    tokenizer=tokenizer
)

# Saa tebea yi maa me papa <mask>.
# =>

print(fill_mask("Eyi de ɔhaw kɛse baa [MASK] hɔ."))
