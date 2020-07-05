from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("twibert", max_len=512)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
print(model.num_parameters())

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="jw300.en-tw.tw",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="twibert",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_gpu_train_batch_size=32,
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

trainer.save_model("twibert")


from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="twibert",
    tokenizer="twibert"
)

# Saa tebea yi maa me papa <mask>.
# =>

print(fill_mask("Saa tebea yi maa me papa <mask>."))
