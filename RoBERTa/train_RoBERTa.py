from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("trained_models/distilbako-base-akuapem-twi-cased", max_len=512, do_lower_case=True)
tokenizer.save_vocabulary("trained_models/distilbako-base-asante-twi-uncased") # saving pretrained tokenizer locally in case of asante

from transformers import RobertaForMaskedLM

#model = RobertaForMaskedLM(config=config) # from scratch, for Akuapem
model = RobertaForMaskedLM.from_pretrained("trained_models/distilbako-base-akuapem-twi-cased") # fine-tune from Akuapem weights, for Asante
print(model.num_parameters())

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="asante_twi_bible.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="trained_models/distilbako-base-asante-twi-uncased",
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

trainer.save_model("trained_models/distilbako-base-asante-twi-uncased")


from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="trained_models/distilbako-base-asante-twi-uncased",
    tokenizer="trained_models/distilbako-base-asante-twi-uncased"
)

# Saa tebea yi maa me papa <mask>.
# =>

print(fill_mask("Saa tebea yi maa me papa <mask>.")) 
