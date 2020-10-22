from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM

# Load the tokenizer
#tokenizer = DistilBertTokenizerFast.from_pretrained("distilabena-base-v2-akuapem-twi-cased", max_len=512) # the one we trained ourselves (akuapem)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilabena-base-v2-akuapem-twi-cased", max_len=512, do_lower_case=True) # the one we trained ourselves (asante, lowercase everything)
#tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-multilingual-cased") # you could also use pre-trained DistilmBERT tokenizer
#tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-multilingual-cased", do_lower_case=True) # for asante, lowercase pretrained tokenizer
#tokenizer.save_vocabulary("distilabena-base-akuapem-twi-cased") # when using pretrained tokenizer, be sure to save it locally
tokenizer.save_vocabulary("distilabena-base-v2-asante-twi-uncased") # saving pretrained tokenizer locally in case of asante 

# Load DistilBERT multilingual base checkpoint
#model = DistilBertForMaskedLM.from_pretrained("distilbert-base-multilingual-cased") # pretrained DistilmBERT weights
model = DistilBertForMaskedLM.from_pretrained("distilabena-base-v2-akuapem-twi-cased") # in the case of Asante Twi, start with Akuapem model weights
print("Number of parameters in the model:")
print(model.num_parameters())

# Create dataset object for JW300 dataset (Akuapem) or Asante Twi Bible 
from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
#    file_path="../../data/jw300.en-tw.tw", # stage 1 - akuapem
    file_path="../../data/asante_twi_bible.txt", # stage 2 - asante
    block_size=128,
)

# Create "data collator" from dataset and tokenizer - with 15% chance of masking
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Define training arguments
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
#    output_dir="distilabena-base-v2-akuapem-twi-cased", # tokenizer from scratch (akuapem)
    output_dir="distilabena-base-v2-asante-twi-uncased", # tokenizer from scratch (asante)
#    output_dir="distilabena-base-akuapem-twi-cased", # pretrained DistilmBERT tokenizer (akuapem)
#    output_dir="distilabena-base-asante-twi-uncased", # pretrained DistilmBERT tokenizer (asante)
    overwrite_output_dir=True,
    num_train_epochs=5, # more epochs needed with tokenizer we trained from scratch
#    num_train_epochs=3, # with pretrained tokenizer, not as many epochs needed
    per_gpu_train_batch_size=16,
    save_steps=10000,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

# Train
trainer.train()

# Save
#trainer.save_model("distilabena-base-v2-akuapem-twi-cased")
#trainer.save_model("distilabena-base-akuapem-twi-cased")
#trainer.save_model("distilabena-base-asante-twi-uncased")
trainer.save_model("distilabena-base-v2-asante-twi-uncased")

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="distilabena-base-v2-asante-twi-uncased",
    tokenizer="distilabena-base-v2-asante-twi-uncased"
)

# We modified a sentences as "Eyi de ɔhaw kɛse baa sukuu hɔ." => "Eyi de ɔhaw kɛse baa [MASK] hɔ."
print(fill_mask("Eyi de ɔhaw kɛse baa [MASK] hɔ."))
