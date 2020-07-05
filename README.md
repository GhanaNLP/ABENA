# train-language-models
Collection of Training Scripts for Various Language Models - BERT/mBERT, distilBERT, etc

# distilBERT
the implementation here, located in the distilBERT folder, is based on https://huggingface.co/blog/how-to-train

In short, first train tokenizer (`train_tokenizer.py`), and then train model (`train_distilBERT.py`).

Twi data we used for this can be found at https://www.kaggle.com/azunre/jw300entw. 

Results pretty good after about 24 hours on a GPU, at the hyper-parameters in the script.
