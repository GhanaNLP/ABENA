# ABENA: BERT Natural Language Processing for Twi
This repo contains the training scripts for training our ABENA family of models for BERT-type NLP on the Ghanaian language Twi. ABENA stands for "A BERT Now in Akan".

**PLEASE NOTE THAT THIS IS A WORK IN PROGRESS. THE DATA HAS A STRONG RELIGIOUS BIAS, EVEN THOUGH THE MODEL IS ALREADY A USEFUL TOOL FOR MANY APPLICATIONS, IT IS CRITICAL TO KEEP THE BIAS IN MIND. WE MAKE IT FREELY AVAILABLE IN THE HOPE THAT IT WILL HELP A LOT OF PEOPLE, BUT YOU USE IT AT YOUR OWN RISK.**

![ABENA - A BERT Now in Akan](https://github.com/GhanaNLP/ABENA/blob/master/Abena.png?raw=true)

Both the Asante and Akuapem dialects are addressed. Akuapem is addressed via the [JW300 twi subset](https://www.kaggle.com/azunre/jw300entw) and Twi via the [Asante Twi Bible](www.bible.org) (get a clean copy of the Bible dataset from the links in [this paper](https://www.aclweb.org/anthology/2020.lrec-1.335.pdf)).

We perform a variety of experiments with BERT, DistilBERT and RoBERTa modeling architectures. You can find more details in our associated (blog post). You can find the 8 models this effort yielded listed in the [Hugging Face Model Hub](https://huggingface.co/Ghana-NLP) and see instructions on how to use them in this [Kaggle Notebook](https://www.kaggle.com/azunre/ghananlp-abena-usage-demo)

All models presented were trained with a single Tesla K80 GPU on an NC6 Azure VM instance.

Unless you are on Kaggle where the dependencies are already installed, be sure to first install dependencies using our `requirements.txt` as

```pip install -r requirements.txt```

# ABENA -- "A BERT Now in Akan" -- BERT for Twi
We introduce at least four different flavors of ABENA:
* We first employ transfer learning to fine-tune a multilingual BERT (mBERT) model on  the Twi subset of the JW300 dataset. This data is largely composed of the Akuapem dialect of Twi. This is a cased model.
* Subsequently, we fine-tune this model further on the Asante Twi Bible data to obtain an Asante Twi version of the model. This model is uncased due to the significantly smaller size of the Bible compared to JW300.
* Additionally, we perform both experiments using the DistilBERT architecture instead of BERT -  this yields smaller and more lightweight versions of the (i) Akuapem and (ii) Asante ABENA models.

We also experiment with training our own tokenizer even when fine-tuning from pretrained mBERT weights, which yields some additional model flavors. 

To train your own flavors of the models, it suffices to train tokenizers with `BERT\train_WordPiece_tokenizer.py` / `DistilBERT\train_WordPiece_tokenizer.py`, followed by training the models with `BERT\train_BERT.py` / `DistilBERT\train_DistilBERT.py`. 

More details on convergence times, numbers of parameters, etc., can be found in our associated (blog post). The scripts have been heavily documented to help you out. We anticipate making these even more user-friendly shortly by wrapping them into our [Kasa Library](https://github.com/GhanaNLP/kasa).

# RoBAKO -- "Robust BERT with Akan Knowledge Only" -- RoBERTA for Twi
This implementation largely follows this [tutorial from Hugging Face](https://huggingface.co/blog/how-to-train). 

BAKO stands for "BERT with Akan Knowledge Only", i.e., trained from scratch on monolingual Twi data. 

We found BERT and DistilBERT not suitable for this right now, given the relatively small size of the dataset (JW300). For this reason we only presented RoBERTa versions of BAKO: RoBAKO - Robustly Optimized BAKO. 

[RoBERTa](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/) is an improvement on BERT that employs clever optimization tricks for better efficiency. In particular, among other improvements, we use byte-pair encoding (BPE) for tokenization, which is arguably more effective than the WordPiece approach used by BERT and DistilBERT.

In short, first train tokenizer (`RoBERTa/train_BPE_tokenizer.py`), and then train model (`RoBERTa/train_RoBERTa.py`).
