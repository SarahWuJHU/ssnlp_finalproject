#%%
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from datetime import datetime
import os
from scipy.io import savemat
from torch.utils.data import DataLoader
from transformers import EncoderDecoderModel
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import get_scheduler
import gensim.downloader
import numpy as np
from transformers import BertGenerationEncoder, BertGenerationDecoder, BertGenerationConfig
import torch.optim as optim
from transformers import PreTrainedModel, PretrainedConfig, BertModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import pandas as pd
import requests
import wikipedia
import re
import json
import nltk
nltk.download('punkt')

API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
#%%
# Loading the Wikipedia data
DATA_SET = ""
EXCLUDED_SECTIONS = ['See also', 'References',
                     'Further reading', 'External links']
ARTICLE_TITLES = wikipedia.search('story telling', results=20)
MARKDOWN_PATTERN = re.compile(r'(\'{2,5})(.*?)\1')

cleaned_articles = []
wikipedia.set_lang("en")
testing_plain_sentence = []
for article_title in ARTICLE_TITLES:
    try:
      # Get the raw text of the Wikipedia article
      raw_article = wikipedia.page(article_title).content
      # Remove any markdown from the article text
      cleaned_article = re.sub(MARKDOWN_PATTERN, r'\2', raw_article)
      # Split the article text into sections
      sections = cleaned_article.split('\n==')

      # Loop over each section and exclude any sections in the excluded_sections list
      cleaned_sections = []
      for section in sections:
          if not any(section.startswith(f"\n{es}\n") for es in EXCLUDED_SECTIONS):
              # Tokenize the section into sentences using nltk
              sentences = nltk.sent_tokenize(section)
              # Append the cleaned sentences to the cleaned_sections list
              testing_plain_sentence.extend(sentences)
    except:
      print(article_title, 'invalid')

emo_data = load_dataset('empathetic_dialogues')
dataset_test_subset_label = emo_data['test']['context']
#%%
dataset_test_subset_label = dataset_test_subset_label[:len(testing_plain_sentence)]
print(dataset_test_subset_label)
print(f" - test: {len(testing_plain_sentence)}")
print(f" - test: {len(dataset_test_subset_label)}")
# %%
prediction = []
#%%
for sentence, emotion in zip(testing_plain_sentence[301:], dataset_test_subset_label[301:]):
    prompt = "Emotionally mutual text: " + sentence + "\n"
    prompt += "Expressing the sentence with a sentiment of "+ emotion +":"
    output = query({"inputs": prompt})
    print(output)
    pred = output[0]['generated_text'].split('\n')[-1].split(':')[-1][1:]
    prediction.append(pred)

print(prediction)
# %%
data = {'prediction': prediction, 'label':dataset_test_subset_label}
with open("test", "w") as fp:
    json.dump(data, fp)
# %%
with open("test_incontext", "r") as fp:
    data = json.load(fp)
print(len(data['prediction']))
# %%
