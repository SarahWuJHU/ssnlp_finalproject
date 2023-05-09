# %%
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelBinarizer
import os
from scipy.io import savemat
from torch.utils.data import DataLoader
from transformers import EncoderDecoderModel
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import get_scheduler
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import pandas as pd
import nltk
nltk.download('punkt')
# tested with this, might need a custom tokenizer for the actual inputs

warnings.filterwarnings("ignore")
# %%
##################################
######## Data Processing #########
##################################


class EmotionPlainDataset(torch.utils.data.Dataset):
    """
    Dataset for the dataset of Emotions data set and plain text
    """

    def __init__(self, emotion_text, emotion_label, tokenizer, max_len):
        self.emotion_texts = emotion_text
        self.emotion_labels = emotion_label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.emotion_texts)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """
        emotion_text = self.emotion_texts[index]
        emotion_label = self.emotion_labels[index]

        encoded_emotion_text = self.tokenizer.encode_plus(
            emotion_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {
            'emotion_input_ids': encoded_emotion_text['input_ids'][0],
            'emotion_attention_mask': encoded_emotion_text['attention_mask'][0],
            'emotion_label': emotion_label
            # attention mask tells the model where tokens are padding
        }


# Load the EmpatheticDialogues dataset
emo_data = load_dataset('empathetic_dialogues')
training_emo_sentence = emo_data['train']['utterance']
lb = LabelBinarizer()
training_emo_label = lb.fit_transform(emo_data['train']['context'])
# Creating DataLoader
MAX_LEN = 32
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 64
text_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = EmotionPlainDataset(
    emotion_text=list(training_emo_sentence),
    emotion_label=list(training_emo_label),
    tokenizer=text_tokenizer,
    max_len=MAX_LEN
)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
# %%
#######################################
######## Model Initialization #########
#######################################
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# creating the discriminator
discriminator = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=32).to(DEVICE)
# %%
#######################################
########### Model Training ############
#######################################
NUM_EPOCHS = 10
LR = 1e-3  # learning rate
BETA1 = 0.5
TIME_STAMP = datetime.now().strftime("_%m_%d_%Y__%H_%M")
NUM_TRAINING_POINTS = 1000
SAVE_FILE = 'eval_' + TIME_STAMP + "/epochs.mat"
MODEL_FILE = 'eval_' + TIME_STAMP + "/models/"

if(not os.path.exists('eval_'+TIME_STAMP)):
    os.mkdir('eval_'+TIME_STAMP)
if(not os.path.exists('eval_' + TIME_STAMP + "/models")):
    os.mkdir('eval_' + TIME_STAMP + "/models")
# %%
criterion = nn.BCELoss()
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(),
                        lr=LR, betas=(BETA1, 0.999))
# Learning rate scheduler
lr_scheduler_D = get_scheduler(
    "linear",
    optimizer=optimizerD,
    num_warmup_steps=50,
    num_training_steps=NUM_TRAINING_POINTS * NUM_EPOCHS
)
D_losses = []
print("Starting Training Loop...")
# For each epoch
for epoch in range(NUM_EPOCHS):
    # For each batch in the dataloader
    for i, data in enumerate(train_dataloader, 0):

        # Format batch
        input_ids_real = data['emotion_input_ids'].to(DEVICE)
        input_ids_real_mask = data['emotion_attention_mask'].to(DEVICE)
        input_label = torch.argmax(data['emotion_label'].to(DEVICE), dim=1)
        b_size = input_ids_real.size(0)
        # Forward pass real batch through D
        output = discriminator(
            input_ids_real, attention_mask=input_ids_real_mask, labels=input_label)
        # Calculate loss on all-real batch
        errD_real = output.loss
        # Calculate gradients for D in backward pass
        errD_real.backward()
        optimizerD.step()
        lr_scheduler_D.step()
        optimizerD.zero_grad()
        errD = output.loss.mean().item()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f'
                  % (epoch, NUM_EPOCHS, i, NUM_TRAINING_POINTS,
                     errD))
        D_losses.append(errD)
    if epoch % 10 == 0:
        savemat(SAVE_FILE, {'discriminator_loss': D_losses})
torch.save(discriminator, MODEL_FILE+"disciminator.pt")
# %%
