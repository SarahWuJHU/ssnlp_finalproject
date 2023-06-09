# %%
import warnings
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
import wikipedia
import re
import nltk
nltk.download('punkt')
# tested with this, might need a custom tokenizer for the actual inputs

warnings.filterwarnings("ignore")

###################################
######## Customized Model #########
###################################


class TextAndEmotionEncoder(BertModel):
    def __init__(self,
                 base_encoder: BertGenerationEncoder,
                 num_emotions: int,
                 hidden_size: int,
                 num_layers: int):
        super(TextAndEmotionEncoder, self).__init__(config=BertConfig(
            is_encoder=True
        ))
        self.base_encoder = base_encoder
        # num_layers should be at least 2
        self.linears = nn.ModuleList(
            [nn.Linear(num_emotions, hidden_size), nn.Dropout(0.1), nn.ReLU()])
        for _ in range(num_layers - 2):
            self.linears.extend(
                [nn.Linear(hidden_size, hidden_size), nn.Dropout(0.1), nn.ReLU()])
        self.linears.append(
            nn.Linear(hidden_size, base_encoder.config.hidden_size))

    def forward(self, input, attention_mask=None, **kwargs):
        input_ids, emotion_label = input
        outputs_base = self.base_encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        outputs_text = outputs_base[0]
        outputs_emotion = emotion_label
        for linear in self.linears:
            outputs_emotion = linear(outputs_emotion)
        output = outputs_text + outputs_emotion[:, None, :]
        outputs_base["last_hidden_state"] = output
        return outputs_base

# %%
##################################
######## Data Processing #########
##################################


class EmotionPlainDataset(torch.utils.data.Dataset):
    """
    Dataset for the dataset of Emotions data set and plain text
    """

    def __init__(self, plain_text, emotion_text, emotion_label, tokenizer, max_len, w2v_encoder):
        self.plain_texts = plain_text
        self.emotion_texts = emotion_text
        self.emotion_labels = emotion_label
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.w2v_encoder = w2v_encoder

    def __len__(self):
        return len(self.plain_texts)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """

        plain_text = self.plain_texts[index]
        emotion_text = self.emotion_texts[index]
        emotion_label = self.emotion_labels[index]

        # encode_plus will encode the input and return a dictionary of tensors
        encoded_plain_text = self.tokenizer.encode_plus(
            plain_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
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
        label = embeddings[emotion_label]

        return {
            # we only have one example in the batch
            'plain_input_ids': encoded_plain_text['input_ids'][0],
            'emotion_input_ids': encoded_emotion_text['input_ids'][0],
            'plain_attention_mask': encoded_plain_text['attention_mask'][0],
            'emotion_attention_mask': encoded_emotion_text['attention_mask'][0],
            # attention mask tells the model where tokens are padding
            'labels': torch.tensor(label)
        }


# Loading the Wikipedia data
DATA_SET = ""
EXCLUDED_SECTIONS = ['See also', 'References',
                     'Further reading', 'External links']
ARTICLE_TITLES = wikipedia.search('movie', results=800)
MARKDOWN_PATTERN = re.compile(r'(\'{2,5})(.*?)\1')

cleaned_articles = []
wikipedia.set_lang("en")
training_plain_sentence = []
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
              training_plain_sentence.extend(sentences)
    except:
      print(article_title, 'invalid')
# Load the EmpatheticDialogues dataset
emo_data = load_dataset('empathetic_dialogues')
training_emo_sentence = emo_data['train']['utterance'][:len(
    training_plain_sentence)]
training_emo_label = emo_data['train']['context'][:len(
    training_plain_sentence)]
embeddings = gensim.downloader.load('glove-twitter-50')

# Creating DataLoader
MAX_LEN = 32
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 32
text_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = EmotionPlainDataset(
    plain_text=list(training_plain_sentence),
    emotion_text=list(training_emo_sentence),
    emotion_label=list(training_emo_label),
    tokenizer=text_tokenizer,
    max_len=MAX_LEN,
    w2v_encoder=embeddings
)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
# %%
#######################################
######## Model Initialization #########
#######################################
NUM_EMOTIONS = 50  # dimension of emotions embedding
HIDDEN_SIZE = 64  # size of hidden state
NUM_LAYERS = 3  # number of layers
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# constructing the encoder
# tested with bert-base-uncased, should probably be bert-large-uncased in actual training
base_encoder = BertGenerationEncoder.from_pretrained(
    MODEL_NAME).to(DEVICE)
encoder = TextAndEmotionEncoder(base_encoder, num_emotions=NUM_EMOTIONS, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(
    DEVICE)  # just temporary parameters, num_emotions especially is definitely not 3
# creating encoder-decoder model
# tested with bert-base-uncased, should probably be bert-large-uncased in actual training
base_decoder = BertGenerationDecoder.from_pretrained(
    MODEL_NAME, add_cross_attention=True, is_decoder=True).to(DEVICE)
input_reconstructor = EncoderDecoderModel(
    encoder=encoder, decoder=base_decoder).to(DEVICE)
# creating the generator
# tested with bert-base-uncased, should probably be bert-large-uncased in actual training
base_decoder_2 = BertGenerationDecoder.from_pretrained(
    MODEL_NAME, add_cross_attention=True, is_decoder=True).to(DEVICE)
generator = EncoderDecoderModel(
    encoder=encoder, decoder=base_decoder_2).to(DEVICE)
# creating the discriminator
discriminator = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2).to(DEVICE)
# %%
#######################################
########### Model Training ############
#######################################
NUM_EPOCHS = 20
LR = 1e-5  # learning rate
GENERATOR_LR = 0.1
BETA1 = 0.5
TIME_STAMP = datetime.now().strftime("_%m_%d_%Y__%H_%M")
NUM_TRAINING_POINTS = 1000
SAVE_FILE = 'trial_' + TIME_STAMP + "/epochs.mat"
MODEL_FILE = 'trial_' + TIME_STAMP + "/models/"

if(not os.path.exists('trial_'+TIME_STAMP)):
    os.mkdir('trial_'+TIME_STAMP)  # Create cross validation folder f.
if(not os.path.exists('trial_' + TIME_STAMP + "/models")):
    # Create cross validation folder f.
    os.mkdir('trial_' + TIME_STAMP + "/models")

criterion = nn.BCELoss()
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(),
                        lr=LR, betas=(BETA1, 0.999))
optimizerE = optim.Adam(input_reconstructor.parameters(),
                        lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(generator.parameters(),
                        lr=GENERATOR_LR, betas=(BETA1, 0.999))
# Learning rate scheduler
lr_scheduler_D = get_scheduler(
    "linear",
    optimizer=optimizerD,
    num_warmup_steps=50,
    num_training_steps=NUM_TRAINING_POINTS * NUM_EPOCHS
)
lr_scheduler_E = get_scheduler(
    "linear",
    optimizer=optimizerE,
    num_warmup_steps=50,
    num_training_steps=NUM_TRAINING_POINTS * NUM_EPOCHS
)
lr_scheduler_G = get_scheduler(
    "linear",
    optimizer=optimizerG,
    num_warmup_steps=50,
    num_training_steps=NUM_TRAINING_POINTS * NUM_EPOCHS
)

G_losses = []
D_losses = []
E_losses = []

print("Starting Training Loop...")
# For each epoch
for epoch in range(NUM_EPOCHS):
    # For each batch in the dataloader
    for i, data in enumerate(train_dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        input_ids_real = data['emotion_input_ids'].to(DEVICE)
        input_ids_real_mask = data['emotion_attention_mask'].to(DEVICE)
        b_size = input_ids_real.size(0)
        label = torch.full((b_size, 2), real_label,
                           dtype=torch.float, device=DEVICE)
        label[:, 1] = fake_label
        # Forward pass real batch through D
        output = discriminator(
            input_ids_real, attention_mask=input_ids_real_mask, labels=label)
        # Calculate loss on all-real batch
        errD_real = output.loss
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.loss.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        input_ids_fake = data['plain_input_ids'].to(DEVICE)
        input_ids_fake_mask = data['plain_attention_mask'].to(DEVICE)
        emotion_label = data['labels'].to(DEVICE)
        # Generate fake image batch with G
        fake = generator(input=(input_ids_fake, emotion_label), decoder_input_ids=input_ids_fake, attention_mask=input_ids_fake_mask)[
            "logits"].argmax(dim=-1)
        label.fill_(fake_label)
        label[:, 1] = real_label
        # Classify all fake batch with D
        output = discriminator(
            fake.detach(), attention_mask=input_ids_fake_mask, labels=label)
        # Calculate D's loss on the all-fake batch
        errD_fake = output.loss
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.loss.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()
        lr_scheduler_D.step()
        optimizerD.zero_grad()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        label[:, 1] = fake_label
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(
            fake.detach(), attention_mask=input_ids_fake_mask, labels=label)
        # Calculate G's loss based on this output
        errG = output.loss * 10
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.loss.mean().item()
        # Update G
        optimizerG.step()
        lr_scheduler_D.step()
        optimizerD.zero_grad() 

        ############################
        # (3) Regulate network: minimize reconstruction loss
        ###########################
        input_reconstructor.zero_grad()
        y = input_reconstructor(input=(input_ids_fake, emotion_label),
                                decoder_input_ids=input_ids_fake, attention_mask=input_ids_fake_mask, labels=input_ids_fake)
        errE = y.loss
        errE.backward()
        # Update G
        optimizerE.step()
        lr_scheduler_E.step()
        optimizerE.zero_grad()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t Loss_R: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, NUM_EPOCHS, i, NUM_TRAINING_POINTS,
                     errD.item(), errG.item(), errE.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        E_losses.append(errE.item())
    if epoch % 10 == 0:
        savemat(SAVE_FILE, {'generator_loss': G_losses,
                            'discriminator_loss': D_losses, 'reconstruction_loss': E_losses})
torch.save(generator, MODEL_FILE+"generator.pt")
torch.save(discriminator, MODEL_FILE+"disciminator.pt")
torch.save(input_reconstructor, MODEL_FILE+"input_reconstructor.pt")