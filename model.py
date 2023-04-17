import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, BertModel, BertConfig
import torch.optim as optim
from transformers import BertGenerationEncoder, BertGenerationDecoder, BertGenerationConfig
from transformers import get_scheduler
# tested with this, might need a custom tokenizer for the actual inputs
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import EncoderDecoderModel
from scipy.io import savemat

import warnings
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
        output = torch.cat((outputs_text, outputs_emotion[:, None, :]), dim=1)
        outputs_base["last_hidden_state"] = output
        return outputs_base


##################################
######## Data Processing #########
##################################
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_ids = tokenizer("This is a long article to summarize",
                      add_special_tokens=False, return_tensors="pt").input_ids  # random sample text
# index of whatever the emotion is (0, 1, 2, etc.)
emotion_label = torch.rand((1, 50))

#######################################
######## Model Initialization #########
#######################################
NUM_EMOTIONS = 50  # dimension of emotions embedding
HIDDEN_SIZE = 256  # size of hidden state
NUM_LAYERS = 4  # number of layers
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# constructing the encoder
# tested with bert-base-uncased, should probably be bert-large-uncased in actual training
base_encoder = BertGenerationEncoder.from_pretrained(
    'bert-base-uncased').to(DEVICE)
encoder = TextAndEmotionEncoder(base_encoder, num_emotions=NUM_EMOTIONS, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(
    DEVICE)  # just temporary parameters, num_emotions especially is definitely not 3
# creating encoder-decoder model
# tested with bert-base-uncased, should probably be bert-large-uncased in actual training
base_decoder = BertGenerationDecoder.from_pretrained(
    'bert-base-uncased', add_cross_attention=True, is_decoder=True).to(DEVICE)
input_reconstructor = EncoderDecoderModel(
    encoder=encoder, decoder=base_decoder).to(DEVICE)
# creating the generator
# tested with bert-base-uncased, should probably be bert-large-uncased in actual training
base_decoder_2 = BertGenerationDecoder.from_pretrained(
    'bert-base-uncased', add_cross_attention=True, is_decoder=True).to(DEVICE)
generator = EncoderDecoderModel(
    encoder=encoder, decoder=base_decoder_2).to(DEVICE)
# creating the discriminator
discriminator = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=1).to(DEVICE)

#######################################
########### Model Training ############
#######################################
BATCH_SIZE = 32
NUM_EPOCHS = 100
LR = 1e4  # learning rate
BETA1 = 0.5
NUM_TRAINING_POINTS = 1000
SAVE_FILE = "epochs.mat"

criterion = nn.BCELoss()
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(),
                        lr=LR, betas=(BETA1, 0.999))
optimizerE = optim.Adam(input_reconstructor.parameters(),
                        lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
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
x = encoder((input_ids, emotion_label))
print(x[0].shape)

print("Starting Training Loop...")
# For each epoch
for epoch in range(NUM_EPOCHS):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        inpu_ids_real = data['emotional_text'].to(DEVICE)
        b_size = inpu_ids_real.size(0)
        label = torch.full((b_size,), real_label,
                           dtype=torch.float, device=DEVICE)
        # Forward pass real batch through D
        output = discriminator(inpu_ids_real).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        inpu_ids_real = data['plain_text'].to(DEVICE)
        # Generate fake image batch with G
        fake = generator(input=(input_ids, emotion_label), decoder_input_ids=input_ids)[
            "logits"].argmax(dim=-1)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
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
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        lr_scheduler_D.step()
        optimizerD.zero_grad()

        ############################
        # (3) Regulate network: minimize reconstruction loss
        ###########################
        input_reconstructor.zero_grad()
        y = input_reconstructor(input=(input_ids, emotion_label),
                        decoder_input_ids=input_ids, label=input_ids)
        errE = y.loss()
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

savemat(SAVE_FILE, {'generator_loss': G_losses, 'discriminator_loss': D_losses, 'reconstruction_loss': E_losses})
# input_ids.shape should be batch_size x seq_len, emotion_label.shape should be batch_size x 1
x = encoder((input_ids, emotion_label))
print(x[0].shape)

# testing the encoder-decoder model
# this is how inputs need to be given to the encoder-decoder model
y = input_reconstructor(input=(input_ids, emotion_label),
                        decoder_input_ids=input_ids)
print(y[0].shape)
print(y[0].argmax(dim=-1))

# testing the discriminator
y = generator(input=(input_ids, emotion_label), decoder_input_ids=input_ids)[
    "logits"].argmax(dim=-1)
print(discriminator(y))
print(discriminator(input_ids))
