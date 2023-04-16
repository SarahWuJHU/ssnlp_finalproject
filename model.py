import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, BertModel, BertConfig
from transformers import BertGenerationEncoder, BertGenerationDecoder, BertGenerationConfig
from transformers import BertTokenizer # tested with this, might need a custom tokenizer for the actual inputs
from transformers import EncoderDecoderModel

import warnings
warnings.filterwarnings("ignore")

class TextAndEmotionEncoder(BertModel):
    def __init__(self,
                 base_encoder: BertGenerationEncoder,
                 num_emotions: int,
                 embedding_size: int,
                 hidden_size: int,
                 num_layers: int):
        super(TextAndEmotionEncoder, self).__init__(config=BertConfig(
            is_encoder=True
        ))

        self.base_encoder = base_encoder
        
        self.emotion_embedding = nn.Embedding(num_emotions, embedding_size)

        # num_layers should be at least 2
        self.linears = nn.ModuleList([nn.Linear(embedding_size, hidden_size), nn.ReLU()])
        for _ in range(num_layers - 2):
            self.linears.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.linears.append(nn.Linear(hidden_size, base_encoder.config.hidden_size))
        
    def forward(self, input, attention_mask=None, **kwargs):
        #import pdb; pdb.set_trace()
        input_ids, emotion_label = input
        outputs_text = self.base_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        
        outputs_emotion = self.emotion_embedding(emotion_label)
        for linear in self.linears:
            outputs_emotion =  linear(outputs_emotion)
        output = torch.cat((outputs_text, outputs_emotion), dim=1)

        return output


 # testing the encoder
base_encoder = BertGenerationEncoder.from_pretrained('bert-base-uncased') # tested with bert-base-uncased, should probably be bert-large-uncased in actual training
encoder = TextAndEmotionEncoder(base_encoder, num_emotions=3, embedding_size=128, hidden_size=256, num_layers=4) # just temporary parameters, num_emotions especially is definitely not 3
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_ids = tokenizer("This is a long article to summarize", add_special_tokens=False, return_tensors="pt").input_ids # random sample text
emotion_label = torch.tensor([1]).unsqueeze(0) # index of whatever the emotion is (0, 1, 2, etc.)
x = encoder((input_ids, emotion_label))  # input_ids.shape should be batch_size x seq_len, emotion_label.shape should be batch_size x 1
print(x.shape)

# creating encoder-decoder model
base_decoder = BertGenerationDecoder.from_pretrained('bert-base-uncased', add_cross_attention=True, is_decoder=True) # tested with bert-base-uncased, should probably be bert-large-uncased in actual training
input_reconstructor = EncoderDecoderModel(encoder=encoder, decoder=base_decoder)
y = input_reconstructor(input=(input_ids, emotion_label), decoder_input_ids=input_ids)