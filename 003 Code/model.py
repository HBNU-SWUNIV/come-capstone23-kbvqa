import torch
import torch.nn as nn
from transformers import AutoModel
import timm
import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VQAModel(nn.Module):
    def __init__(self, num_target, dim_i, dim_h=1024, config=None):
        super(VQAModel, self).__init__()
        self.config = config
        self.dim_i = dim_i
        self.bert = AutoModel.from_pretrained('xlm-roberta-base')

        self.i_model = timm.create_model('resnet50', pretrained=True) 
        self.i_model.fc = nn.Linear(self.i_model.fc.in_features, dim_i) 
        self.i_drop = nn.Dropout(0.25)
        self.linear = nn.Linear(dim_i, dim_h)

        self.h_layer_norm = nn.LayerNorm(dim_h)
        self.layer_norm = nn.LayerNorm(num_target)

        self.relu = nn.ReLU()
        self.out_linear = nn.Linear(dim_h, num_target)
        self.drop = nn.Dropout(0.2)
        self.tanh = nn.Tanh()
        
    def forward(self, idx, mask, image):
        q_f = self.bert(idx, mask) 
        q_f = q_f.pooler_output
        q_f = q_f
        i_f = self.i_drop(self.tanh(self.i_model(image))) 
        uni_f = i_f * q_f

        if self.config.use_transformer_layer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=uni_f.shape[1], nhead=8, dropout=0.2).to(DEVICE)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3).to(DEVICE)
            uni_f = transformer_encoder(uni_f)

        outputs = self.out_linear(self.relu(self.drop(self.h_layer_norm(self.linear(uni_f)))))

        return outputs