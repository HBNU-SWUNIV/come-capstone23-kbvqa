import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from vqa_dataset import VQADataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import os
import numpy as np
from transformers import XLMRobertaTokenizer

import torch
import torch.nn as nn
from transformers import AutoModel
import timm
import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from PIL import Image
from make_csv import preprocess

device = 'cuda'
import unicodedata

def fill_str_with_space(input_s="", max_size=20, fill_char=" "):
    l = 0 
    for c in input_s:
        if unicodedata.east_asian_width(c) in ['F', 'W']:
            l+=2
        else: 
            l+=1
    return input_s+fill_char*(max_size-l)



def get_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--infer_data', default='all', choices=['abstract', 'triple', 'all', 'vqa'])
    parser.add_argument('--model_path', type=str, required=False, default="./results/selected/infer_model.pt")
    parser.add_argument('--lang', required=False, default='ko', choices=['ko', 'en'])
    
    config = parser.parse_args()
    return config

config = get_args()
preprocess(config.lang)

class VQAModel(nn.Module):
    def __init__(self, num_target, dim_i, dim_h=1024, config=None):
        super(VQAModel, self).__init__()
        self.config = config
        self.dim_i = dim_i
        self.bert = AutoModel.from_pretrained('xlm-roberta-base')

        self.i_model = timm.create_model('resnet50',pretrained=True) 
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
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dropout=0.2).to(device)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3).to(device)
            uni_f = transformer_encoder(uni_f)

        outputs = self.out_linear(self.relu(self.drop(self.h_layer_norm(self.linear(uni_f)))))

        return outputs

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, answer_list, max_token, transform=None):
        
        self.tokenizer = tokenizer
        self.data = data
        self.max_token = max_token
        self.answer_list = answer_list      
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        qustion = self.data['question'][index] 
        answer = self.data['answer'][index] 
        img_loc = self.data['img_path'][index] 

        tokenized = self.tokenizer.encode_plus("".join(qustion),
                                     None,
                                     add_special_tokens=True,
                                     max_length = self.max_token,
                                     truncation=True,
                                     pad_to_max_length = True
                                              )
        
        ids = tokenized['input_ids']
        mask = tokenized['attention_mask']
        image = Image.open(img_loc).convert('RGB')  
        image = self.transform(image)  
        answer_ids1 = self.answer_list[self.answer_list['answer']==answer].index 


        return {'ids': torch.tensor(ids, dtype=torch.long), 
                'mask': torch.tensor(mask, dtype=torch.long),
                'answer': torch.tensor(answer_ids1, dtype=torch.long),
                'image': image}

def get_answerlist():
    data = pd.read_csv(f"./data/data_{config.lang}.csv")
    data = data[data['split'] != 'test']
    data = data[['img_path', 'question', 'answer']]
    data = data.dropna()
    answer_list = data['answer'].value_counts().reset_index()
    answer_list.columns=['answer', 'count']
    answer_list['weight'] = 1 - answer_list['count']/answer_list['count'].sum()

    return answer_list

def answering(img_file, question, answer):
    # with torch.no_grad():
        model.eval()
        img = transform(Image.open(img_file).convert("RGB")).unsqueeze(0)
        img = img.to(device)
        encoded = tokenizer.encode_plus("".join(question),
                                        None,
                                        add_special_tokens=True,
                                        max_length = 50,
                                        truncation=True,
                                        pad_to_max_length = True
                                                )

        ids, mask = encoded['input_ids'], encoded['attention_mask']
        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(device)
        output = model(ids, mask, img)
        predicted = torch.argmax(output, dim=1).item()
        pred_ans = answer_list.loc[predicted]['answer']
        if pred_ans == answer:
            return True, pred_ans, answer
        else:
            return False, pred_ans, answer


print("Load Model ...")
model = torch.load(config.model_path)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

answer_list = get_answerlist()

data = pd.read_csv(f'./data/data_{config.lang}.csv')
data = data[data['split']=='test']
# if config.infer_data == 'triple':
#     data = data[data['kb_source'] == 'triple']

# if config.infer_data == 'abstract':
#     data = data[data['kb_source'] == 'abstract']

# if config.infer_data == 'vqa':
#     data = data[data['kb_source'] == 'vqa']

test_data = data.reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

print("Start Inference ...")
correct_count = 0
total = len(test_data)
file_ = open("./infer_log.txt", "a")

for index in trange(len(test_data)):
    is_correct, pred_ans, gold_label = answering(test_data['img_path'][index], test_data['question'][index], test_data['answer'][index])
    print_string = f"[{index+1:5d}] GOLD LABEL : {fill_str_with_space(gold_label, max_size=20)} PRED : {fill_str_with_space(pred_ans, max_size=20)}"
    

    if is_correct:
        correct_count += 1
        print_string += "(O)"
    else:
        print_string += "(X)"
    
    # print(print_string)
    file_.write(print_string+"\n")

print_string = f"Inference Accuracy : {(correct_count/total)*100:.2f}%"
# print(print_string)
file_.write(print_string+"\n")
file_.close()




