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
def get_args():
    parser = argparse.ArgumentParser()

    # Train Config
    parser.add_argument('--n_epoch', type=int, required=False, default=50)
    parser.add_argument('--lr', type=float, required=False, default=3e-5)
    parser.add_argument('--weight_decay', type=float, required=False, default=0.001)
    parser.add_argument('--batch_size', type=int, required=False, default=512)

    # Loss Config
    parser.add_argument('--focal_gamma', type=float, required=False, default=2.0)
    parser.add_argument('--use_weight', action='store_true')
    parser.add_argument('--use_focal', action='store_true')

    # Model Config
    parser.add_argument('--use_transformer_layer', action='store_true', default=False)

    # Dataset Config
    parser.add_argument('--train_data', default='all', choices=['A', 'B', 'all'])
    
    # Tokenizer Config
    parser.add_argument('--max_token', type=int, required=False, default=50)

    # Language Config
    parser.add_argument('--lang', required=False, default='ko', choices=['ko', 'en'])
    
    config = parser.parse_args()
    return config

def get_answerlist(lang):
    data = pd.read_csv(f"./data/data_{lang}.csv")
    data = data[data['split'] != 'test']
    data = data[['img_path', 'question', 'answer']]
    data = data.dropna()
    answer_list = data['answer'].value_counts().reset_index()
    answer_list.columns=['answer', 'count']
    answer_list['weight'] = 1 - answer_list['count']/answer_list['count'].sum()

    return answer_list

def get_dataloader(config):

    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    answer_list = get_answerlist(config.lang)
    
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    data = pd.read_csv(f"./data/data_{config.lang}.csv")

    train_data = data[data['split'] == 'train']
    
    A_train_data = train_data[train_data['kb_source'] != 'vqa']
    B_train_data = train_data[train_data['kb_source'] == 'vqa'] 

    valid_data = data[data['split'] == 'valid']
    A_valid_data = valid_data[valid_data['kb_source'] != 'vqa']
    B_valid_data = valid_data[valid_data['kb_source'] == 'vqa']

    A_train_data = A_train_data.reset_index(drop=True)
    B_train_data = B_train_data.reset_index(drop=True)

    A_valid_data = A_valid_data.reset_index(drop=True)
    B_valid_data = B_valid_data.reset_index(drop=True)

    '''
    get train dataset
    '''
    print("=======================================================")
    print(f"# A type Train : {len(A_train_data)}, # B type Train : {len(B_train_data)}")
    print(f"# A type Valid : {len(A_valid_data)}, # B type Valid : {len(B_valid_data)}")
    if config.train_data == 'A': #triple
        train_data = A_train_data[A_train_data['kb_source']=='triple'].reset_index(drop=True)
        A_valid_data = A_valid_data[A_valid_data['kb_source']=='triple'].reset_index(drop=True)
        print(f"# Training data [A type]: {len(train_data)}")

    if config.train_data == 'B':
        train_data = B_train_data.reset_index(drop=True)
        print(f"# Training data [B type]: {len(train_data)}")

    if config.train_data == 'all':
        train_data = pd.concat([A_train_data, B_train_data]).reset_index(drop=True)
        print(f"# Training data [A type, B type]: {len(train_data)}")

    print("=======================================================")


    train_dataset = VQADataset(tokenizer, train_data, answer_list, config.max_token, transform) 
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=10, shuffle=True, pin_memory=True)

    A_valid_dataset = VQADataset(tokenizer, A_valid_data, answer_list, config.max_token, transform) 
    A_valid_loader = DataLoader(dataset=A_valid_dataset, batch_size=config.batch_size, num_workers=10, shuffle=True, pin_memory=True)

    B_valid_dataset = VQADataset(tokenizer, B_valid_data, answer_list, config.max_token, transform) 
    B_valid_loader = DataLoader(dataset=B_valid_dataset, batch_size=config.batch_size, num_workers=10, shuffle=True, pin_memory=True)

    return train_loader, A_valid_loader, B_valid_loader

def plot_acc(train_acc, A_valid_acc, B_valid_acc, n_epoch, path):
    fig = plt.figure(figsize=(8,6))
    x = range(n_epoch)

    plt.title("Accuracy", fontsize=18)
    plt.grid()
    plt.ylim([0, 100])
    plt.xticks(range(0, n_epoch, 5))
    plt.xlim([0, n_epoch])

    plt.plot(x, train_acc, 'b-',label="Train")
    plt.plot(x, A_valid_acc, 'c-',label="A Type Valid")
    plt.plot(x, B_valid_acc, 'm-',label="B Type Valid")

    plt.xlabel("Training epoch")
    plt.ylabel("Accuracy(%)")
    plt.legend(loc = 'best')

    plt.savefig(path+"Accuracy.png")
    plt.close()
    
def plot_loss(train_loss, A_valid_loss, B_valid_loss, n_epoch, path):
    fig = plt.figure(figsize=(8,6))

    x = range(n_epoch)
    plt.plot(x, train_loss, 'b-',label='Train')
    plt.plot(x, A_valid_loss, 'c-', label='A Type Valid')
    plt.plot(x, B_valid_loss, 'm-',label='B Type Valid')


    plt.title("Loss", fontsize=18)
    plt.grid()

    plt.ylim([0, 9])
    plt.xticks(range(0, n_epoch, 5))
    plt.xlim([0, n_epoch])

    plt.xlabel("Training epoch")
    plt.ylabel("Loss")
    plt.legend(loc = 'best')

    plt.savefig(path+"Loss.png")
    plt.close()

    
import torch
def seedEverything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False