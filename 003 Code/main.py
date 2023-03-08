import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"]="0,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import get_args, get_dataloader, get_answerlist, seedEverything
from focal import FocalLoss
from train import train_fn
from model import VQAModel
from make_csv import preprocess
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_args()
    preprocess(config.lang)

    answer_list = get_answerlist(config.lang)

    seedEverything(2022)

    current_time = datetime.now().strftime(r'%m%d_%H%M')
    SAVE_PATH = './results/'+config.train_data
    if config.use_transformer_layer:
        SAVE_PATH += '_tf'
    else:
        SAVE_PATH += '_'

    SAVE_PATH += current_time + '/'
    
    print(str(config))
    model = VQAModel(len(answer_list), dim_i = 768, dim_h = 1024, config=config)
    model = nn.DataParallel(model)
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr = config.lr, weight_decay = config.weight_decay)

    if config.use_weight:
        weight = torch.tensor(answer_list['weight'], dtype=torch.float).to(DEVICE)

    if config.use_focal:
        criterion = FocalLoss(weight = weight)
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader, A_valid_loader, B_valid_loader = get_dataloader(config)

    best_path, metric = train_fn(model, train_loader, A_valid_loader, B_valid_loader, criterion, optimizer, DEVICE, config.n_epoch, SAVE_PATH)

    with open(f"{SAVE_PATH}/config.txt", "w") as f:
        f.write(str(config))
        f.write("\nTrain : [Image, Question], Target : [Answer]") 
