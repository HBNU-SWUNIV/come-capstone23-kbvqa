import torch
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from utils import plot_acc, plot_loss
import os 
import time
from datetime import timedelta

total_train_loss = []
total_train_acc = []

total_valid_loss = []
total_valid_acc = []

total_A_valid_loss = []
total_A_valid_acc = []

total_B_valid_loss = []
total_B_valid_acc = []

def train_fn(model, train_loader, A_valid_loader, B_valid_loader, criterion, optimizer, device, n_epoch, save_path):
    A_valid_acc = 0
    B_valid_acc = 0
    
    best_acc = 0
    
    start = time.process_time()

    for epoch in range(n_epoch):
        
        start_epoch = time.process_time()
             
        train_total_num = 0
        A_valid_total_num = 0
        B_valid_total_num = 0

        train_count_correct = 0
        A_valid_count_correct = 0
        B_valid_count_correct = 0
        
        train_loss = 0
        A_valid_loss = 0
        B_valid_loss = 0

        model.train()
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), leave=False): 
            optimizer.zero_grad()
            imgs = batch['image'].to(device)  
            q_bert_ids = batch['ids'].to(device) 
            q_bert_mask = batch['mask'].to(device) 
            
            answers = batch['answer'].to(device) 
            answers = answers.squeeze()


            outputs = model(q_bert_ids, q_bert_mask, imgs) 

            loss = criterion(outputs, answers)

            train_loss += float(loss)
            loss.backward(loss)
            optimizer.step()

            '''
            acc
            '''
            predicted = torch.argmax(outputs, dim=1)
            count_correct = np.count_nonzero((np.array(predicted.cpu())==np.array(answers.cpu())) == True)      
            train_count_correct += count_correct
            train_total_num += answers.size(0)
            
        train_loss /= len(train_loader)

        train_acc = train_count_correct/train_total_num

        '''
        answer1 validation
        '''
        model.eval()
        for idx, batch in tqdm(enumerate(A_valid_loader), total=len(A_valid_loader), leave=False):
            with torch.no_grad():
                imgs = batch['image'].to(device)
                q_bert_ids = batch['ids'].to(device)
                q_bert_mask = batch['mask'].to(device)
                answers = batch['answer'].to(device) 
                answers = answers.squeeze()
                
                outputs = model(q_bert_ids, q_bert_mask, imgs)
                
                loss = criterion(outputs, answers)

                A_valid_loss += float(loss)

                predicted = torch.argmax(outputs, dim=1)
                count_correct = np.count_nonzero((np.array(predicted.cpu())==np.array(answers.cpu())) == True)      
                A_valid_count_correct += count_correct
                A_valid_total_num += answers.size(0)
            
        A_valid_loss /= len(A_valid_loader)
        A_valid_acc = A_valid_count_correct/A_valid_total_num

        '''
        answer2 validation
        '''
        model.eval()
        for idx, batch in tqdm(enumerate(B_valid_loader), total=len(B_valid_loader), leave=False):
            with torch.no_grad():
                imgs = batch['image'].to(device)
                q_bert_ids = batch['ids'].to(device)
                q_bert_mask = batch['mask'].to(device)
                answers = batch['answer'].to(device) 
                answers = answers.squeeze()
                outputs = model(q_bert_ids, q_bert_mask, imgs)
                loss = criterion(outputs, answers)
                B_valid_loss += float(loss)

                predicted = torch.argmax(outputs, dim=1)
                count_correct = np.count_nonzero((np.array(predicted.cpu())==np.array(answers.cpu())) == True)      
                B_valid_count_correct += count_correct
                B_valid_total_num += answers.size(0)
            
        B_valid_loss /= len(B_valid_loader)
        B_valid_acc = B_valid_count_correct/B_valid_total_num

        valid_acc = (A_valid_count_correct + B_valid_count_correct) / (A_valid_total_num + B_valid_total_num)
        
        if valid_acc > best_acc:
            best_model = deepcopy(model)
            best_acc = valid_acc
        
        total_train_acc.append(train_acc*100)
        total_train_loss.append(train_loss)

        total_A_valid_loss.append(A_valid_loss)
        total_A_valid_acc.append(A_valid_acc*100)
        total_B_valid_loss.append(B_valid_loss)
        total_B_valid_acc.append(B_valid_acc*100)

        
        end_epoch = time.process_time()
        
        print(f"[{epoch+1}/{n_epoch}] [TRAIN LOSS: {train_loss:.4f}] [TRAIN ACC: {train_acc:.4f}] [VALID ACC: {valid_acc:.4f}] [A TYPE VALID ACC: {A_valid_acc:.4f}] [B TYPE VALID ACC: {B_valid_acc:.4f}]")
        epoch_time = timedelta(seconds=end_epoch - start_epoch)
        total_time = timedelta(seconds=end_epoch - start)
        print(f"Epoch elapsed time : {epoch_time} | Total elapsed time : {total_time}")
        print()
    os.makedirs(save_path, exist_ok=True)
    last_model = deepcopy(model)
    best_path = f'{save_path}infer_model_{best_acc*100:.2f}.pt'
    plot_acc(total_train_acc, total_A_valid_acc, total_B_valid_acc, n_epoch, save_path)
    plot_loss(total_train_loss,total_A_valid_loss, total_B_valid_loss, n_epoch, save_path)

    torch.save(best_model, best_path)
    print(f"===== Model Saved in {best_path} =====")
    # torch.save(last_model, last_path)
    return best_path, best_acc