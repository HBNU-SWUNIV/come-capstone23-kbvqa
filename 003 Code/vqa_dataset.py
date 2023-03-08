import torch
from PIL import Image


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