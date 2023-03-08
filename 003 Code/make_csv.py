import json
from tqdm import tqdm, trange
import pandas as pd
import os

def _preprocess(data, kb, lang, split):
    data_arr = []

    # index = 22
    for index in trange(len(data['annotations'])):

        a1_dict = {}
        a2_dict = {}

        cur_data = data['annotations'][index]
        img_path = "./data/image/"+cur_data['IMAGE_NAME']
        if not os.path.exists(img_path):
            continue


        A1_question = cur_data['questions'][0][f'question_{lang}']
        A1_answer = cur_data['questions'][0][f'answer_{lang}']
        A1_source = cur_data['questions'][0]['ans_source'] #kb_source

        if A1_source == 'triple':
            triple_str = ""
            for kb_id in cur_data['questions'][0]['fact']:
                try:
                    triple_str += kb[kb_id]['surface_ko'] + "="
                except:
                    triple_str = float('Nan')

            if type(triple_str) != float:
                triple_str = triple_str[:-1]
            abs_str = float('Nan')


        if A1_source == 'abstract':
            try:
                abs_str = kb[cur_data['questions'][0]['fact'][0]]['e2_label']
            except:
                abs_str = float('Nan')
            triple_str = float('Nan')

        if A1_source == 'multiple':
            for mul_idx in [0, 1]:
                mul_kb = cur_data['questions'][0]['fact'][mul_idx]
                try:
                    if kb[mul_kb]['surface_ko'] == None: 
                        try:
                            abs_str = kb[mul_kb]['e2_label'] # abstract 
                        except:
                            abs_str = float('Nan')
                    else:
                        try:
                            triple_str = kb[mul_kb]['surface_ko'] # triple
                        except:
                            triple_str = float('Nan')

                except:
                    try:
                        triple_str = kb[mul_kb]['surface_ko'] # triple
                    except:
                        triple_str = float('Nan')
                        abs_str = float('Nan')

        A2_question = cur_data['questions'][1][f'question_{lang}']
        A2_answer = cur_data['questions'][1][f'answer_{lang}']
        A2_source = cur_data['questions'][1]['ans_source'] #kb_source

        #question	answer	question_en	answer_en	question_type	Triple	Abstract	Answer_type	set
        a1_dict['img_path'] = img_path
        a1_dict['question'] = A1_question
        a1_dict['answer'] = A1_answer
        a1_dict['kb_source'] = A1_source
        a1_dict['triple'] = triple_str
        a1_dict['abstract'] = abs_str
        a1_dict['split'] = split

        a2_dict['img_path'] = img_path
        a2_dict['question'] = A2_question
        a2_dict['answer'] = A2_answer
        a2_dict['kb_source'] = 'vqa'
        a2_dict['triple'] = float('Nan')
        a2_dict['abstract'] = float('Nan')
        a2_dict['split'] = split

        data_arr.append(a1_dict)
        data_arr.append(a2_dict)

    return pd.DataFrame(data_arr)

def preprocess(lang):
    if not os.path.exists("./data/kb.json"):
        '''
        kb pre-processing
        '''
        with open("./data/knowledge_base.json") as f:
            kb = json.load(f)
            
        kb_dict = {}
        kb_dict['INFO'] = kb['INFO']

        each_dict = {}
        for d in tqdm(kb['knowledgebase']['fact']):
            fact_id = "/".join(d['fact_id'].split("/")[1:])
            each_dict[fact_id] = {
                'KB' : d['KB'],
                'e1_label' : d['e1_label'],
                'e2_label' : d['e2_label'],
                'surface_ko' : d['surface_ko'],
                'surface_en' : d['surface_en'],
                'sources' : d['sources'],
                'r' : d['r'],
                'e1' : d['e1'],
                'e2' : d['e2']
            }

        kb_dict['fact'] = each_dict

        with open('./data/kb.json','w') as f:
            json.dump(kb_dict,f, ensure_ascii=False, indent=4)



    if not os.path.exists(f"./data/data_{lang}.csv"):
        print("--- preprocessing data ---")
        with open("./data/kb.json") as f:
            kb = json.load(f)
            kb = kb['fact']

        with open("./data/train.json", "r") as f:
            train = json.load(f)

        with open("./data/valid.json", "r") as f:
            valid = json.load(f)

        with open("./data/test.json", "r") as f:
            test = json.load(f)

        train_df = _preprocess(train, kb, lang, 'train')
        valid_df = _preprocess(valid, kb, lang, 'valid')
        test_df = _preprocess(test, kb, lang, 'test')

        data_df = pd.concat([train_df, valid_df, test_df]).reset_index(drop=True)
        data_df.to_csv(f"./data/data_{lang}.csv", index=False)

if __name__ == '__main__':
    preprocess("ko")
