#%%
from bigbird_train import training
from MRC_hub_data import KoMRC, preprocess, preprocess_for_test
from MRC import KoMRCT
from postprocessing import NNG_SN_postpro, postprocess_cum_text, process_predicted
from transformers import (BigBirdConfig, 
                          BigBirdForQuestionAnswering,
                          BigBirdTokenizer,
                          BigBirdModel,
                          BertTokenizer,
                          BigBirdTokenizerFast,
                          AutoTokenizer)
import torch

#%% - ai hub, 기존데이터 합치기
dataset = KoMRC.load('/home/administrator/ingeol/NLP_Project/project2_Machine_Reading_Comprehension/train.json',
                     '/home/administrator/ingeol/NLP_Project/project2_Machine_Reading_Comprehension/VL_span_extraction.json')


tra_dataset, de_dataset = KoMRC.split(dataset)
#%%

config = BigBirdConfig()
# model = BigBirdModel(config)
tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-roberta-base")
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
loss_fn = torch.nn.CrossEntropyLoss()
# %%
train_dataset = preprocess(tokenizer, tra_dataset)
dev_dataset = preprocess(tokenizer, de_dataset)
#%% # 학습시키는 부분
training(model, tokenizer, loss_fn, train_dataset, dev_dataset, optimizer, epoch=3)

#%%
model = BigBirdForQuestionAnswering.from_pretrained('/home/administrator/ingeol/NLP_Project/project2_Machine_Reading_Comprehension/model.9')
model.cuda()
model.eval()

test_data = KoMRCT.load('/home/administrator/ingeol/NLP_Project/project2_Machine_Reading_Comprehension/test.json')
test_dataset = preprocess_for_test(tokenizer, test_data)
#%%
import re
for idx, sample in zip(range(1, 50), train_dataset):
    print(f'------{idx}------')
    #print('Context:', sample['context'])
    #print('Question:', sample['question'])
    
    input_ids, token_type_ids = [
        torch.tensor(sample[key], dtype=torch.long, device="cuda")
        for key in ("input_ids", "token_type_ids")
    ]
    
    # print(input_ids[None, :].squeeze(0).size(),token_type_ids[None, :].squeeze(0).size())
    # break
    with torch.no_grad():
        outputs = model(input_ids=input_ids[None, :].squeeze(0), token_type_ids=token_type_ids[None, :].squeeze(0))
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    #start_logits.squeeze_(0), end_logits.squeeze_(0)
    
    start_prob = start_logits[token_type_ids.bool()][1:-1].softmax(-1)
    end_prob = end_logits[token_type_ids.bool()][1:-1].softmax(-1)
    probability = torch.triu(start_prob[:, None] @ end_prob[None, :])
    index = torch.argmax(probability).item()
    
    start = index // len(end_prob)
    end = index % len(end_prob)
    #print(start, end)
    
    start = sample['offset_mapping'][0][start][0]
    end = sample['offset_mapping'][0][end][1]
    
    L = list(sample['context'][start:end])
    
    regex = r"[^\w\s]"
    clean_text = re.sub(regex, "", ''.join(L))

    # 명사, 숫자로 답변 압축
    #answer = NNG_SN_postpro(test_data[idx], clean_text)
    
    # 원하는 pos가 문장에 있을때 계속 더해주는 방식
    answer = process_predicted(clean_text)
    
    print('Answer:', clean_text)

# %%
import csv
from tqdm import tqdm
import os
import re

os.makedirs('out', exist_ok=True)
with torch.no_grad(), open('/home/administrator/ingeol/NLP_Project/project2_Machine_Reading_Comprehension/baseline_2.csv', 'w') as fd:
    writer = csv.writer(fd)
    writer.writerow(['Id', 'Predicted'])

    rows = []
    for sample in tqdm(range(len(test_dataset)), "Testing"):
        input_ids, token_type_ids = [
            torch.tensor(test_dataset[sample][key], dtype=torch.long, device="cuda")
            for key in ("input_ids", "token_type_ids")
        ]
    
        with torch.no_grad():
            outputs = model(input_ids=input_ids[None, :].squeeze(0), token_type_ids=token_type_ids[None, :].squeeze(0))
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
    
        start_prob = start_logits[token_type_ids.bool()][1:-1].softmax(-1)
        end_prob = end_logits[token_type_ids.bool()][1:-1].softmax(-1)
        probability = torch.triu(start_prob[:, None] @ end_prob[None, :])
        index = torch.argmax(probability).item()
    
        start = index // len(end_prob)
        end = index % len(end_prob)
        
        # 1024 를 넘어가는 부분 처리
        if start >1024 or end>1024:
            start,end = 0,0
        
        if start +15 < end:
            start_1 = start
            start = test_dataset[sample]['offset_mapping'][0][start_1][0]
            end = test_dataset[sample]['offset_mapping'][0][start_1][0]
        else:
            start = test_dataset[sample]['offset_mapping'][0][start][0]
            end = test_dataset[sample]['offset_mapping'][0][end][1]

        # 후처리
        L = list(test_dataset[sample]['context'][start:end]) 
        # 정규표현식을 사용하여 모든 특수문자를 제거
        regex = r"[^\w\s]"
        clean_text = re.sub(regex, "", ''.join(L))
        # 명사, 숫자로 답변 압축
        answer = process_predicted(clean_text)
        
        rows.append([test_dataset[sample]["guid"], answer])   
        
        #rows.append([sample["guid"], sample['context'][start:end]])
    
    writer.writerows(rows)