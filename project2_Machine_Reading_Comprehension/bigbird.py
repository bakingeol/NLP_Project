#%%
from bigbird_train import training
from MRC import KoMRC, preprocess, preprocess_for_test
from transformers import (BigBirdConfig, 
                          BigBirdForQuestionAnswering,
                          BigBirdTokenizer,
                          BigBirdModel,
                          BertTokenizer,
                          BigBirdTokenizerFast,
                          AutoTokenizer)
import torch
# %%
dataset = KoMRC.load('/home/administrator/ingeol/NLP_Project/project2_Machine_Reading_Comprehension/train.json')
tra_dataset, de_dataset = KoMRC.split(dataset)
#%%
tra_dataset[0]
# %%
config = BigBirdConfig()
model = BigBirdModel(config)
tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
#tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-roberta-base")

optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
loss_fn = torch.nn.CrossEntropyLoss()
# %%
train_dataset = preprocess(tokenizer, tra_dataset)
dev_dataset = preprocess(tokenizer, de_dataset)
# %% 학습돌리면 gpu 메모리가 부족해서 에러가 남
# training(model, tokenizer, loss_fn, train_dataset, dev_dataset, optimizer, epoch=3)
#%%
# print(train_dataset[0])
# len(train_dataset[0]['attention_mask'][0]), len(train_dataset[0]['token_type_ids'][0])
#%%
model = BigBirdForQuestionAnswering.from_pretrained('/home/administrator/ingeol/NLP_Project/project2_Machine_Reading_Comprehension/model.2')
model.cuda()
model.eval()
#%% ----------------- 연습장
print(train_dataset[9]['input_ids'])
# output = model(train_dataset[8]['input_ids'], attention_mask = train_dataset[8]['attention_mask'])
# print(output)
#%%
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
    
    #answer_start_index = outputs.start_logits.argmax(dim=1) # - 예측값 시작 위치
    #answer_end_index = outputs.end_logits.argmax(dim=1)
    #print(tokenizer.decode(train_dataset[i]['input_ids'][0][answer_start_index.tolist()[0]:answer_end_index.tolist()[0]+1])) # - 예측 값)
    L = list(sample['context'][start:end])
    ans_return = ''.join(L)
    if len(L)>15:
        L = []
    
    # print(type(''.join(L)),ans_return)
    # if len(sample['context'][start:end]) > 20:
    #     L = list(sample['context'][start:end])
    #     L=[]
    #     sample['context'][start:end] = ''.join(L)
    
    print('Answer:', ''.join(L))
    #print('Answer:', sample['context'][start:end])
# %%
test_dataset = KoMRC.load('/home/administrator/ingeol/NLP_Project/project2_Machine_Reading_Comprehension/test.json')
test_dataset = preprocess_for_test(tokenizer, test_dataset)
# %%
import csv
from tqdm import tqdm
import os

os.makedirs('out', exist_ok=True)
with torch.no_grad(), open('/home/administrator/ingeol/NLP_Project/project2_Machine_Reading_Comprehension/baseline.csv', 'w') as fd:
    writer = csv.writer(fd)
    writer.writerow(['Id', 'Predicted'])

    rows = []
    for sample in tqdm(test_dataset, "Testing"):
        input_ids, token_type_ids = [
            torch.tensor(sample[key], dtype=torch.long, device="cuda")
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
        
        #print(sample['context'][start:end])
        if start >1024 or end>1024:
            start,end = 0,0
        
        if start +15 < end:
            start_1 = start
            start = sample['offset_mapping'][0][start_1][0]
            end = sample['offset_mapping'][0][start_1][0]
        else:
            start = sample['offset_mapping'][0][start][0]
            end = sample['offset_mapping'][0][end][1]

        # #1338
        # start = sample['offset_mapping'][0][start][0]
        # end = sample['offset_mapping'][0][end][1]

        L = list(sample['context'][start:end])
        # if len(L)>15:
        #     L = []
        
        rows.append([sample["guid"], ''.join(L)])   
        #rows.append([sample["guid"], sample['context'][start:end]])
    
    writer.writerows(rows)

# %%
sample = test_dataset[2454]
input_ids, token_type_ids = [
            torch.tensor(sample[key], dtype=torch.long, device="cuda")
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
print(start,end)

# if start +15 < end:
#     start_1 = start
#     start = sample['offset_mapping'][0][start_1][0]
#     end = sample['offset_mapping'][0][start_1][0]
# else:
#     start = sample['offset_mapping'][0][start][0]
#     end = sample['offset_mapping'][0][end][1]

# print(start,end,sample['context'][start:end])
# for i in test_dataset:
#     print(i)
#     break
# %%
239<1074
# %%
