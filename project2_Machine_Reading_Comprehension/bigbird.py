

#%%
from bigbird_train import training
from MRC import KoMRC, preprocess
from transformers import (BigBirdConfig, 
                          BigBirdForQuestionAnswering,
                          BigBirdTokenizer,
                          BigBirdModel,
                          BertTokenizer,
                          BigBirdTokenizerFast,
                          AutoTokenizer)
import torch
# %%
dataset = KoMRC.load('/Users/baeg-ingeol/Desktop/practice/NLP_Project/project2_Machine_Reading_Comprehension/train.json')
tra_dataset, de_dataset = KoMRC.split(dataset)
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
model = BigBirdForQuestionAnswering.from_pretrained('/Users/baeg-ingeol/Desktop/practice/NLP_Project/project2_Machine_Reading_Comprehension/model.2')
# model.cuda()
model.eval()
#%%
for i in train_dataset:
    print(i['context'])
    break
#%%
for idx, sample in zip(range(1, 4), train_dataset):
    print(f'------{idx}------')
    print('Context:', sample['context'])
    print('Question:', sample['question'])
    
    input_ids, token_type_ids = [
        torch.tensor(sample[key], dtype=torch.long, device="gpu")
        for key in ("input_ids", "token_type_ids")
    ]
    
    with torch.no_grad():
        start_logits, end_logits = model(input_ids=input_ids[None, :], token_type_ids=token_type_ids[None, :])
    start_logits.squeeze_(0), end_logits.squeeze_(0)
    
    start_prob = start_logits[token_type_ids.bool()][1:-1].softmax(-1)
    end_prob = end_logits[token_type_ids.bool()][1:-1].softmax(-1)
    probability = torch.triu(start_prob[:, None] @ end_prob[None, :])
    index = torch.argmax(probability).item()
    
    start = index // len(end_prob)
    end = index % len(end_prob)
    
    start = sample['position'][start][0]
    end = sample['position'][end][1]

    print('Answer:', sample['context'][start:end])
# %%
