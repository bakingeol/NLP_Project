#%%
from MRC import KoMRC, preprocess
from transformers import (BigBirdConfig, 
                          BigBirdForQuestionAnswering,
                          BigBirdTokenizer,
                          BigBirdModel,
                          BigBirdTokenizerFast)
import torch
# %%
dataset = KoMRC.load('/home/administrator/ingeol/NLP_Project/project2_Machine_Reading_Comprehension/train.json')
train_dataset, dev_dataset = KoMRC.split(dataset)
# %%
'''
토큰, 모델, 컨피그, optim, loss

guid, context, question, answers, 
'''
# %%
config = BigBirdConfig()
model = BigBirdModel(config)
# %%
model
# %%
tokenizer = BigBirdTokenizerFast.from_pretrained("google/bigbird-roberta-base")
model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-roberta-base")
#%%
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
loss_fn = torch.nn.CrossEntropyLoss()
# %%
for data in train_dataset:
    # context = data['context']
    # question = data['question']
    # text = data['answers'][0]['text']
    # answer_start = data['answers'][0]['answer_start']
    # print(answer_start)
    print(data)
    break
# %%
tra_dataset = preprocess(tokenizer,train_dataset)
# %%
dev_dataset = preprocess(tokenizer,dev_dataset)
# %%
from tqdm import tqdm
# %%
device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
# %%
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# try:
for epoch in range(3):
    print('epoch:',epoch)
    for i in tqdm(range(len(train_dataset))):

        optimizer.zero_grad()

        input_ids = torch.IntTensor(tra_dataset[i]['input_ids']).to(device)
        attention_mask = torch.IntTensor(tra_dataset[i]['attention_mask']).to(device)
        
        # trian_dataset[i]['']
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        # i = i.unsqueeze(0)
        # a = a.unsqueeze(0)
        
        output = model(input_ids, attention_mask = attention_mask)

        start_logits= output[0]
        end_logits = output[1]
        start_pos = torch.IntTensor([tra_dataset[i]['start_positions']]).to(device)
        end_pos = torch.IntTensor([tra_dataset[i]['end_positions']]).to(device)

        # print(start_logits, type(start_logits), len(start_logits))
        # print('-'*50)
        # print(start_pos, type(start_pos), len(start_pos))

        start_loss = loss_fn(start_logits, start_pos)
        end_loss = loss_fn(end_logits, end_pos)
        
        loss = start_loss + end_loss

        print('loss:',loss)

        loss.backward()
        optimizer.step()
# except RuntimeError:
#     import pdb; pdb.set_trace()
# %%
import torch
from transformers import AutoTokenizer, BigBirdForQuestionAnswering
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-roberta-base")
squad_ds = load_dataset("squad_v2", split="train")
# select random article and question
LONG_ARTICLE = squad_ds[81514]["context"]
QUESTION = squad_ds[81514]["question"]
QUESTION
'During daytime how high can the temperatures reach?'
#%%
inputs = tokenizer(QUESTION, LONG_ARTICLE, return_tensors="pt")
# long article and question input
list(inputs["input_ids"].shape)


with torch.no_grad():
    outputs = model(**inputs)
#%%
###########################################################################################################33
#%%
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()
predict_answer_token_ids = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
predict_answer_token = tokenizer.decode(predict_answer_token_ids)
#%%
answer_start_index, answer_end_index, predict_answer_token_ids, predict_answer_token
# %%
target_start_index, target_end_index = torch.tensor([130]), torch.tensor([132])
outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
loss = outputs.loss
# %%
type(predict_answer_token)
# %%
answer_start_index = outputs.start_logits.argmax()
# %%
answer_start_index, answer_end_index
# %%
for i in train_dataset:
    print(i['answers'][0]['text'])
    break

# %%
config = BigBirdConfig()
x = BigBirdTokenizer(config)
# %%
x
# %%
