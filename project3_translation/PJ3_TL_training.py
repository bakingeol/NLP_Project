#%%
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pandas as pd
from torch.utils.data import DataLoader
import tqdm
import random

#%%
device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
tokenizer = T5Tokenizer.from_pretrained("t5-base", src_lang="en", tgt_lang="ko")
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
# %%
df = pd.read_csv('/home/administrator/ingeol/NLP_Project/project3_translation/문어체_뉴스 편집본.csv')
df = df.iloc[:,:2]
src_data = df.iloc[:,1].to_list()[150000:] # 영어
tar_data = df.iloc[:,0].to_list()[150000:] # 한국어
len(tar_data)
#%% training토큰화
x = []
y = []
task_prefix = "translate English to Korean: "
for i,j in zip(src_data, tar_data):
    x.append(tokenizer(task_prefix+i, return_tensors = 'pt',padding='max_length', max_length =1024).input_ids)
    y.append(tokenizer(task_prefix+j, return_tensors = 'pt',padding='max_length', max_length =1024).input_ids)
#%%

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(x, y, shuffle=True, 
                                                    random_state=1234,test_size=0.2)
#X_train, X_test, y_train, y_test= train_test_split(src_data, tar_data, shuffle=True, 
#                                                   random_state=1234,test_size=0.2)
#%% train  
X_train_batch= DataLoader(X_train, batch_size=1, shuffle=False, pin_memory=True)
y_train_batch= DataLoader(y_train, batch_size=1, shuffle=False, pin_memory=True)
#eval
X_test_batch= DataLoader(X_test, batch_size=1, shuffle=False, pin_memory=True)
y_test_batch= DataLoader(y_test, batch_size=1, shuffle=False, pin_memory=True)
#%%
for i in X_train_batch:
    print(i.shape)
#%% 학습 부분
loss_train = []
loss_test = []
for epoch in range(0,3):
    for batch_idx, (input_ids, labels) in enumerate(zip(X_train_batch, y_train_batch)):
        output = model(input_ids=input_ids.squeeze(1).to(device), labels=labels.squeeze(1).to(device))
        
        optimizer.zero_grad()
        
        loss = output.loss
        loss_train.append(loss)
        loss.backward()
        
        optimizer.step()
        print(f'epoch: {epoch}, loss: {loss_train[-1]}')    
# eval


    for batch_idx, (input_ids, labels) in enumerate(zip(X_test_batch, y_test_batch)):
        
        with torch.no_grad():
            output = model(input_ids=input_ids.squeeze(1).to(device), labels=labels.squeeze(1).to(device))
            loss = output.loss
            loss_test.append(loss)
        print(f'epoch: {epoch}, loss: {loss_test[-1]}')

model.save_pretrained(f'/content/drive/MyDrive/Colab Notebooks/groom/project3/dump/model.{epoch}')
#%% #1/4epoch 마다 loss 보여주는 부분
loss_len= 4 * 3 # 1/4epoch 마다 loss * 총 epoch 횟수
for i in range(0, len(loss_train), len(loss_train)//loss_len):
    for j in range(0,3):
        loss_train_cum = sum(loss_train[i*j:i*(j+1)])/len(loss_train[i*j:i*(j+1)])
for i in range(0, len(loss_test), len(loss_test)//loss_len):
    for j in range(0,3):
        loss_test_cum = sum(loss_test[i*j:i*(j+1)])/len(loss_test[i*j:i*(j+1)])
#%%
import matplotlib.pyplot as plt

t = list(range(1, loss_len))
plt.plot(t, loss_train_cum, label="Train Loss")
plt.plot(t, loss_test_cum, label="Dev Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
#%% 
from bleu import list_bleu
'''
1. 테스트 세로운 csv 하나 만들기
2. 모델 불러오기
3. 불러온 모델 테스트, BLUE score 구하기
'''
#%%
model = T5ForConditionalGeneration.from_pretrained("-----모델주소-----")
model.cuda()
model.eval()
#%%
task_prefix = "translate English to Korean: "
test_dataset = pd.read_csv('/home/administrator/ingeol/NLP_Project/project3_translation/문어체_뉴스 편집본_test.csv')
test_df=test_dataset.iloc[:5000,:2]

# %%
test_src_data = test_df.iloc[:,1].to_list() # 영어
test_tar_data = test_df.iloc[:,0].to_list() # 한국어
# %%
test_src_data[0]
# %%

for sample_src, sample_tar in zip(test_src_data,test_tar_data):
    input_ids= tokenizer(task_prefix+sample_src, return_tensors = 'pt',padding='max_length', max_length =1024).input_ids
    output = model.generate(input_ids.to(device)).to(device)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print(answer,sample_tar)
    #print(list_bleu([answer], [sample_tar]))
    

#%%
from bleu import list_bleu

ref = ['it is a white cat .',
             'wow , this dog is huge .']
ref1 = ['This cat is white .',
             'wow , this is a huge dog .']
hyp = ['it is a white kitten .',
            'wowww , the dog is huge !']
hyp1 = ["it 's a white kitten .",
             'wow , this dog is huge !']
list_bleu([ref], hyp)
# %%
