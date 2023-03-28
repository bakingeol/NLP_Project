#%%
import torch
from tqdm import tqdm
import Levenshtein
device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
def training(model, tokenizer,loss_fn, train_dataset, dev_dataset, optimizer, epoch):

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    wandb_step = 0

    #print('epoch:',epoch)
    accumulation_steps = 16
    leven_batch = []
    leven_loss = [] # 16개씩 평균낸 값
    loss_append = []
    loss_dev_append = []
    loss_batch = []
    total_batch = len(train_dataset)
    for epoch in range(0,epoch):
        avg_cost = 0
        avg_dev_cost =0
        for i in tqdm(range(len(train_dataset))):

            optimizer.zero_grad()

            input_ids = torch.LongTensor(train_dataset[i]['input_ids']).to(device)
            attention_mask = torch.LongTensor(train_dataset[i]['attention_mask']).to(device)
            
            output = model(input_ids, attention_mask = attention_mask)
            start_logits= output[0] # - 예측값
            end_logits = output[1] # - 예측값
            
            if len(start_logits)>1: # 문장이 짤리는 경우 
                start_logits = start_logits[0].unsqueeze(0)
                end_logits = end_logits[0].unsqueeze(0)
                print('문장이 길어서 짤린 부분')

            start_pos = torch.LongTensor([train_dataset[i]['start_positions']]).to(device)
            end_pos = torch.LongTensor([train_dataset[i]['end_positions']]).to(device)
        
            start_loss = loss_fn(start_logits, start_pos) # 시작점에서 예측갑과 실측값의 crossenthropy 
            end_loss = loss_fn(end_logits, end_pos)       # 끝점에서 예측갑과 실측값의 crossenthropy 
            loss = start_loss + end_loss

            # gradient accumulation, batch_size = 16
            loss = loss / accumulation_steps
            loss.backward()
            if (i+1) % accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                print("batch_loss :",loss)
                avg_cost += loss / total_batch
            
            answer_start_index = output.start_logits.argmax(dim=1) # - 예측값 시작 위치
            answer_end_index = output.end_logits.argmax(dim=1) # - 예측값 끝점 위치

            #leven거리 구하기
            answer = train_dataset[i]['answer'] # - 정답값
            predict_answer = tokenizer.decode(train_dataset[i]['input_ids'][0][answer_start_index.tolist()[0]:answer_end_index.tolist()[0]+1]) # - 예측 값
            
            if abs(len(answer)-len(predict_answer)) >5 :
                predict_answer = ''

            LD = Levenshtein.distance(predict_answer, answer)
            leven_batch.append(LD)
            if len(leven_batch) == 16:
                leven_loss.append(sum(leven_batch)/len(leven_batch))
                leven_batch = []
                print('Levenshtein.distance : ',leven_loss[-1])
            #if len(loss) ==16:

        loss_append.append(avg_cost)
        
        print('*'*100,'\n','avg_cost :',avg_cost,'\n','*'*100)
        

        # Evaluation
        dev_batch= len(dev_dataset)
        for i in tqdm(range(len(dev_dataset))):

            with torch.no_grad():
                output = model(input_ids, attention_mask = attention_mask)
                start_logits= output[0] # - 예측값
                end_logits = output[1] # - 예측값        
                if len(start_logits)>1: # 문장이 짤리는 경우 
                    start_logits = start_logits[0].unsqueeze(0)
                    end_logits = end_logits[0].unsqueeze(0)
                    print('문장이 길어서 짤린 부분')    
            start_pos = torch.LongTensor([train_dataset[i]['start_positions']]).to(device)
            end_pos = torch.LongTensor([train_dataset[i]['end_positions']]).to(device)
        
            start_loss = loss_fn(start_logits, start_pos) # 시작점에서 예측갑과 실측값의 crossenthropy 
            end_loss = loss_fn(end_logits, end_pos)       # 끝점에서 예측갑과 실측값의 crossenthropy 
            loss = start_loss + end_loss

            if (i+1) % accumulation_steps == 0:
                print("batch_loss :",loss)
                avg_dev_cost += loss / dev_batch 

            print('*'*100,'\n','avg_cost :',avg_dev_cost,'\n','*'*100) # tab down
        loss_dev_append.append(avg_dev_cost)

        model.save_pretrained(f'/content/drive/MyDrive/Colab Notebooks/groom/project2/dump/model.{epoch}')
        # # except RuntimeError:
        # #     import pdb; pdb.set_trace()