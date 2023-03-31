#%%
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import torch

# GPU 사용 여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 및 토크나이저 불러오기
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
#%%

def encoding(text, tokenizer):
    
    encoded = tokenizer.encode_plus(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']


# fine-tuning할 데이터셋 로드 및 전처리
train_dataset = ... # TODO: fine-tuning할 데이터셋 로드 및 전처리

# optimizer와 learning rate 설정
optimizer = AdamW(model.parameters(), lr=3e-5)

#%%
# fine-tuning 시작
model.train()
for epoch in range(10):
    for batch in train_dataset:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 모델 출력 계산
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # 손실 계산
        loss = outputs[0]

        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 매 epoch마다 모델 평가
    model.eval()
    val_loss = 0
    
    for batch in val_dataset:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs[0]

    print('Epoch: {}, Training Loss: {:.3f}, Validation Loss: {:.3f}'.format(epoch+1, loss.item(), val_loss.item()))

# fine-tuning 완료 후 모델 저장
torch.save(model.state_dict(), 't5-finetuned-koen.pt')
