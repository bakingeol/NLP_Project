
#%%
from typing import List, Tuple, Dict, Any
import json
import random

class KoMRC:
    def __init__(self, data, indices: List[Tuple[int, int, int]]):
        self._data = data
        self._indices = indices

    # Json을 불러오는 메소드
    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as fd:
            data = json.load(fd)

        indices = []
        for d_id, document in enumerate(data['data']):
            for p_id, paragraph in enumerate(document['paragraphs']):
                for q_id, _ in enumerate(paragraph['qas']):
                    indices.append((d_id, p_id, q_id))
        
        return cls(data, indices)

    # 데이터 셋을 잘라내는 메소드
    @classmethod
    def split(cls, dataset, eval_ratio: float=.1, seed=42):
        indices = list(dataset._indices)
        random.seed(seed)
        random.shuffle(indices)
        train_indices = indices[int(len(indices) * eval_ratio):]
        eval_indices = indices[:int(len(indices) * eval_ratio)]

        return cls(dataset._data, train_indices), cls(dataset._data, eval_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        d_id, p_id, q_id = self._indices[index]
        paragraph = self._data['data'][d_id]['paragraphs'][p_id]

        context = paragraph['context']
        qa = paragraph['qas'][q_id]

        guid = qa['guid']
        question = qa['question']
        answers = qa['answers']

        return {
            'guid': guid,
            'context': context,
            'question': question,
            'answers': answers
        }

    def __len__(self) -> int:
        return len(self._indices)
#%%
dataset = KoMRC.load('/home/administrator/ingeol/NLP_Project/project2_Machine_Reading_Comprehension/train.json')

tra_dataset, de_dataset = KoMRC.split(dataset)


from transformers import (LongformerModel, 
                          LongformerConfig, 
                          LongformerPreTrainedModel,
                          LongformerForQuestionAnswering,
                          LongformerTokenizer,
                          LongformerTokenizerFast)
import torch
#%%
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
config = LongformerConfig()
model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096")

model.config.return_dict = False
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
loss_fn = torch.nn.CrossEntropyLoss()
#%%
# 데이터셋 전처리 함수
def preprocess(dataset):
    examples = []
    for data in dataset:
        context = data['context']
        question = data['question']
        answer_text = data['answers'][0]['text']
        start_position = data['answers'][0]['answer_start']

        inputs = tokenizer(question, context, padding='max_length', 
                           truncation=True, max_length=4096, 
                           return_overflowing_tokens=True, return_offsets_mapping=True)
        
        if len(inputs['input_ids'][0]) <= 4096:  # 수정된 부분
            input_ids = inputs['input_ids'][0]  # 수정된 부분
            attention_mask = inputs['attention_mask'][0]  # 수정된 부분
            start_position_encoded = tokenizer.encode_plus(answer_text, context, return_offsets_mapping=True, 
                                                           add_special_tokens=False)['offset_mapping'][0][0]
            end_position_encoded = start_position_encoded + len(answer_text)

            # start/end position이 4096을 넘어가는 경우 처리
            if start_position_encoded >= 4096:
                start_position_encoded = 4096
                end_position_encoded = 4096
            elif end_position_encoded >= 4096:
                end_position_encoded = 4096

            start_position += len(tokenizer.tokenize(context[:start_position])[1:-1])
            end_position = start_position + len(tokenizer.tokenize(answer_text)[1:-1])
            examples.append({'input_ids': input_ids, 'attention_mask': attention_mask, 
                             'start_positions': start_position, 'end_positions': end_position})
        else:
            # overflow 데이터 처리
            for i, (input_ids, attention_mask) in enumerate(zip(inputs['input_ids'], inputs['attention_mask'])):
                if i == 0:
                    continue
                start_position_encoded = tokenizer.encode_plus(answer_text, context, 
                                                               return_offsets_mapping=True, 
                                                               add_special_tokens=False)['offset_mapping'][0][0]
                end_position_encoded = start_position_encoded + len(answer_text)

                # start/end position이 4096을 넘어가는 경우 처리
                if start_position_encoded >= 4096:
                    start_position_encoded = 4096
                    end_position_encoded = 4096
                elif end_position_encoded >= 4096:
                    end_position_encoded = 4096

                start_position = inputs['offset_mapping'][i][start_position_encoded][0]
                end_position = inputs['offset_mapping'][i][end_position_encoded-1][1]

                examples.append({'input_ids': input_ids, 'attention_mask': 
                                 attention_mask, 'start_positions': start_position, 'end_positions': end_position})

    return examples
#%%
train_dataset = preprocess(tra_dataset)
eval_dataset = preprocess(de_dataset)
#%%
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#%%
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
torch.autograd.set_detect_anomaly(True)
#%%
# input_ids = torch.LongTensor(train_dataset[198]['input_ids']).to(device)
# attention_mask = torch.LongTensor(train_dataset[198]['attention_mask']).to(device)
# input_ids = input_ids.unsqueeze(0)
# attention_mask = attention_mask.unsqueeze(0)
# #%%
# model.to(device)
# model(input_ids, attention_mask = attention_mask)
#%%
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
try:
    for epoch in range(30):
        print('epoch:',epoch)
        for i in tqdm(range(len(train_dataset))):

            optimizer.zero_grad()

            input_ids = torch.LongTensor(train_dataset[i]['input_ids']).to(device)
            attention_mask = torch.LongTensor(train_dataset[i]['attention_mask']).to(device)
            
            # trian_dataset[i]['']
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            # i = i.unsqueeze(0)
            # a = a.unsqueeze(0)
            
            output = model(input_ids, attention_mask = attention_mask)

            start_logits= output[0]
            end_logits = output[1]
            start_pos = torch.LongTensor([train_dataset[i]['start_positions']]).to(device)
            end_pos = torch.LongTensor([train_dataset[i]['end_positions']]).to(device)
            
            # print(start_logits, type(start_logits), len(start_logits))
            # print('-'*50)
            # print(start_pos, type(start_pos), len(start_pos))

            start_loss = loss_fn(start_logits, start_pos)
            end_loss = loss_fn(end_logits, end_pos)
            
            loss = start_loss + end_loss

            print('loss:',loss)

            loss.backward()
            optimizer.step()
except RuntimeError:
    import pdb; pdb.set_trace()