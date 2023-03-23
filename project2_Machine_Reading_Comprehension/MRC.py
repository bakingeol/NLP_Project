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
    
def preprocess(tokenizer, dataset):
    example=[]
    loc_start, loc_end = 0,0
    for data in dataset:

        context = data['context'] # str
        question = data['question'] # str
        answer_text = data['answers'][0]['text'] # str
        start_position = data['answers'][0]['answer_start'] # int

        end_position = start_position + len(answer_text) # int # 돌려보고 만약 answer_text의 길이가 초과화는경우 if 문 코드 추가

        # start_position_encoded = tokenizer.encode_plus(answer_text, context, return_offsets_mapping=True, 
        #                                            add_special_tokens=False)['offset_mapping'][0][0]
        # end_position_encoded = start_position_encoded + len(answer_text)
        inputs = tokenizer(question, context, padding='max_length', 
                            truncation=True, max_length=1024, 
                            return_overflowing_tokens=True, return_offsets_mapping=True)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        
        for i,v in enumerate(inputs['offset_mapping'][0]):
            if v[0] == start_position:
                loc_start = i
            if v[1] == end_position:
                loc_end = i
        token_type_ids = [0] * (len(question) + 1) + [1] * (len(context) + 2)
        
        example.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids' : token_type_ids,
                        'start_positions': loc_start, 'end_positions': loc_end, 'answer': answer_text, 'context': context, 'question': question})

    return example   # 2차원 2차원, int, int, str


# %%
