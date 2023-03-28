import torch
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
import re 
from konlpy.tag import Mecab
    
# 모델과 토크나이저 로드
model_name = 'google/bigbird-roberta-base'
model = BigBirdForQuestionAnswering.from_pretrained(model_name)
tokenizer = BigBirdTokenizer.from_pretrained(model_name)

def normalize(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def process_predicted(predicted):
    mecab = Mecab()
    if isinstance(predicted, str):  # 예외 처리
        pos = mecab.pos(predicted)
        allowed_pos_start = ['XR', 'SL', 'SN', 'SH', 'NNG', 'NNP', 'NNB', 'NR', 'NP', 'VA', 'MM', 'MAG', 'MAJ']
        allowed_pos_end = ['NNP', 'NNB', 'NR', 'NP', 'SH', 'SL', 'SN', 'XSN', 'XSV', 'XSA', 'EF']
        start_idx = None
        for i, (word, x) in enumerate(pos):
            if x in allowed_pos_start:
                start_idx = i
                break
        end_idx = None
        for i in range(len(pos)-1, -1, -1):
            if mecab.pos(pos[i][0])[0] in allowed_pos_end:
                print(pos[i][0])
                end_idx = i
                break
        if start_idx is not None and end_idx is not None:
            processed_predicted = ''.join([pos[i][0] for i in range(start_idx, end_idx+1)])
        else:
            if start_idx is not None:
                processed_predicted = ''.join([pos[i][0] for i in range(start_idx, len(pos))])
            elif end_idx is not None:
                processed_predicted = ''.join([pos[i][0] for i in range(end_idx+1)])
            else:
                processed_predicted = predicted
        return processed_predicted
    else:
        return ''    

def NNG_SN_postpro(sample,text):
    mecab = Mecab()
    answer = []
    answer2 = []
    count = 0
    if len(text) > 10:
        for i, (v,j) in enumerate(mecab.pos(text)):
            if count == 1:
                break
            elif j == 'NNG':
                if v in sample['context']:
                    answer2.append(v)
            elif j =='SN' and i == 0:
                if len(text) > len(v)+5:
                    answer_row = v
                    count += 1    
                else:
                    answer_row = text
                    count += 1
            else:
                answer_row = ''    
        if len(answer2)>1:
            # for k in range(len(answer)):
            answer_row = answer2[0]
        elif len(answer2)==1:
            answer_row = answer2[0]
        elif len(answer2)==0:
            answer_row = ''
        
        return answer_row
    else:
        return text

def postprocess_cum_text(sample, text):
    mecab = Mecab()
    fpos_list = ['NNG', 'XR', 'SL', 'SN', 'MAG', 'VA']
    answer = ''
    cum_count = 0
    for i, (val, pos) in enumerate(mecab.pos(text)):
        if pos not in fpos_list:
            text = text[len(val):]
        elif pos in fpos_list:
            if val in sample:
                
                cum_count += len(val)
                answer = text[:cum_count]
            else:
                return answer
        else:
            return ''
    cum_count = 0
    answer = ''
    for i, (val, pos) in enumerate(mecab.pos(text)):
        if val not in sample:
            answer = text.replace(val,'')
        else:
            cum_count += len(val)
            answer = text[:cum_count]
            

def get_candidate_answers(model_output, top_k=10, score_threshold=0.5):
    candidate_answers = []
    for answer in model_output['answers']:
        if answer['score'] >= score_threshold:
            answer_text = answer['text']
            answer_start = answer['start']
            answer_end = answer['end']
            candidate_answers.append((answer_text, answer_start, answer_end, answer['score']))
    candidate_answers = sorted(candidate_answers, key=lambda x: x[3], reverse=True)[:top_k]
    return candidate_answers

def select_answer(model, question, context, candidate_answers):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    scores = []
    for answer in candidate_answers:
        start, end = answer[1], answer[2]
        input_start = input_ids[0][start:end]
        mask_start = attention_mask[0][start:end]
        input_start = torch.cat([input_start, torch.tensor([tokenizer.sep_token_id])])
        mask_start = torch.cat([mask_start, torch.tensor([1])])
        input_end = input_ids[0][end:]
        mask_end = attention_mask[0][end:]
        input_end = torch.cat([torch.tensor([tokenizer.sep_token_id]), input_end])
        mask_end = torch.cat([torch.tensor([1]), mask_end])
        input_ids_start = torch.cat([input_start, input_end])
        attention_mask_start = torch.cat([mask_start, mask_end])
        with torch.no_grad():
            start_scores, end_scores = model(input_ids=input_ids_start.unsqueeze(0), attention_mask=attention_mask_start.unsqueeze(0))
        start_scores = start_scores.squeeze(0).cpu().numpy()
        end_scores = end_scores.squeeze(0).cpu().numpy()
        score = (start_scores[start] + end_scores[end]) / 2.0
        scores.append(score)
    index = scores.index(max(scores))
    return candidate_answers[index][0]

def bigbird_qa(question, context):
    question = normalize(question)
    context = normalize(context)
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        model_output = model(input_ids=input_ids, attention_mask=attention_mask)
    candidate_answers = get_candidate_answers(model_output)
    answer = select_answer(model, question, context, candidate_answers)
    return answer
