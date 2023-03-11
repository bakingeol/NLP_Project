#%%
import os
import pdb
import argparse
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from tqdm import tqdm, trange

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AutoConfig,
    AdamW)

import pickle
import argparse
#%%
class SentimentDataset(object):
    def __init__(self, tokenizer, pos, neg):
        self.tokenizer = tokenizer
        self.data = []
        self.label = []

        for pos_sent in pos:
            self.data += [self._cast_to_int(pos_sent.strip().split())]
            self.label += [[1]]
        for neg_sent in neg:
            self.data += [self._cast_to_int(neg_sent.strip().split())]
            self.label += [[0]]

    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample), np.array(self.label[index])
#%%
def collate_fn_style(samples):
    input_ids, labels = zip(*samples)
    max_len = max(len(input_id) for input_id in input_ids)
    #sorted_indices = np.argsort([len(input_id) for input_id in input_ids])[::-1] # len이 긴것부터 정렬해준다.
    
    # 2번
    sorted_indices = range(len(input_ids))
    
    attention_mask = torch.tensor(
        [[1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index])) for index in
         sorted_indices])
    
    #input_ids = pad_sequence([torch.tensor(input_ids[index]) for index in sorted_indices],
    #                         batch_first=True)
    # 1번
    input_ids = pad_sequence([torch.tensor(input_id) for input_id in input_ids],
                             batch_first=True)
    
    token_type_ids = torch.tensor([[0] * len(input_ids[index]) for index in sorted_indices])
    position_ids = torch.tensor([list(range(len(input_ids[index]))) for index in sorted_indices])
    labels = torch.tensor(np.stack(labels, axis=0)[sorted_indices]) # y도 정렬된 형태로 맞춰준다. 
    # 정렬을 하면 학습의 속도가 빨리지기 때문에 진행한다.
    
    
    
    return input_ids, attention_mask, token_type_ids, position_ids, labels  
# %%
def collate_fn_style_test(samples):
    input_ids = samples
    max_len = max(len(input_ids) for input_id in input_ids)
    
    
    sorted_indices = range(len(input_ids))
    attention_mask = torch.tensor(
        [[1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index])) for index in
         sorted_indices])
    
    input_ids = pad_sequence([torch.tensor(input_id) for input_id in input_ids],
                             batch_first=True)
    
    token_type_ids = torch.tensor([[0] * len(input_ids[index]) for index in sorted_indices])
    position_ids = torch.tensor([list(range(len(input_ids[index]))) for index in sorted_indices])
    
    return input_ids, attention_mask, token_type_ids, position_ids
# %%
