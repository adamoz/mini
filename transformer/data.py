from torch.utils.data import Dataset
import numpy as np
import torch
import random
from enum import Enum
import pandas as pd
from collections import Counter

class DataMode(Enum):
    GPT = 1
    BERT = 2
    BERT_FINETUNE = 3


def separate_person_and_speech(data):
    idx, text = data
    try:
        person, speech = text.split(':\n', 1)
    except Exception as err:
        return None, None
    person = person.strip().lower()
    speech = speech.strip()
    return person, speech


def get_person_speech_df(text):
    parsed_text = list(map(lambda data: separate_person_and_speech(data), enumerate(text.split('\n\n'))))
    persons, speeches = zip(*parsed_text)

    df = pd.DataFrame({'person': persons, 'speech': speeches}).dropna()
    df = pd.merge(df, df.person.value_counts().to_frame().reset_index().rename(columns={'index': 'person', 'person': 'speech_freq'}), on='person', how='left')
    return df


def split_train_valid(df, n=0.9, speech_freq=50):
    ddf = df[df.speech_freq > speech_freq]
    size_map = dict(ddf.person.value_counts())
    
    for person in ddf.person.unique().tolist():
        train_size = int(n * size_map[person])
        ddf.loc[ddf.person==person, 'idx'] = np.arange(size_map[person])
        ddf.loc[(ddf.person==person) & (ddf.idx < train_size), 'split'] = 'train'
        ddf.loc[(ddf.person==person) & (ddf.idx >= train_size), 'split'] = 'valid'
    ddf = ddf.drop(columns=['idx'])
    return ddf

class Data(object):
    def __init__(self, config, data_path='shakespear.txt', mode=DataMode.GPT):
        with open(data_path, 'r', encoding='utf-8') as fr:
            text = fr.read()
        chars = sorted(list(set(text)))
        if mode == DataMode.BERT or mode==DataMode.BERT_FINETUNE:
            chars = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + chars
        self.mode = mode
        self.config = config
        self.text = text
        self.vocab_size = len(chars)
        self.block_size = config.block_size
        self.stoi = { ch:i for i,ch in enumerate(chars)}
        self.itos = { i:ch for i,ch in enumerate(chars)}
        print(f'Dataset has {len(self.text)} characters.\nUpdating config.vocab_size={len(chars)}')

        self.split_data(mode=mode)
        config = self.adjust_config(config)

    def split_data(self, mode=None):
        if mode == mode.BERT_FINETUNE:
            assert self.mode in [DataMode.BERT_FINETUNE, DataMode.BERT]
        if mode is None:
            mode = self.mode
        self.mode = mode
         
        if mode != DataMode.BERT_FINETUNE:        
            n = int(self.config.split_ratio * len(self.text))
            self.train_data, self.valid_data = self.text[:n], self.text[n:]
        else:
            df = get_person_speech_df(self.text)
            df = split_train_valid(df, n=self.config.split_ratio, speech_freq=50)
            self.train_data = df[df.split=='train']
            self.valid_data = df[df.split=='valid']
            self.person_id_map = {person: i for i, person in enumerate(df.person.unique())}
            self.id_person_map = {i: person for i, person in enumerate(df.person.unique())}
            self.person_vocab_size = len(self.person_id_map)

    def adjust_config(self, config):
        config.vocab_size = self.vocab_size
        if self.mode == DataMode.BERT_FINETUNE:
            config.person_vocab_size = self.person_vocab_size
        return config

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size
    
    def encode(self, array:list):
        return [self.stoi[s] for s in array]
    
    def encode_person(self, person):
        return self.person_id_map[person]

    def mask_encode(self, array:list, step=5, word_range=[2, 4]):
        encoded_array = self.encode(array)
        output_labels = [0] * len(encoded_array)

        array_len = len(array)
        pos = 0
        while pos < array_len:
            if random.random() < 0.15:
                word_len = min(step, np.random.randint(*word_range))
                output_labels[pos: pos+word_len] = encoded_array[pos: pos+word_len]
                if random.random() < 0.8:
                    encoded_array[pos: pos+word_len] = [self.stoi['[MASK]']] * word_len
                elif random.random() < 0.5:
                    encoded_array[pos: pos+word_len] = [random.randrange(self.vocab_size) for _ in range(word_len)]
            pos += step
        return encoded_array[:array_len], output_labels[:array_len]

    def decode(self, array:list):
        return "".join([self.itos[s] for s in array])

    def decode_person(self, person_id):
        return self.id_person_map[person_id]


class BERTFinetuneDataset(Dataset):
    def __init__(self, data, split='train'):
        self.data = data
        self.split = split

    def set_split(self, split):
        self.split = split

    def __len__(self):
        if self.split == 'train':
            return len(self.data.train_data)
        else:
            return len(self.data.valid_data)
        
    def __getitem__(self, idx):
        d = self.data.train_data if self.split == 'train' else self.data.valid_data
        person = d.iloc[idx].person
        speech = d.iloc[idx].speech
        person_id = self.data.encode_person(person)
        speech = self.data.encode(speech[:self.data.block_size-2])

        x = [self.data.stoi['[CLS]']] + speech + [self.data.stoi['[SEP]']]
        segment_ids = [0] * len(x)
        padding = [self.data.stoi['[PAD]']] * (self.data.block_size - len(x))
        
        return {
            'x': torch.tensor(x + padding, dtype=torch.long),
            'segment_ids': torch.tensor(segment_ids + padding, dtype=torch.long),
            'person_id': torch.tensor(person_id, dtype=torch.long)
        }
    
class BERTDataset(Dataset):
    def __init__(self, data, split='train'):
        self.data = data
        self.split = split
        
    def set_split(self, split):
        self.split = split

    def __len__(self):
        if self.split == 'train':
            return len(self.data.train_data) - self.data.block_size
        else:
            return len(self.data.valid_data) - self.data.block_size

    def __getitem__(self, idx):
        d = self.data.train_data if self.split == 'train' else self.data.valid_data

        new_block_size = np.random.randint(int(0.95*(self.data.block_size-3)), self.data.block_size+1-3)
        block_split = int(new_block_size/2) + np.random.randint(-int(0.05*new_block_size), int(0.05*new_block_size))
        missing_chunk_size = new_block_size - block_split
        
        chunk = d[idx:idx + self.data.block_size]
        x_a, y_a = self.data.mask_encode(chunk[:block_split])
        
        does_continue = 0
        if np.random.rand() < 0.5:
            does_continue = 1
            x_b, y_b = self.data.mask_encode(chunk[block_split:new_block_size])
        else:
            random_start = np.random.randint(0, len(d) - self.data.block_size)
            random_chunk = d[random_start: random_start + missing_chunk_size]
            x_b, y_b = self.data.mask_encode(random_chunk)
        
        x = [self.data.stoi['[CLS]']] + x_a + [self.data.stoi['[SEP]']] + x_b + [self.data.stoi['[SEP]']]
        y = [0] + y_a + [0] + y_b + [0] # 0 is ignored label
        segment_ids = [1] * (block_split + 2) + [2] * (missing_chunk_size + 1)
        padding = [self.data.stoi['[PAD]']] * (self.data.block_size - len(x))
        
        return {
            'x': torch.tensor(x+padding, dtype=torch.long),
            'y': torch.tensor(y+padding, dtype=torch.long),
            'segment_ids': torch.tensor(segment_ids+padding, dtype=torch.long),
            'does_continue': torch.tensor(does_continue, dtype=torch.long)
        }


class GPTDataset(Dataset):
    def __init__(self, data, split='train'):
        self.data = data
        self.split = split
        
    def set_split(self, split):
        self.split = split

    def __len__(self):
        if self.split == 'train':
            return len(self.data.train_data) - self.data.block_size - 1
        else:
            return len(self.data.valid_data) - self.data.block_size - 1

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        if self.split == 'train':
            chunk = self.data.train_data[idx:idx + self.data.block_size + 1]
        else:
            chunk = self.data.valid_data[idx:idx + self.data.block_size + 1]
        dix = self.data.encode(chunk)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y