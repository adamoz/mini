from torch.utils.data import Dataset
import numpy as np
import torch
import random

class Data(object):
    def __init__(self, config, data_path='shakespear.txt', bert=False):
        with open(data_path, 'r', encoding='utf-8') as fr:
            text = fr.read()
        chars = sorted(list(set(text)))
        if bert:
            chars = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + chars
        self.vocab_size = len(chars)
        self.block_size = config.block_size
        self.stoi = { ch:i for i,ch in enumerate(chars)}
        self.itos = { i:ch for i,ch in enumerate(chars)}
        
        n = int(0.9 * len(text))
        self.train_data, self.valid_data = text[:n], text[n:]
        print(f'Dataset has {len(text)} characters.\nUpdating config.vocab_size={len(chars)}')
        config = self.adjust_config(config)

    def adjust_config(self, config):
        config.vocab_size = self.vocab_size
        return config

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size
    
    def encode(self, array:list):
        return [self.stoi[s] for s in array]
    
    def mask_encode(self, array:list, step=4, word_range=[2,5]):
        encoded_array = self.encode(array)
        output_labels = [0] * len(encoded_array)

        array_len = len(array)
        pos = 0
        while pos < array_len:
            if random.random() < 0.15:
                word_len = np.random.randint(*word_range)
                output_labels[pos: pos+word_len] = encoded_array[pos: pos+word_len]
                if random.random() < 0.8:
                    encoded_array[pos: pos+word_len] = [self.stoi['[MASK]']] * word_len
                elif random.random() < 0.5:
                    encoded_array[pos: pos+word_len] = [random.randrange(self.vocab_size) for _ in range(word_len)]
                pos += word_len
            else:
                pos += step
        return encoded_array[:array_len], output_labels[:array_len]

    def decode(self, array:list):
        return "".join([self.itos[s] for s in array])


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

        new_block_size = np.random.randint(int(0.9*(self.data.block_size-3)), self.data.block_size+1-3)
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