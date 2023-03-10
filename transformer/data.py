from torch.utils.data import Dataset
import torch

class Data(object):
    def __init__(self, config, data_path='shakespear.txt'):
        with open(data_path, 'r', encoding='utf-8') as fr:
            text = fr.read()
        chars = sorted(list(set(text)))
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

    def decode(self, array:list):
        return "".join([self.itos[s] for s in array])


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