import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import bert_medium_config
from model import BERT, BERTCore
from utils import adjust_optimizer_lr
from data import Data, BERTDataset, DataMode
torch.manual_seed(1337)

config = bert_medium_config
data = Data(config, mode=DataMode.BERT)
config = data.adjust_config(config)

train_dataset = BERTDataset(data, split='train')
valid_dataset = BERTDataset(data, split='valid')
train_loader = DataLoader(dataset=train_dataset, shuffle=True, pin_memory=True, batch_size=config.batch_size)
valid_loader = DataLoader(dataset=train_dataset, shuffle=True, pin_memory=True, batch_size=config.batch_size)
train_iter = iter(train_loader)
valid_iter = iter(valid_loader)

bert_core = BERTCore(config)
model = BERT(bert_core, config)
model = model.to(config.device)
optimizer = model.get_optimizer()
print(config)

best_mse = np.inf
for step in range(config.max_iters):
    optimizer  = adjust_optimizer_lr(optimizer, step, config)

    if step % config.eval_interval == 0:
        losses = model.estimate_loss(train_loader, valid_loader)
        print(f"Step {step:<5} | train loss: {losses['train'][0]:<8.4f} | valid loss: {losses['valid'][0]:<8.4f} | train acc: {losses['train'][1]:<8.2f} | valid acc: {losses['valid'][1]:<8.2f}")
        if losses['valid'][0] < best_mse:
            best_mse = losses['valid'][0]
            torch.save(model.bert_core.state_dict(), f'checkpoints/{config.name}_core_{str(round(losses["valid"][0].item(), 3)).replace(".", "_")}_{step}.pt')
            torch.save(model.state_dict(), f'checkpoints/{config.name}_{str(round(losses["valid"][0].item(), 3)).replace(".", "_")}_{step}.pt')

    for micro_step in range(config.gradient_accumulation_steps):
        try:
            b = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            b = next(train_iter)

        x, segment_ids, y, does_continue = b['x'], b['segment_ids'], b['y'], b['does_continue']
        x, segment_ids, y, does_continue = x.to(config.device), segment_ids.to(config.device), y.to(config.device), does_continue.to(config.device)

        _, _, loss = model(x, segments=segment_ids, mask_sentence_target=y, next_sentece_target=does_continue)
        loss.backward()
    
    nn.utils.clip_grad_norm_(model.parameters(), config.grand_norm_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)