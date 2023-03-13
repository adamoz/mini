import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import gpt_small_config, gpt_medium_config, gpt_big_config
from model import GPT, adjust_optimizer_lr
from data import Data, GPTDataset
torch.manual_seed(1337)

config = gpt_big_config
Model = GPT
Dataset = GPTDataset

data = Data(config)
config = data.adjust_config(config)

train_dataset = Dataset(data, split='train')
valid_dataset = Dataset(data, split='valid')
train_loader = DataLoader(dataset=train_dataset, shuffle=True, pin_memory=True, batch_size=config.batch_size)
valid_loader = DataLoader(dataset=train_dataset, shuffle=True, pin_memory=True, batch_size=config.batch_size)
train_iter = iter(train_loader)
valid_iter = iter(valid_loader)

model = Model(config)
model = model.to(config.device)
optimizer = model.get_optimizer()
print(config)

for step in range(config.max_iters):
    optimizer  = adjust_optimizer_lr(optimizer, step, config)

    if step % config.eval_interval == 0:
        losses = model.estimate_loss(train_iter, valid_iter)
        print(f"Step {step:<5} | train loss: {losses['train']:<8.4f} | valid loss: {losses['valid']:<8.4f}")
   
    for micro_step in range(config.gradient_accumulation_steps):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb, yb = xb.to(config.device), yb.to(config.device)
        _, loss = model(xb, yb)
        loss.backward()
    
    nn.utils.clip_grad_norm_(model.parameters(), config.grand_norm_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

torch.save(model.state_dict(), 'model.pt')
print(data.decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=config.device), n=1000, do_sample=config.do_sample)[0].tolist()))
