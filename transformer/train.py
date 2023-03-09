import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import decoder_config as config
from decoder import DecoderModel, get_lr
from data import Data, DecoderDataset
torch.manual_seed(1337)


data = Data(config)
train_dataset = DecoderDataset(data, split='train')
valid_dataset = DecoderDataset(data, split='valid')
train_loader = DataLoader(dataset=train_dataset, shuffle=True, pin_memory=True, batch_size=config.batch_size)
valid_loader = DataLoader(dataset=train_dataset, shuffle=True, pin_memory=True, batch_size=config.batch_size)
train_iter = iter(train_loader)
valid_iter = iter(valid_loader)

model = DecoderModel(config)
model = model.to(config.device)
optimizer = model.get_optimizer()
print(config)

for step in range(config.max_iters):
    lr = get_lr(step, config) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
