import torch
import torch.nn as nn
from torch.nn import functional as F
from config import decoder_config as config
from decoder import DecoderModel
import math
torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O shakespear.txt
with open('shakespear.txt', 'r', encoding='utf-8') as fr:
    text = fr.read()
chars = sorted(list(set(text)))
config.vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[ch] for ch in s]
def decode(l): return "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, valid_data = data[:n], data[n:]


def get_batch(split, config=config):
    data = train_data if split == 'train' else valid_data
    ix = torch.randint(0, len(data) - config.block_size, (config.batch_size, ))
    x = torch.stack([data[i: i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1: i + config.block_size + 1] for i in ix])
    return x.to(config.device), y.to(config.device)


@torch.no_grad()
def estimate_loss(model, config=config):
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(config.eval_iters)
        for i in range(config.eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it, config):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.max_lr * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


model = DecoderModel(config)
model = model.to(config.device)
optimizer = model.get_optimizer(config)
print(config)

for step in range(config.max_iters):
    lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step % config.eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Step {step} | train loss: {losses['train']:.4f} | valid loss: {losses['valid']:.4f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config.grand_norm_clip)
    optimizer.step()

torch.save(model.state_dict(), 'model.pt')
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=config.device), n=1000, do_sample=config.do_sample)[0].tolist()))
