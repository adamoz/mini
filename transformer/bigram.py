import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
block_size = 128
batch_size = 64
n_embd = 96
n_heads = 6
n_layers = 6
dropout = 0.2
max_iters = 5001
eval_iters = 200
eval_interval = 500
learning_rate = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O shakespear.txt
with open('shakespear.txt', 'r', encoding='utf-8') as fr:
    text = fr.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[ch] for ch in s]
def decode(l): return "".join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, valid_data = data[:n], data[n:]


def get_batch(split):
    data = train_data if split == 'train' else valid_data
    ix = torch.randint(0, len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets: Batch Timestamp
        logits = self.token_embedding_table(idx)  # Batch Timestamp Channels
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # Batch vocab_size
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size=vocab_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Step {step} | train loss: {losses['train']:.4f} | valid loss: {losses['valid']:.4f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#torch.save(model.state_dict(), 'model.pt')
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))
