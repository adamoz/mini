import torch
import torch.nn as nn
from torch.nn import functional as F

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
torch.manual_seed(1337)

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


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x: Batch Timestamp n_embd
        B, T, C = x.shape
        k = self.key(x)  # B T head_size
        q = self.query(x)  # B T head_size
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (C ** -0.5) || (self.head_size ** -0.5) # B T T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # random prevention of nodes communication
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_heads * head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: Batch Timestamp n_embd
        B, T, C = x.shape
        assert C == self.n_heads * self.head_size
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.sa_head = MultiHeadAttention(n_heads, n_embd // n_heads)
        self.ffwd = FeedForward(n_embd)
        # B T behaves as a batch in normalization layer
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class AttentionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        assert torch.all(idx >= 0), "idx tensor contains negative indices"
        assert torch.all(idx < self.token_embedding_table.num_embeddings), "idx tensor contains out-of-range indices"

        # take max last block_size tokens
        idx_cond = idx[:, -block_size:]
        B, T = idx_cond.shape
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Timestamp n_embd
        tok_emb = self.token_embedding_table(idx_cond)  # Batch Timestamp n_embd
        x = tok_emb + pos_emb  # Batch Timestamp n_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets[:, -block_size:].view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            assert torch.all(probs >= 0), "probs tensor contains negative values"
            assert torch.allclose(torch.sum(probs, dim=1), torch.ones(probs.shape[0], device=device)), "probs tensor does not sum to 1"
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = AttentionLanguageModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Step {step} | train loss: {losses['train']:.4f} | "
              f"valid loss: {losses['valid']:.4f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'model.pt')
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device),
      max_new_tokens=500)[0].tolist()))
