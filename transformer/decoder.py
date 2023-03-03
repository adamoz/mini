import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.head_size
        self.key = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.query = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

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
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.head_size
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_heads * config.head_size, config.n_heads * config.head_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: Batch Timestamp n_embd
        B, T, C = x.shape
        assert C == self.n_heads * self.head_size
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa_head = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        # B T behaves as a batch in normalization layer
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)


    def forward(self, idx, targets=None):
        assert torch.all(idx >= 0), "idx tensor contains negative indices"
        assert torch.all(idx < self.token_embedding_table.num_embeddings), "idx tensor contains out-of-range indices"
        device = idx.device

        # take max last block_size tokens
        idx_cond = idx[:, -self.block_size:]
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
            targets = targets[:, -self.block_size:].view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        device = idx.device
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            assert torch.all(probs >= 0), "probs tensor contains negative values"
            assert torch.allclose(torch.sum(probs, dim=1), torch.ones(probs.shape[0], device=device)), "probs tensor does not sum to 1"
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx