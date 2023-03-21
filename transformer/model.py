import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from utils import get_model_optimizer, init_weights

def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


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
        self.proj = nn.Linear(config.n_heads * config.head_size, config.n_heads * config.head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: Batch Timestamp n_embd
        B, T, C = x.shape
        assert C == self.n_heads * self.head_size
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class MultiHeadAttentionV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        # key, query, value projections for all heads, but in a batch
        # n_embed is split into n_heads
        # 3 n_embd covers query value and key (for all heads)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask[:,:,:T, :T] == 0, float('-inf'))
        else:
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa_head = MultiHeadAttentionV2(config)
        self.ffwd = FeedForward(config)
        # B T behaves as a batch in normalization layer
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa_head(self.ln1(x), mask=mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

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
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, do_sample=True):
        device = idx.device
        for _ in range(n):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            assert torch.all(probs >= 0), "probs tensor contains negative values"
            assert torch.allclose(torch.sum(probs, dim=1), torch.ones(probs.shape[0], device=device)), "probs tensor does not sum to 1"
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


    @torch.no_grad()
    def estimate_loss(self, train_loader, valid_loader):
        out = {}
        iterator = {'train': iter(train_loader), 'valid': iter(valid_loader)}
        self.eval()
        for split in ['train', 'valid']:
            losses = torch.zeros(self.config.eval_iters)
            for i in range(self.config.eval_iters):
                try:
                    xb, yb = next(iterator[split])
                except StopIteration:
                    iterator[split] = iter(train_loader if split == 'train' else valid_loader)
                    xb, yb = next(iterator[split])
                xb, yb = xb.to(self.config.device), yb.to(self.config.device)
                _, loss = self(xb, yb)
                losses[i] = loss
            out[split] = losses.mean()
        self.train()
        return out

    def get_optimizer(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        return get_model_optimizer(self, self.config)


class BERTCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd) # TODO cosine
        self.segment_embedding_table = nn.Embedding(3, config.n_embd, padding_idx=0)        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        
        self.apply(init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    def forward(self, idx, segments):
        assert torch.all(idx >= 0), "idx tensor contains negative indices"
        assert torch.all(idx < self.token_embedding_table.num_embeddings), "idx tensor contains out-of-range indices"
        device = idx.device

        # take max last block_size tokens
        idx_cond = idx[:, -self.block_size:]
        B, T = idx_cond.shape
        mask = idx_cond.type(torch.float32).unsqueeze(1).repeat(1, idx_cond.shape[1], 1).unsqueeze(1)
        
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Timestamp n_embd
        seg_emb = self.segment_embedding_table(segments[:, -self.block_size:])
        tok_emb = self.token_embedding_table(idx_cond)  # Batch Timestamp n_embd
        x = tok_emb + pos_emb + seg_emb  # Batch Timestamp n_embd
        
        for block in self.blocks:
            x = block(x, mask=mask)
        return x
              
            
class BERT(nn.Module):
    def __init__(self, bert_core, config):
        super().__init__()
        self.bert_core = bert_core
        self.config = config
        
        self.ln = nn.LayerNorm(config.n_embd)        
        self.ms = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.ns = nn.Linear(config.n_embd, 2, bias=False)
        
    def forward(self, idx, segments, mask_sentence_target=None, next_sentece_target=None):
        x = self.bert_core(idx, segments)
        x = self.ln(x)
        mask_sentence, next_sentence = self.ms(x), self.ns(x[:, 0])
        if mask_sentence_target is None and next_sentece_target is None:
            loss = None
        else:
            loss = 0
            if mask_sentence_target is not None:
                B, T, C = mask_sentence.shape
                mask_sentence = mask_sentence.view(B * T, C)
                mask_sentence_target = mask_sentence_target[:, -self.config.block_size:].view(B * T)
                loss += F.cross_entropy(mask_sentence, mask_sentence_target, ignore_index=0)
                
            if next_sentece_target is not None:
                loss += F.cross_entropy(next_sentence, next_sentece_target)
        return mask_sentence, next_sentence, loss
    
    @torch.no_grad()
    def estimate_loss(self, train_loader, valid_loader):
        out = {}
        iterator = {'train': iter(train_loader), 'valid': iter(valid_loader)}
        self.eval()
        for split in ['train', 'valid']:
            losses = torch.zeros(self.config.eval_iters)
            conts = torch.zeros(self.config.eval_iters)
            for i in range(self.config.eval_iters):
                try:
                    b = next(iterator[split])
                except StopIteration:
                    iterator[split] = iter(train_loader if split == 'train' else valid_loader)
                    b = next(iterator[split])
                x, segment_ids, y, does_continue = b['x'], b['segment_ids'], b['y'], b['does_continue']
                x, segment_ids, y, does_continue = x.to(self.config.device), segment_ids.to(self.config.device), y.to(self.config.device), does_continue.to(self.config.device)
                _, next_sentence_pred, loss = self(x, segments=segment_ids, mask_sentence_target=y, next_sentece_target=does_continue)
                losses[i] = loss
                conts[i] = torch.argmax(next_sentence_pred, dim=-1).eq(does_continue).type(torch.float).mean() * 100
            out[split] = (losses.mean(), conts.mean())
        self.train()
        return out

    def get_optimizer(self):
        return get_model_optimizer(self, self.config)

class BERTFinetune(nn.Module):
    def __init__(self, bert_core, config):
        super().__init__()
        self.bert_core = bert_core
        self.config = config
        self.ln = nn.LayerNorm(config.n_embd)
        self.fc = nn.Linear(config.n_embd, config.person_vocab_size)

    def forward(self, idx, segments, target=None):
        x = self.bert_core(idx, segments)
        x = self.ln(x)
        x = self.fc(x[:, 0])
        if target is None:
            loss = None
        else:
            loss = F.cross_entropy(x, target)
        return x, loss


    @torch.no_grad()
    def estimate_loss(self, train_loader, valid_loader):
        out = {}
        iterator = {'train': iter(train_loader), 'valid': iter(valid_loader)}
        self.eval()
        for split in ['train', 'valid']:
            losses = torch.zeros(self.config.eval_iters)
            accs = torch.zeros(self.config.eval_iters)
            class_correct = [0 for i in range(self.config.person_vocab_size)]
            class_total = [0 for i in range(self.config.person_vocab_size)]
            for i in range(self.config.eval_iters):
                try:
                    b = next(iterator[split])
                except StopIteration:
                    iterator[split] = iter(train_loader if split == 'train' else valid_loader)
                    b = next(iterator[split])
                x, segment_ids, person_ids = b['x'], b['segment_ids'], b['person_id']
                x, segment_ids, person_ids = x.to(self.config.device), segment_ids.to(self.config.device), person_ids.to(self.config.device)
                pred, loss = self(x, segments=segment_ids, target=person_ids)
                losses[i] = loss
                a = torch.argmax(pred, dim=-1).eq(person_ids).type(torch.float)
                for i in range(a.size(0)):
                    p = person_ids[i].item()
                    class_total[p] += 1
                    class_correct[p] += a[i].item()
                accs[i] = a.mean() * 100
            out[split] = (losses.mean(), accs.mean(), class_correct, class_total)
        self.train()
        return out

    def get_optimizer(self):
        return get_model_optimizer(self, self.config)