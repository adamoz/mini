import torch
from dataclasses import dataclass
import typing

@dataclass
class DecoderConfig:
    # model oriented
    n_embd: int
    n_heads: int
    n_layers: int
    dropout: float
    # data oriented
    block_size: int
    batch_size: int
    device: torch.device
    do_sample: bool
    # training oriented
    max_iters: int
    eval_iters: int
    eval_interval: int
    lr_decay_iters: int
    warmup_iters: int
    gradient_accumulation_steps: int
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    learning_rate: float = 3e-4
    decay_lr: bool = True
    betas: typing.Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.1
    vocab_size: int = 10000
    grand_norm_clip: float = 1.0

    def __post_init__(self):
        self.head_size = self.n_embd // self.n_heads



gpt_small_config = DecoderConfig(n_embd=96, n_heads=3, n_layers=3, dropout=0.2, block_size=128, batch_size=64, 
                               max_iters=5001, eval_iters=200, eval_interval=500, lr_decay_iters=5001, warmup_iters=100, gradient_accumulation_steps=1, do_sample=True,
                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

gpt_medium_config = DecoderConfig(n_embd=192, n_heads=6, n_layers=3, dropout=0.2, block_size=128, batch_size=64, 
                               max_iters=2001, eval_iters=200, eval_interval=500, lr_decay_iters=2001, warmup_iters=100, gradient_accumulation_steps=2, do_sample=True,
                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

gpt_medium_finetune_config = DecoderConfig(n_embd=192, n_heads=6, n_layers=3, dropout=0.2, block_size=128, batch_size=64, 
                               max_iters=2001, eval_iters=200, eval_interval=500, lr_decay_iters=2001, warmup_iters=100, gradient_accumulation_steps=2, do_sample=True, decay_lr=False, learning_rate=3e-5,
                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

gpt_big_config = DecoderConfig(n_embd=384, n_heads=6, n_layers=6, dropout=0.2, block_size=128, batch_size=64, 
                               max_iters=2001, eval_iters=200, eval_interval=500, lr_decay_iters=2001, warmup_iters=100, gradient_accumulation_steps=2, do_sample=True,
                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


