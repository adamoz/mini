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
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    betas: typing.Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.1
    vocab_size: int = 10000
    grand_norm_clip: float = 1.0


    def __post_init__(self):
        self.head_size = self.n_embd // self.n_heads


"""
decoder_config = DecoderConfig(n_embd=384, n_heads=6, n_layers=6, dropout=0.2, block_size=64, batch_size=256, 
                               max_iters=5001, eval_iters=200, eval_interval=500, lr_decay_iters=5000, warmup_iters=100, do_sample=True,
                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
"""

decoder_config = DecoderConfig(n_embd=192, n_heads=3, n_layers=3, dropout=0.2, block_size=64, batch_size=64, 
                               max_iters=5001, eval_iters=200, eval_interval=500, lr_decay_iters=5000, warmup_iters=100, do_sample=True,
                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))