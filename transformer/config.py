import torch
from dataclasses import dataclass

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
    # training oriented
    max_iters: int
    eval_iters: int
    eval_interval: int
    learning_rate: float

    def __post_init__(self):
        self.head_size = self.n_embd // self.n_heads


decoder_config = DecoderConfig(n_embd=384, n_heads=6, n_layers=6, dropout=0.2, block_size=256, batch_size=64, 
                               max_iters=5001, eval_iters=200, eval_interval=500, learning_rate=3e-4, 
                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))





   
