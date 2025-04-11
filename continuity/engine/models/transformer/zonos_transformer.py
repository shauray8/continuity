# Based on gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/095b2229ee3a40e379c11f05b94bd6923db63b4b/model.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import Literal


from ..attention import ZonosAttentionBlock

@dataclass
class BackboneConfig:
    d_model: int = 1024
    d_intermediate: int = 0
    attn_mlp_d_intermediate: int = 0
    n_layer: int = 16
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = False
    residual_in_fp32: bool = False
    norm_epsilon: float = 1e-5

@dataclass
class InferenceParams:
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: torch.Tensor | None = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()

def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache

class FeedForward(nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 2 * config.attn_mlp_d_intermediate, bias=False)
        self.fc2 = nn.Linear(config.attn_mlp_d_intermediate, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))

class TransformerBlock(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config

        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mixer = ZonosAttentionBlock(config, layer_idx)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mlp = FeedForward(config)

        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // config.attn_cfg["num_heads"]

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
        return torch.empty(batch_size, max_seqlen, 2, self.num_heads_kv, self.head_dim, dtype=dtype), None

    def forward(self, x: torch.Tensor, inference_params: InferenceParams, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm(x), inference_params, freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x

class ZonosTransformer(nn.Module):
    supported_architectures = ["transformer"]
    freqs_cis: torch.Tensor

    def __init__(self, config: BackboneConfig):
        assert not config.ssm_cfg, "This backbone implementation only supports the Transformer model."
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(TransformerBlock(config, i) for i in range(config.n_layer))
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
        # TODO: This function should be pure
        head_dim = self.config.d_model // self.config.attn_cfg["num_heads"]
        self.freqs_cis = precompute_freqs_cis(16384, head_dim)
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states: torch.Tensor, inference_params: InferenceParams) -> torch.Tensor:
        input_pos = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
        input_pos = input_pos + inference_params.lengths_per_sample.unsqueeze(-1)

        freqs_cis = self.freqs_cis[input_pos].expand(hidden_states.shape[0], -1, -1, -1)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, inference_params, freqs_cis)
        return self.norm_f(hidden_states)


