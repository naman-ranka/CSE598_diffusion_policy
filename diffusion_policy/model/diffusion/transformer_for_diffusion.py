import torch
import torch.nn as nn
from typing import Union, Optional, Tuple
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import logging

logger = logging.getLogger(__name__)

class AdaLN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )

    def forward(self, x, cond):
        shift, scale = self.mod(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * scale[:, None, :] + shift[:, None, :]

class AdaLNTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = AdaLN(d_model)
        self.norm2 = AdaLN(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, cond):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src, cond)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src, cond)
        return src

class AdaLNHead(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, cond):
        shift, scale = self.modulation(cond).chunk(2, dim=1)
        x = self.norm(x)
        x = x * scale[:, None, :] + shift[:, None, :]
        return self.linear(x)

class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 horizon: int,
                 n_obs_steps: int = None,
                 cond_dim: int = 0,
                 n_layer: int = 12,
                 n_head: int = 12,
                 n_emb: int = 768,
                 p_drop_emb: float = 0.1,
                 p_drop_attn: float = 0.1,
                 causal_attn: bool = False,
                 time_as_cond: bool = True,
                 obs_as_cond: bool = False) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb) if obs_as_cond else None
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb)) if T_cond > 0 else None

        self.encoder = nn.ModuleList([
            AdaLNTransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn
            ) for _ in range(n_layer)
        ])

        self.ada_head = AdaLNHead(n_emb, output_dim)
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], cond: Optional[torch.Tensor] = None):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        time_emb = self.time_emb(timestep).unsqueeze(1)

        input_emb = self.input_emb(sample)
        token_embeddings = input_emb

        if self.obs_as_cond:
            cond = self.cond_obs_emb(cond)
            cond_token = cond.mean(dim=1) + time_emb.squeeze(1)
        else:
            cond_token = time_emb.squeeze(1)

        t = token_embeddings.shape[1]
        pos = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + pos)

        for layer in self.encoder:
            x = layer(x, cond_token)

        x = self.ada_head(x, cond_token)
        return x
