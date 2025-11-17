"""
mamba2-minimal
==============

A minimal, single-file implementation of the Mamba-2 model in PyTorch.

> **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**
> Authors: Tri Dao, Albert Gu
> Paper: https://arxiv.org/abs/2405.21060
"""

import json
from dataclasses import dataclass
from typing import Iterable, NamedTuple, TypeAlias, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import LongTensor, Tensor, nn

Device: TypeAlias = str | torch.device | None


@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 24  # number of Mamba-2 layers in the language model
    d_state: int = 128  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16
    ngroups: int = 1  # number of groups for group normalization
    A_init_range: tuple[float, float] = (1, 16)  # A parameter initialization range
    dt_min: float = 0.001  # minimum delta time
    dt_max: float = 0.1  # maximum delta time
    dt_init_floor: float = 1e-4  # floor for dt initialization
    dt_limit: tuple[float, float] = (0.0, float("inf"))  # dt limit range
    learnable_init_states: bool = False  # whether to learn initial states
    activation: str = "swish"  # activation function ("silu" or "swish")
    conv_init: float | None = None  # convolution initialization range
    conv_bias: bool = True  # whether to use bias in convolution
    use_mem_eff_path: bool = True  # use memory efficient (fused) path

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * ngroups * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.ngroups * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )


def inverse_softplus(x: Tensor) -> Tensor:
    """Inverse of softplus function.
    
    softplus(y) = log(1 + exp(y)), so y = log(exp(x) - 1)
    Numerically stable form: y = x + log(-expm1(-x))
    """
    return x + torch.log(-torch.expm1(-x))


def causal_conv1d_pytorch(
    x: Tensor, weight: Tensor, bias: Tensor | None = None, activation: str = "silu"
) -> Tensor:
    """Pure PyTorch implementation of causal_conv1d with activation.
    
    Arguments:
        x: (batch, seq_len, channels) input
        weight: (channels, kernel_width) grouped conv weight
        bias: (channels,) optional bias
        activation: "silu" or "swish"
    
    Returns:
        (batch, seq_len, channels) output
    """
    # Transpose to (batch, channels, seq_len) for conv1d
    x = x.transpose(1, 2)
    k = weight.shape[-1]
    # Convert weight to conv1d format: (out_channels, 1, kernel_width)
    w = weight.unsqueeze(1)
    # Apply grouped conv with causal padding
    out = F.conv1d(x, w, bias=bias, padding=k - 1, groups=x.shape[1])
    # Truncate to original length (causal: only use past)
    out = out[:, :, :x.shape[2]]
    
    if activation in ["silu", "swish"]:
        out = out * torch.sigmoid(out)
    else:
        raise NotImplementedError(f"Activation {activation} not implemented")
    
    # Transpose back to (batch, seq_len, channels)
    return out.transpose(1, 2)


def mamba_split_conv1d_scan_combined(
    zxbcdt: Tensor,
    conv_weight: Tensor,
    conv_bias: Tensor | None,
    dt_bias: Tensor,
    A: Tensor,
    D: Tensor,
    chunk_size: int,
    d_inner: int,
    ngroups: int,
    d_state: int,
    headdim: int,
    nheads: int,
    activation: str = "swish",
    rmsnorm_weight: Tensor | None = None,
    rmsnorm_eps: float = 1e-5,
    norm_before_gate: bool = False,
    outproj_weight: Tensor | None = None,
    outproj_bias: Tensor | None = None,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    seq_idx: Tensor | None = None,
    initial_states: Tensor | None = None,
    device: Device = None,
) -> tuple[Tensor, Tensor]:
    """Pure PyTorch implementation of mamba_split_conv1d_scan_combined.
    
    This fused operation combines:
    1. Split zxbcdt into z, xBC, dt
    2. Apply causal conv + activation to xBC
    3. Split xBC into x, B, C
    4. Run SSD with x*dt, A*dt, B, C
    5. Apply D skip connection
    6. Apply RMSNorm with gating
    7. Apply output projection
    
    Arguments:
        zxbcdt: (batch, seqlen, d_in_proj) concatenated input projections
        conv_weight: (conv_dim, kernel_width) convolution weight
        conv_bias: (conv_dim,) convolution bias
        dt_bias: (nheads,) dt bias
        A: (nheads,) A parameter (should be negative)
        D: (nheads,) D skip parameter
        chunk_size: chunk size for SSD
        d_inner: inner dimension
        ngroups: number of groups
        d_state: state dimension
        headdim: head dimension
        nheads: number of heads
        activation: activation function
        rmsnorm_weight: (d_inner,) RMSNorm weight
        rmsnorm_eps: RMSNorm epsilon
        norm_before_gate: whether to normalize before gating
        outproj_weight: (d_model, d_inner) output projection weight
        outproj_bias: (d_model,) output projection bias
        dt_limit: (min, max) dt limit tuple
        seq_idx: optional sequence indices
        initial_states: optional initial SSM states
        device: device for computation
    
    Returns:
        out: (batch, seqlen, d_model) output
        final_state: (batch, nheads, headdim, d_state) final SSM state
    """
    batch, seqlen, _ = zxbcdt.shape
    
    # 1. Split input
    z, xBC, dt = torch.split(
        zxbcdt,
        [d_inner, d_inner + 2 * ngroups * d_state, nheads],
        dim=-1,
    )
    
    # 2. Apply dt bias and softplus
    dt = F.softplus(dt + dt_bias)  # (batch, seqlen, nheads)
    
    # Apply dt_limit if specified
    if dt_limit != (0.0, float("inf")):
        dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1])
    
    # 3. Apply causal convolution with activation
    xBC = causal_conv1d_pytorch(xBC, conv_weight, conv_bias, activation=activation)
    
    # 4. Split xBC into x, B, C
    x, B, C = torch.split(
        xBC,
        [d_inner, ngroups * d_state, ngroups * d_state],
        dim=-1,
    )
    
    # 5. Reshape for SSD
    x = rearrange(x, "b l (h p) -> b l h p", p=headdim)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    
    # 6. Run SSD
    y, final_state = ssd(
        x * dt.unsqueeze(-1),
        A * dt,
        B,
        C,
        chunk_size,
        initial_states=initial_states,
        device=device,
    )
    
    # 7. Apply D skip connection
    y = y + x * D.unsqueeze(-1)
    
    # 8. Rearrange back
    y = rearrange(y, "b l h p -> b l (h p)")
    
    # 9. Apply RMSNorm with gating
    if rmsnorm_weight is not None:
        if norm_before_gate:
            # Normalize first, then gate
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + rmsnorm_eps) * rmsnorm_weight
            y = y * silu(z)
        else:
            # Gate first, then normalize (default)
            y = y * silu(z)
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + rmsnorm_eps) * rmsnorm_weight
    
    # 10. Apply output projection
    if outproj_weight is not None:
        y = F.linear(y, outproj_weight, outproj_bias)
    
    return y, final_state


class Mamba2LMHeadModel(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(args.vocab_size, args.d_model, device=device),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba2(args, device=device),
                                norm=RMSNorm(args.d_model, device=device),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),
                norm_f=RMSNorm(args.d_model, device=device),
            )
        )
        self.lm_head = nn.Linear(
            args.d_model, args.vocab_size, bias=False, device=device
        )
        self.lm_head.weight = self.backbone.embedding.weight

    @staticmethod
    def from_pretrained(huggingface_model_id: str, device: Device = None):
        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
        from transformers.utils.hub import cached_file

        config_path = cached_file(huggingface_model_id, CONFIG_NAME)
        assert config_path, "Failed to get huggingface config file"
        state_dict_path = cached_file(huggingface_model_id, WEIGHTS_NAME)
        assert state_dict_path, "Failed to get huggingface state dict file"

        config = json.load(open(config_path))
        args = Mamba2Config(
            d_model=config["d_model"],
            n_layer=config["n_layer"],
            vocab_size=config["vocab_size"],
            pad_vocab_size_multiple=config["pad_vocab_size_multiple"],
        )

        map_location = "cpu" if device is None else device
        state_dict = torch.load(
            state_dict_path, weights_only=True, map_location=map_location, mmap=True
        )
        model = Mamba2LMHeadModel(args, device=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(
        self, input_ids: LongTensor, h: list[InferenceCache] | list[None] | None = None
    ) -> tuple[LongTensor, list[InferenceCache]]:
        """
        Arguments
            input_ids: (batch, seqlen) tokens from `EleutherAI/gpt-neox-20b` tokenizer
            h: hidden states for inference step. If present the constant-time
               (wrt sequence length) inference path will be taken, input_ids
               should have shape (batch, 1) containing the next batch of prompt
               token.

        Return (logits, h)
            logits: (batch, seqlen, vocab_size)
            h: updated inference cache after processing `input_ids`
        """
        seqlen = input_ids.shape[1]

        if h is None:
            h = [None for _ in range(self.args.n_layer)]

        x = self.backbone.embedding(input_ids)
        for i, layer in enumerate(self.backbone.layers):
            y, h[i] = layer.mixer(layer.norm(x), h[i])
            x = y + x

        x = self.backbone.norm_f(x)
        logits = self.lm_head(x)
        return logits[:, :seqlen], cast(list[InferenceCache], h)

    def generate(
        self,
        input_ids: LongTensor,
        max_new_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 0,
    ) -> Iterable[tuple[int, list[InferenceCache]]]:
        prefix, tokens = input_ids[:-1], input_ids[-1:].unsqueeze(0)

        # Process prompt
        # The input sequence to forward (non-inference path) must have length multiple that of chunk_size.
        # We split out excess tokens so that n_chunked tokens can be processed by one forward call and
        # process the rest in multiple inference steps.
        n_chunked = (prefix.shape[0] // self.args.chunk_size) * self.args.chunk_size
        if n_chunked > 0:
            _, h = self(prefix[:n_chunked].unsqueeze(0), None)
        else:
            h = [
                InferenceCache.alloc(1, self.args, device=self.device)
                for _ in range(self.args.n_layer)
            ]
        for i in range(n_chunked, prefix.shape[0]):
            _, h = self(prefix[i : i + 1].unsqueeze(0), h)

        # Generate
        for _ in range(max_new_length):
            with torch.no_grad():
                out, h = self(tokens, h)
            logits = out[0, -1]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, k=top_k)[0][-1]
                logits[indices_to_remove] = -torch.inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > 0.5
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == eos_token_id:
                return
            tokens = next_token.unsqueeze(0)
            yield cast(int, next_token.item()), h


class Mamba2(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * args.d_inner + 2 * args.ngroups * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device)

        conv_dim = args.d_inner + 2 * args.ngroups * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            bias=args.conv_bias,
            padding=args.d_conv - 1,
            device=device,
        )
        
        # Initialize convolution weights if conv_init is specified
        if args.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -args.conv_init, args.conv_init)

        # Learnable initial states
        if args.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(args.nheads, args.headdim, args.d_state, device=device)
            )
            self.init_states._no_weight_decay = True
        else:
            self.init_states = None

        # Initialize dt_bias using inverse softplus
        import math
        dt = torch.exp(
            torch.rand(args.nheads, device=device)
            * (math.log(args.dt_max) - math.log(args.dt_min))
            + math.log(args.dt_min)
        )
        dt = torch.clamp(dt, min=args.dt_init_floor)
        inv_dt = inverse_softplus(dt)
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Initialize A_log from uniform range
        assert args.A_init_range[0] > 0 and args.A_init_range[1] >= args.A_init_range[0]
        A = torch.empty(args.nheads, dtype=torch.float32, device=device).uniform_(
            *args.A_init_range
        )
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(args.nheads, device=device))
        self.D._no_weight_decay = True

        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)

    def forward(self, u: Tensor, h: InferenceCache | None = None, seq_idx: Tensor | None = None):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.
            seq_idx: (seqlen,) optional sequence indices for chunked processing

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing `u`
        """
        if h:
            return self.step(u, h)

        batch, seqlen = u.shape[0], u.shape[1]
        A = -torch.exp(self.A_log)  # (nheads,)
        
        # Get initial states
        initial_states = None
        if self.args.learnable_init_states and self.init_states is not None:
            # initial_states shape: (nheads, headdim, d_state)
            # Need shape: (batch, 1, nheads, headdim, d_state) for ssd
            initial_states = repeat(self.init_states, "h p n -> b 1 h p n", b=batch)
        
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        
        # Use fused path if enabled
        if self.args.use_mem_eff_path:
            y, ssm_state = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                self.D,
                self.args.chunk_size,
                self.args.d_inner,
                self.args.ngroups,
                self.args.d_state,
                self.args.headdim,
                self.args.nheads,
                activation=self.args.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                norm_before_gate=self.norm.norm_before_gate,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                dt_limit=self.args.dt_limit,
                seq_idx=seq_idx,
                initial_states=initial_states,
                device=self.device,
            )
            
            # For inference cache, we need to compute conv_state
            # Conv state: last d_conv positions of xBC
            z, xBC, dt = torch.split(
                zxbcdt,
                [
                    self.args.d_inner,
                    self.args.d_inner + 2 * self.args.ngroups * self.args.d_state,
                    self.args.nheads,
                ],
                dim=-1,
            )
            conv_state = F.pad(
                rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
            )
            
            h = InferenceCache(conv_state, ssm_state)
            return y, h
        
        # Simple (non-fused) path
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.ngroups * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)
        
        # Apply dt_limit if specified
        if self.args.dt_limit != (0.0, float("inf")):
            dt = torch.clamp(dt, min=self.args.dt_limit[0], max=self.args.dt_limit[1])

        # Pad or truncate xBC seqlen to d_conv
        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )

        # Apply convolution with activation
        if self.args.activation in ["silu", "swish"]:
            xBC = silu(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen, :]
            )  # (batch, seqlen, d_inner + 2 * ngroups * d_state)
        else:
            raise NotImplementedError(f"Activation {self.args.activation} not implemented")
            
        x, B, C = torch.split(
            xBC, 
            [self.args.d_inner, self.args.ngroups * self.args.d_state, self.args.ngroups * self.args.d_state], 
            dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        
        # Reshape B and C for ngroups
        B = rearrange(B, "b l (g n) -> b l g n", g=self.args.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.args.ngroups)
        
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            B,
            C,
            self.args.chunk_size,
            initial_states=initial_states,
            device=self.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)
        return y, h

    def step(self, u: Tensor, h: InferenceCache) -> tuple[Tensor, InferenceCache]:
        """Take a single inference step for the current input and hidden state

        Unlike attention-based models, RNN-based models (eg Mamba) does not need
        to look back at all the past tokens to generate a new token. Instead a
        hidden state (initialized to 0s initially) is updated for each input and
        passed to the next inference step. This means that the total inference
        time is linear with respect to the sequence length instead of quadratic
        in attention's case.

        Arguments
            u: (batch, 1, d_model)
            h: initial/running hidden state

        Return (y, h)
            y: (batch, 1, d_model)
            h: updated hidden state
        """
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.ngroups * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )

        # Advance convolution input
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        # Convolution step
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        if self.conv1d.bias is not None:
            xBC += self.conv1d.bias
        xBC = silu(xBC)

        x, B, C = torch.split(
            xBC, 
            [self.args.d_inner, self.args.ngroups * self.args.d_state, self.args.ngroups * self.args.d_state], 
            dim=-1
        )
        A = -torch.exp(self.A_log)  # (nheads,)

        # SSM step
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        
        # Apply dt_limit if specified
        if self.args.dt_limit != (0.0, float("inf")):
            dt = torch.clamp(dt, min=self.args.dt_limit[0], max=self.args.dt_limit[1])
            
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        
        # For ngroups support in step: expand B and C to per-head shapes
        # When ngroups > 1, B and C have shape (batch, ngroups * d_state)
        # We need to repeat groups to match heads
        if self.args.ngroups < self.args.nheads:
            # Reshape B and C for groups
            B_grouped = rearrange(B, "b (g n) -> b g n", g=self.args.ngroups)
            C_grouped = rearrange(C, "b (g n) -> b g n", g=self.args.ngroups)
            # Repeat groups to heads
            heads_per_group = self.args.nheads // self.args.ngroups
            B = repeat(B_grouped, "b g n -> b (g h) n", h=heads_per_group)
            C = repeat(C_grouped, "b g n -> b (g h) n", h=heads_per_group)
        
        dBx = torch.einsum("bh, bhn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bhn -> bhp", h.ssm_state, C)
            
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_groups, d_state) where n_groups can equal n_heads
        C: (batch, seqlen, n_groups, d_state) where n_groups can equal n_heads

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    assert x.shape[1] % chunk_size == 0
    
    # When ngroups < nheads, expand B and C to match nheads by repeating groups
    nheads = x.shape[2]
    ngroups = B.shape[2]
    if ngroups < nheads:
        # Expand groups to heads: each group is repeated for multiple heads
        assert nheads % ngroups == 0, f"nheads ({nheads}) must be divisible by ngroups ({ngroups})"
        heads_per_group = nheads // ngroups
        B = repeat(B, "b l g n -> b l (g h) n", h=heads_per_group)
        C = repeat(C, "b l g n -> b l (g h) n", h=heads_per_group)

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader 😜
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, norm_before_gate: bool = False, device: Device = None):
        """Gated Root Mean Square Layer Normalization (RMSNormGated)

        Paper: https://arxiv.org/abs/1910.07467
        
        Args:
            d: dimension
            eps: epsilon for numerical stability
            norm_before_gate: if True, normalize before gating; if False, gate before normalizing
        """
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            if self.norm_before_gate:
                # Normalize first, then gate
                x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
                x = x * silu(z)
            else:
                # Gate first, then normalize (default behavior matching repo)
                x = x * silu(z)
                x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        else:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return x


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * F.sigmoid(x)
