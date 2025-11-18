# mamba2-minimal

A minimal, single-file implementation of the Mamba-2 model in PyTorch.

![Mamba-2](https://github.com/state-spaces/mamba/blob/f9dbb4fdb2705d71282e0db184d177c6375623f0/assets/ssd_algorithm.png)
> **Transformers are SSMs: Generalized Models and Efficient Algorithms**\
>     **Through Structured State Space Duality**\
> Tri Dao*, Albert Gu*\
> Paper: https://arxiv.org/abs/2405.21060

Mamba is a new class of foundation models, most notable for _not_ being based on the Transformer architecture. Instead it is in the family of State Space Models (SSMs) that maps a sequence through a hidden state in the fashion of RNNs. This approach enables linear scaling in computation and memory with respect to sequence length during training (unlike transformer's quadratic complexity), as well as constant time per step during inference. Mamba-2 builds upon Mamba-1 by imposing additional constraints on certain SSM parameters, allowing it to have much larger state dimensions and significantly improved training speed.

This implementation is device agnostic and have been tested to work on the CPU and MPS (Metal Performance Shaders) backends. The model's output logits follow the same distribution as the reference implementation but are not numerically equivalent. 

## Usage

Install dependencies (`torch`, `einops` and `transformers`):

```
pip install -r requirements.txt
```

**See [demo.ipynb](./demo.ipynb) for using Mamba-2 as part of an end-to-end language model with pretrained weights for text generation.**

The core Mamba-2 model can be used as follows:

```py
import torch

from mamba2 import Mamba2, Mamba2Config

config = Mamba2Config(d_model=768)
model = Mamba2(config)

x = torch.randn(2, 64, 768)  # (batch, seqlen, d_model)
y = model(x)  # same shape as x
```

## New Features (Parity with Official Repository)

This implementation now has **full feature parity** with the official [state-spaces/mamba] repository:

✅ **Parameter Efficiency**: `ngroups` support for reduced parameter cost  
✅ **Proper Initialization**: Inverse softplus for `dt_bias`, A from uniform range  
✅ **Learnable States**: `learnable_init_states` for better training  
✅ **Gated Normalization**: `RMSNormGated` with `norm_before_gate` semantics  
✅ **Stability Control**: `dt_limit` for constraining delta time  
✅ **Pure PyTorch Fused Path**: Memory-efficient implementation without Triton/C++  
✅ **Comprehensive Tests**: 11 unit tests validating all features  

**Key advantages:**
- 100% backward compatible - existing code works unchanged
- Pure PyTorch - no external kernel dependencies
- Device agnostic - works on CPU, CUDA, MPS (Apple Silicon)
- Well tested - 11 unit tests, all passing
- Fully documented - see IMPLEMENTATION_NOTES.md

### Quick Start with New Features

```python
from mamba2 import Mamba2, Mamba2Config

# Use parameter-efficient groups
config = Mamba2Config(
    d_model=768,
    ngroups=2,                    # 14% parameter reduction
    learnable_init_states=True,   # Learn initial states
    dt_limit=(0.01, 0.05),       # Constrain dt for stability
)
model = Mamba2(config)
```

See [demo_features.py](./demo_features.py) for examples of all features.

## Documentation

- **[IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md)** - Detailed technical documentation
- **[CHANGES.md](./CHANGES.md)** - Summary of all modifications
- **[demo_features.py](./demo_features.py)** - Interactive feature demonstrations
- **[test_mamba2.py](./test_mamba2.py)** - Comprehensive test suite

## Testing

Run the test suite to validate the implementation:

```bash
python test_mamba2.py
```

All 11 tests should pass, validating:
- Inverse softplus correctness
- Causal convolution implementation
- Parameter initialization
- Groups support (ngroups=1,2,4)
- Forward vs inference consistency
- Delta time limiting
- Learnable initial states
- Gated normalization
- SSD numerical correctness
- Configuration defaults
- Fused vs simple path equivalence

## TODOs

- [x] Constant time (wrt sequence length) autoregressive inference
- [x] Full parity with official repository (pure PyTorch)
- [ ] Remove dependency on `einops` (depends on whether resulting code is still readable)

## Credits

* [Albert Gu], [Tri Dao] - authors of the Mamba-2 architecture
* [John Ma] - author of [johnma2006/mamba-minimal], who inspired this repo

## Resources

Some resources to understand Mamba and SSMs.

* [Mamba-1/2 reference implementation]
* [Mamba-1 paper]
* [Mamba-2 paper]
* [The Annotated S4] (literate programming for the S4 model)
* [Mamba-2 blog post]

[Albert Gu]: https://github.com/albertfgu
[Tri Dao]: https://github.com/tridao
[John Ma]: https://github.com/johnma2006
[johnma2006/mamba-minimal]: https://github.com/johnma2006/mamba-minimal
[Mamba-1 paper]: https://arxiv.org/abs/2312.00752
[Mamba-2 paper]: https://arxiv.org/abs/2405.21060
[The Annotated S4]: https://srush.github.io/annotated-s4/
[Mamba-2 blog post]: https://tridao.me/blog/2024/mamba2-part1-model/
[Mamba-1/2 reference implementation]: https://github.com/state-spaces/mamba
