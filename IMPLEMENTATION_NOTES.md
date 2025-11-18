# Mamba2 Implementation Notes

This document describes the changes made to bring the minimal Mamba-2 PyTorch implementation to parity with the official `state-spaces/mamba` repository.

## Overview

All changes are implemented in **pure PyTorch** without any Triton, CUDA, or C++ dependencies. The implementation maintains compatibility with CPU, CUDA, and MPS (Apple Silicon) devices.

## Key Features Implemented

### 1. Parameter Initialization (High Priority)

#### `dt_bias` Initialization with Inverse Softplus
- **Why**: The official repo uses an inverse softplus transform to initialize `dt_bias` so that after applying `softplus(dt_bias + noise)`, the resulting `dt` values are in the desired range `[dt_min, dt_max]`.
- **Implementation**: Added `inverse_softplus()` function and updated initialization:
  ```python
  dt = torch.exp(torch.rand(nheads) * (log(dt_max) - log(dt_min)) + log(dt_min))
  dt = torch.clamp(dt, min=dt_init_floor)
  dt_bias = inverse_softplus(dt)
  ```

#### `A_log` Initialization from Uniform Range
- **Why**: The official repo initializes A from a uniform distribution in range `[1, 16]` by default, then takes the log.
- **Implementation**:
  ```python
  A = torch.empty(nheads).uniform_(*A_init_range)
  A_log = torch.log(A)
  ```

#### Weight Decay Attributes
- **Why**: Certain parameters (`dt_bias`, `A_log`, `D`, `init_states`) should not have weight decay applied during training.
- **Implementation**: Set `._no_weight_decay = True` on these parameters.

### 2. Groups Support (`ngroups`)

- **Why**: The official implementation uses `ngroups` to reduce parameter cost. When `ngroups < nheads`, multiple heads share the same B and C parameters.
- **Implementation**:
  - Updated `d_in_proj` calculation: `2 * d_inner + 2 * ngroups * d_state + nheads`
  - Updated `conv_dim`: `d_inner + 2 * ngroups * d_state`
  - Modified `ssd()` to expand groups to heads when needed
  - Updated step() method to handle grouped B and C tensors
- **Default**: `ngroups=1` (each head has its own parameters)

### 3. Learnable Initial States

- **Why**: Allows the model to learn better initial SSM states instead of always starting from zeros.
- **Implementation**:
  - Added `learnable_init_states` config parameter
  - When enabled, creates `init_states` parameter with shape `(nheads, headdim, d_state)`
  - In forward pass, repeats initial states for batch: `repeat(init_states, "h p n -> b 1 h p n", b=batch)`
- **Default**: `False`

### 4. RMSNormGated (Gated Normalization)

- **Why**: The official repo uses `RMSNormGated` which supports two gating modes: normalize-then-gate or gate-then-normalize.
- **Implementation**:
  - Added `norm_before_gate` parameter to `RMSNorm`
  - When `norm_before_gate=False` (default): `x = x * silu(z)`, then normalize
  - When `norm_before_gate=True`: normalize first, then `x = x * silu(z)`
- **Default**: `norm_before_gate=False` (matches official repo)

### 5. Delta Time Limiting (`dt_limit`)

- **Why**: Allows constraining dt values to a specific range for stability.
- **Implementation**: After `dt = softplus(dt + dt_bias)`, apply `torch.clamp(dt, min=dt_limit[0], max=dt_limit[1])`
- **Default**: `(0.0, inf)` (no limiting)

### 6. Activation Selection

- **Why**: Official repo supports both "silu" and "swish" activation functions (they're equivalent).
- **Implementation**: Added `activation` config parameter
- **Default**: `"swish"`

### 7. Convolution Options

- **Why**: For flexibility in initialization and bias usage.
- **Implementation**:
  - `conv_bias`: Whether to use bias in Conv1d layer (default: `True`)
  - `conv_init`: If specified, initializes conv weights uniformly in `[-conv_init, conv_init]`

### 8. Pure PyTorch Fused Path

#### `causal_conv1d_pytorch()`
- **Purpose**: Pure PyTorch replacement for the `causal_conv1d` package
- **Implementation**: Uses `F.conv1d` with `padding=kernel_width-1` and truncates output to maintain causality
- **Benefits**: No external dependencies, works on all devices

#### `mamba_split_conv1d_scan_combined()`
- **Purpose**: Pure PyTorch equivalent of the Triton `mamba_split_conv1d_scan_combined` kernel
- **Operations**:
  1. Split `zxbcdt` into `z`, `xBC`, `dt`
  2. Apply `softplus` to dt with bias
  3. Apply causal convolution + activation to xBC
  4. Split xBC into `x`, `B`, `C`
  5. Run SSD algorithm
  6. Apply D skip connection
  7. Apply RMSNorm with gating
  8. Apply output projection
- **Benefits**: Single fused forward pass, cleaner code flow

#### `use_mem_eff_path` Toggle
- **Why**: Allows switching between fused (efficient) and simple (easier to debug) paths
- **Implementation**: When `True`, uses `mamba_split_conv1d_scan_combined()`; when `False`, uses step-by-step operations
- **Default**: `True`
- **Note**: Both paths produce bit-exact identical results

### 9. Sequence Indexing Support (`seq_idx`)

- **Why**: Official repo supports `seq_idx` for chunked processing
- **Implementation**: Added `seq_idx` parameter to forward signature and passed through to fused kernel
- **Status**: Signature in place, full implementation can be extended as needed

## Configuration Defaults

All new parameters have sensible defaults that maintain backward compatibility:

```python
ngroups: int = 1
A_init_range: tuple[float, float] = (1, 16)
dt_min: float = 0.001
dt_max: float = 0.1
dt_init_floor: float = 1e-4
dt_limit: tuple[float, float] = (0.0, float("inf"))
learnable_init_states: bool = False
activation: str = "swish"
conv_init: float | None = None
conv_bias: bool = True
use_mem_eff_path: bool = True
```

## Testing

A comprehensive test suite (`test_mamba2.py`) validates all features:

1. `test_inverse_softplus()` - Validates inverse softplus correctness
2. `test_causal_conv1d()` - Tests causal convolution implementation
3. `test_parameter_initialization()` - Validates all parameter initializations
4. `test_ngroups_support()` - Tests ngroups=1,2,4
5. `test_forward_vs_inference_consistency()` - Tests step-by-step inference
6. `test_dt_limit()` - Validates dt limiting
7. `test_learnable_init_states()` - Tests learnable initial states
8. `test_rmsnorm_gated()` - Tests both gating modes
9. `test_ssd_numerical()` - Tests SSD with ngroups
10. `test_config_defaults()` - Validates all default values
11. `test_fused_vs_simple_path()` - Validates fused and simple paths produce identical results

All tests pass successfully.

## Performance Notes

- **Fused Path**: Combines multiple operations into a single function call, reducing Python overhead
- **Pure PyTorch**: While not as fast as Triton kernels, the pure PyTorch implementation:
  - Works on any device (CPU, CUDA, MPS)
  - Is easier to debug and understand
  - Produces numerically identical results
  - Requires no compilation or external dependencies

## Compatibility

- **Backward Compatible**: Existing code using default config values works unchanged
- **Forward Compatible**: All new features match official repo behavior
- **Device Agnostic**: Works on CPU, CUDA, and MPS (Apple Silicon)
- **No External Dependencies**: Only requires `torch`, `einops`, and `transformers`

## Usage Examples

### Basic Usage (Unchanged)
```python
from mamba2 import Mamba2, Mamba2Config

config = Mamba2Config(d_model=768)
model = Mamba2(config)
x = torch.randn(2, 64, 768)
y, h = model(x)
```

### Using New Features
```python
# With groups and learnable init states
config = Mamba2Config(
    d_model=768,
    ngroups=2,  # Share parameters across heads
    learnable_init_states=True,  # Learn initial states
    dt_limit=(0.01, 0.05),  # Constrain dt range
    activation="silu",  # Use SiLU activation
)

# Use simple path for debugging
config_debug = Mamba2Config(
    d_model=768,
    use_mem_eff_path=False,  # Use step-by-step simple path
)
```

## References

- Official Implementation: https://github.com/state-spaces/mamba
- Mamba-2 Paper: https://arxiv.org/abs/2405.21060
- Reference Files:
  - `mamba_ssm/modules/mamba2_simple.py`
  - `mamba_ssm/modules/ssd_minimal.py`
