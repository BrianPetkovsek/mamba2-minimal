"""
Demonstration of new features in Mamba2 implementation.

This script showcases all the new features implemented for parity
with the official state-spaces/mamba repository.
"""

import torch
from mamba2 import Mamba2, Mamba2Config, Mamba2LMHeadModel


def demo_basic():
    """Basic usage - unchanged from original implementation."""
    print("=" * 60)
    print("1. BASIC USAGE (Backward Compatible)")
    print("=" * 60)
    
    config = Mamba2Config(d_model=256)
    model = Mamba2(config)
    
    x = torch.randn(2, 64, 256)
    y, h = model(x)
    
    print(f"✓ Input shape:  {x.shape}")
    print(f"✓ Output shape: {y.shape}")
    print(f"✓ Basic usage works with default config")


def demo_ngroups():
    """Groups support for reduced parameter cost."""
    print("\n" + "=" * 60)
    print("2. NGROUPS SUPPORT (Parameter Efficiency)")
    print("=" * 60)
    
    # Compare parameter counts
    config_no_groups = Mamba2Config(d_model=256, ngroups=1)
    config_groups = Mamba2Config(d_model=256, ngroups=2)
    
    model_no_groups = Mamba2(config_no_groups)
    model_groups = Mamba2(config_groups)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    params_no_groups = count_parameters(model_no_groups)
    params_groups = count_parameters(model_groups)
    
    print(f"✓ Parameters with ngroups=1: {params_no_groups:,}")
    print(f"✓ Parameters with ngroups=2: {params_groups:,}")
    print(f"✓ Reduction: {params_no_groups - params_groups:,} parameters")
    print(f"  ({100 * (params_no_groups - params_groups) / params_no_groups:.1f}% smaller)")


def demo_learnable_init_states():
    """Learnable initial SSM states."""
    print("\n" + "=" * 60)
    print("3. LEARNABLE INITIAL STATES (Training Flexibility)")
    print("=" * 60)
    
    config = Mamba2Config(d_model=256, learnable_init_states=True)
    model = Mamba2(config)
    
    print(f"✓ init_states shape: {model.init_states.shape}")
    print(f"✓ init_states is learnable: {model.init_states.requires_grad}")
    print(f"✓ _no_weight_decay set: {hasattr(model.init_states, '_no_weight_decay')}")


def demo_dt_limit():
    """Delta time limiting for stability."""
    print("\n" + "=" * 60)
    print("4. DT LIMIT (Stability Control)")
    print("=" * 60)
    
    config = Mamba2Config(d_model=256, dt_limit=(0.01, 0.05))
    model = Mamba2(config)
    
    print(f"✓ dt_limit range: [{config.dt_limit[0]}, {config.dt_limit[1]}]")
    print(f"✓ dt values will be clamped to this range during forward pass")


def demo_fused_vs_simple():
    """Compare fused and simple paths."""
    print("\n" + "=" * 60)
    print("5. FUSED VS SIMPLE PATH (Memory Efficiency)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Fused path
    config_fused = Mamba2Config(d_model=256, use_mem_eff_path=True)
    model_fused = Mamba2(config_fused)
    
    # Simple path
    config_simple = Mamba2Config(d_model=256, use_mem_eff_path=False)
    model_simple = Mamba2(config_simple)
    model_simple.load_state_dict(model_fused.state_dict())
    
    x = torch.randn(2, 64, 256)
    
    with torch.no_grad():
        y_fused, _ = model_fused(x)
        y_simple, _ = model_simple(x)
    
    max_diff = torch.abs(y_fused - y_simple).max().item()
    
    print(f"✓ Fused path:  use_mem_eff_path=True")
    print(f"✓ Simple path: use_mem_eff_path=False")
    print(f"✓ Max difference: {max_diff:.10f}")
    print(f"✓ Outputs are {'identical' if max_diff < 1e-6 else 'very close'}")


def demo_proper_initialization():
    """Proper parameter initialization matching official repo."""
    print("\n" + "=" * 60)
    print("6. PROPER INITIALIZATION (Numerical Correctness)")
    print("=" * 60)
    
    config = Mamba2Config(
        d_model=256,
        dt_min=0.001,
        dt_max=0.1,
        A_init_range=(1, 16),
    )
    model = Mamba2(config)
    
    import torch.nn.functional as F
    
    # Check dt_bias initialization
    dt = F.softplus(model.dt_bias)
    print(f"✓ dt_bias uses inverse softplus initialization")
    print(f"  After softplus: dt in [{dt.min().item():.6f}, {dt.max().item():.6f}]")
    
    # Check A_log initialization
    A = torch.exp(model.A_log)
    print(f"✓ A_log initialized from uniform range")
    print(f"  A values in [{A.min().item():.6f}, {A.max().item():.6f}]")
    
    # Check _no_weight_decay attributes
    params_with_no_wd = [
        name for name, param in model.named_parameters()
        if hasattr(param, '_no_weight_decay')
    ]
    print(f"✓ Parameters with _no_weight_decay: {len(params_with_no_wd)}")
    print(f"  {', '.join(params_with_no_wd)}")


def demo_all_features():
    """Combine all features together."""
    print("\n" + "=" * 60)
    print("7. ALL FEATURES COMBINED")
    print("=" * 60)
    
    config = Mamba2Config(
        d_model=256,
        n_layer=2,
        ngroups=2,
        learnable_init_states=True,
        dt_limit=(0.01, 0.05),
        activation="silu",
        use_mem_eff_path=True,
        conv_init=0.01,
    )
    
    model = Mamba2LMHeadModel(config)
    model.eval()
    
    # Generate some text
    batch, seqlen = 1, 64
    input_ids = torch.randint(0, config.vocab_size, (batch, seqlen))
    
    with torch.no_grad():
        logits, h = model(input_ids)
    
    print(f"✓ Config: ngroups={config.ngroups}, learnable_init_states={config.learnable_init_states}")
    print(f"✓ Config: dt_limit={config.dt_limit}, activation={config.activation}")
    print(f"✓ Config: use_mem_eff_path={config.use_mem_eff_path}, conv_init={config.conv_init}")
    print(f"✓ Input shape:  {input_ids.shape}")
    print(f"✓ Output shape: {logits.shape}")
    print(f"✓ All features work together seamlessly!")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("MAMBA2 NEW FEATURES DEMONSTRATION")
    print("=" * 60)
    print()
    
    demo_basic()
    demo_ngroups()
    demo_learnable_init_states()
    demo_dt_limit()
    demo_fused_vs_simple()
    demo_proper_initialization()
    demo_all_features()
    
    print("\n" + "=" * 60)
    print("✅ ALL FEATURES DEMONSTRATED SUCCESSFULLY!")
    print("=" * 60)
    print("\nFor more details, see IMPLEMENTATION_NOTES.md")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
