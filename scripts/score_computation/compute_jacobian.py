import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer

from olmo.custom_hf_model import CustomOlmo2ForCausalLM, CustomOlmo2MoEForCausalLM
from olmo.custom_hf_model_config import CustomOlmo2Config, CustomOlmo2MoEConfig
from olmo.data.memmap_dataset import MemMapDataset


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")

    config = CustomOlmo2Config.from_pretrained(model_path)
    is_moe = hasattr(config, "model_type") and config.model_type == "customolmo2moe"

    if is_moe:
        print(f"Detected MoE model")
        model = CustomOlmo2MoEForCausalLM.from_pretrained(
            model_path, config=config, dtype=torch.bfloat16, device_map=device
        )
    else:
        print("Loaded as standard (dense) model")
        model = CustomOlmo2ForCausalLM.from_pretrained(
            model_path, config=config, dtype=torch.bfloat16, device_map=device
        )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, is_moe


def create_sample_data(tokenizer, num_samples: int = 50, seq_length: int = 512) -> Dict:
    """Create sample data for evaluation."""
    print(f"Creating {num_samples} sample sequences...")

    dataset = MemMapDataset(
        "./data/FineWeb/part-00019.npy",
        chunk_size=seq_length,
    )

    input_ids_list = []
    attention_masks = []

    for _ in range(num_samples):
        input_ids = dataset[np.random.randint(0, len(dataset))]["input_ids"]
        attention_mask = torch.ones_like(input_ids)

        input_ids_list.append(input_ids)
        attention_masks.append(attention_mask)

    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_masks)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def compute_jacobian_rowwise(output: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian row-by-row for memory efficiency.

    For each output dimension, computes gradient of output w.r.t input.

    Args:
        output: Output tensor of shape [batch, seq_len, d_model]
        input_tensor: Input tensor of shape [batch, seq_len, d_model]

    Returns:
        Jacobian matrix of shape [d_model, d_model] averaged over batch and sequence
    """
    d_model = output.shape[-1]
    jacobian_list = []

    for i in range(d_model):
        # Create gradient for one output dimension
        grad_out = torch.zeros_like(output)
        grad_out[..., i] = 1.0

        # Compute gradient
        grad = torch.autograd.grad(
            output,
            input_tensor,
            grad_outputs=grad_out,
            retain_graph=True,
            create_graph=False,
        )[0]

        # Average over batch and sequence dimensions
        jacobian_list.append(grad.mean(dim=(0, 1)).cpu())

    # Stack to form Jacobian matrix
    jacobian = torch.stack(jacobian_list, dim=-1)  # [d_model, d_model]

    return jacobian


def compute_jacobian_elementwise(output: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian element-wise (one batch position at a time).

    For each batch position and output dimension, computes gradient separately.
    This method misses cross-position dependencies but computes position-specific Jacobians.

    Args:
        output: Output tensor of shape [batch, seq_len, d_model]
        input_tensor: Input tensor of shape [batch, seq_len, d_model]

    Returns:
        Jacobian matrix of shape [d_model, d_model] averaged over batch and sequence
    """
    jacobian = []
    batch_size = output.shape[0]

    for batch_idx in range(batch_size):
        jacobian_one_sample = []
        for embed_idx in range(output.shape[-1]):  # Iterate over each output dimension
            grad_out = torch.zeros_like(output)
            grad_out[batch_idx, ..., embed_idx] = 1  # Select one output unit at one batch position
            grad = torch.autograd.grad(
                output,
                input_tensor,
                grad_outputs=grad_out,
                retain_graph=True,
                create_graph=False,
            )[0]
            jacobian_one_sample.append(grad.cpu().unsqueeze(0))

        jacobian.append(torch.stack(jacobian_one_sample, dim=-1))

    full_jacobian = torch.stack(jacobian, dim=0)
    # Average over batch & seq dim
    return full_jacobian.mean(dim=list(range(len(full_jacobian.size()) - 2)))


def compute_layer_jacobian(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    layer_idx: int,
    method: str = "rowwise",
) -> Dict[str, float]:
    """
    Compute Jacobian metrics for a specific layer's residual connection.

    For layer l, computes Jacobian of output w.r.t input, measuring how much
    the layer transforms its input.

    Args:
        model: The model
        input_ids: Input token IDs
        attention_mask: Attention mask
        device: Device to run on
        layer_idx: Layer index to analyze
        method: 'rowwise' or 'elementwise' - method to compute Jacobian

    Returns:
        Dictionary with Jacobian metrics (both rowwise and element-wise if method='both'):
        - 'jacobian_norm': ||J - I|| (deviation from identity)
        - 'off_diagonal_norm': ||J_off_diag|| (non-identity components)
        - 'diagonal_mean': Mean of diagonal elements
        - 'diagonal_std': Std of diagonal elements
        - 'frobenius_norm': Frobenius norm of J
        - 'spectral_norm': Spectral norm (max singular value) of J
        - If method='both', also includes '_elementwise' versions of all metrics
    """
    model.eval()

    # Storage for layer input during forward pass
    layer_input_storage = {"input": None, "captured": False}

    def hook_fn(module, input, output):
        """Forward hook to capture layer input with gradient tracking."""
        if not layer_input_storage["captured"]:
            layer_input = input[0] if isinstance(input, tuple) else input
            # Store the input tensor directly (it has gradients enabled)
            layer_input_storage["input"] = layer_input
            layer_input_storage["captured"] = True

    # Register hook
    hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    # Enable gradients
    with torch.enable_grad():
        batch_size = input_ids.shape[0]
        micro_batch_size = min(2, batch_size)
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

        jacobian_accum_rowwise = None
        jacobian_accum_elementwise = None
        total_samples = 0  # Track total number of samples for proper averaging

        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)
            micro_batch_actual_size = end_idx - start_idx

            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)

            # Reset capture flag
            layer_input_storage["captured"] = False

            # Forward pass with gradient tracking
            outputs = model(
                input_ids=micro_input_ids,
                attention_mask=micro_attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            # Get layer output from hidden_states
            layer_output = outputs.hidden_states[layer_idx + 1]
            if isinstance(layer_output, tuple):
                layer_output = layer_output[0] if layer_output else None

            # Get layer input from hook storage
            layer_input = layer_input_storage["input"]

            if layer_input is not None and layer_output is not None and isinstance(layer_output, torch.Tensor):
                # Track number of samples (batch_size * seq_len) for proper weighting
                num_samples_in_mb = micro_batch_actual_size * layer_output.shape[1]
                total_samples += num_samples_in_mb

                # Compute Jacobian using rowwise method (always compute for comparison)
                jacobian_rowwise = compute_jacobian_rowwise(layer_output, layer_input)

                # Weight by number of samples in this micro-batch
                if jacobian_accum_rowwise is None:
                    jacobian_accum_rowwise = jacobian_rowwise * num_samples_in_mb
                else:
                    jacobian_accum_rowwise += jacobian_rowwise * num_samples_in_mb

                # Compute element-wise if requested
                if method in ["elementwise", "both"]:
                    jacobian_elementwise = compute_jacobian_elementwise(layer_output, layer_input)
                    if jacobian_accum_elementwise is None:
                        jacobian_accum_elementwise = jacobian_elementwise * num_samples_in_mb
                    else:
                        jacobian_accum_elementwise += jacobian_elementwise * num_samples_in_mb

            del outputs
            if device == "cuda":
                torch.cuda.empty_cache()

    # Remove hook
    hook_handle.remove()

    # Average Jacobian across micro-batches (weighted by number of samples)
    results = {}

    if total_samples > 0 and jacobian_accum_rowwise is not None:
        jacobian_avg_rowwise = jacobian_accum_rowwise / total_samples
    else:
        # Return zeros if no valid data
        hidden_size = model.config.hidden_size
        jacobian_avg_rowwise = torch.zeros(hidden_size, hidden_size)

    # Compute metrics for rowwise
    d_model = jacobian_avg_rowwise.shape[0]
    identity = torch.eye(d_model, device=jacobian_avg_rowwise.device)

    # 1. Deviation from identity (normalized by number of elements)
    jacobian_diff = jacobian_avg_rowwise - identity
    jacobian_norm = torch.norm(jacobian_diff, p="fro").item() / math.sqrt(d_model * d_model)

    # 2. Off-diagonal norm (non-identity components, normalized by off-diagonal element count)
    jacobian_off_diag = jacobian_avg_rowwise.clone()
    jacobian_off_diag.diagonal().zero_()
    off_diagonal_norm = torch.norm(jacobian_off_diag, p="fro").item() / math.sqrt(d_model * d_model - d_model)

    # 3. Diagonal statistics
    diagonal_elements = jacobian_avg_rowwise.diagonal()
    diagonal_mean = diagonal_elements.mean().item()
    diagonal_std = diagonal_elements.std().item()

    # 4. Frobenius norm
    frobenius_norm = torch.norm(jacobian_avg_rowwise, p="fro").item()

    # 5. Spectral norm (max singular value) - convert to float32 for computation
    jacobian_float32 = jacobian_avg_rowwise.float()
    spectral_norm = torch.linalg.matrix_norm(jacobian_float32, ord=2).item()

    results.update(
        {
            "jacobian_norm": float(jacobian_norm),
            "off_diagonal_norm": float(off_diagonal_norm),
            "diagonal_mean": float(diagonal_mean),
            "diagonal_std": float(diagonal_std),
            "frobenius_norm": float(frobenius_norm),
            "spectral_norm": float(spectral_norm),
        }
    )

    # Compute element-wise metrics if requested
    if method in ["elementwise", "both"]:
        if total_samples > 0 and jacobian_accum_elementwise is not None:
            jacobian_avg_elementwise = jacobian_accum_elementwise / total_samples
        else:
            hidden_size = model.config.hidden_size
            jacobian_avg_elementwise = torch.zeros(hidden_size, hidden_size)

        # Compute metrics for element-wise
        identity_elem = torch.eye(d_model, device=jacobian_avg_elementwise.device)

        # 1. Deviation from identity (normalized by number of elements)
        jacobian_diff_elem = jacobian_avg_elementwise - identity_elem
        jacobian_norm_elem = torch.norm(jacobian_diff_elem, p="fro").item() / math.sqrt(d_model * d_model)

        # 2. Off-diagonal norm (normalized by off-diagonal element count)
        jacobian_off_diag_elem = jacobian_avg_elementwise.clone()
        jacobian_off_diag_elem.diagonal().zero_()
        off_diagonal_norm_elem = torch.norm(jacobian_off_diag_elem, p="fro").item() / math.sqrt(
            d_model * d_model - d_model
        )

        # 3. Diagonal statistics
        diagonal_elements_elem = jacobian_avg_elementwise.diagonal()
        diagonal_mean_elem = diagonal_elements_elem.mean().item()
        diagonal_std_elem = diagonal_elements_elem.std().item()

        # 4. Frobenius norm
        frobenius_norm_elem = torch.norm(jacobian_avg_elementwise, p="fro").item()

        # 5. Spectral norm
        jacobian_float32_elem = jacobian_avg_elementwise.float()
        spectral_norm_elem = torch.linalg.matrix_norm(jacobian_float32_elem, ord=2).item()

        results.update(
            {
                "jacobian_norm_elementwise": float(jacobian_norm_elem),
                "off_diagonal_norm_elementwise": float(off_diagonal_norm_elem),
                "diagonal_mean_elementwise": float(diagonal_mean_elem),
                "diagonal_std_elementwise": float(diagonal_std_elem),
                "frobenius_norm_elementwise": float(frobenius_norm_elem),
                "spectral_norm_elementwise": float(spectral_norm_elem),
            }
        )

    return results


def compute_all_layer_jacobians(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    num_layers: int,
    method: str = "rowwise",
) -> Dict[int, Dict[str, float]]:
    """
    Compute Jacobian metrics for all layers.

    Args:
        model: The model
        input_ids: Input token IDs
        attention_mask: Attention mask
        device: Device to run on
        num_layers: Number of layers
        method: 'rowwise', 'elementwise', or 'both' - method to compute Jacobian

    Returns:
        Dictionary mapping layer index to Jacobian metrics
    """
    print(f"Computing Jacobian metrics for all layers using method: {method}...")

    layer_jacobians = {}

    for layer_idx in tqdm(range(num_layers), desc="Computing layer Jacobians"):
        metrics = compute_layer_jacobian(model, input_ids, attention_mask, device, layer_idx, method=method)
        layer_jacobians[layer_idx] = metrics

    return layer_jacobians


def compute_global_jacobian_metrics(layer_jacobians: Dict[int, Dict[str, float]], num_layers: int) -> Dict[str, float]:
    """
    Compute global metrics summarizing Jacobian analysis across all layers.

    Args:
        layer_jacobians: Dictionary of layer index to Jacobian metrics
        num_layers: Number of layers

    Returns:
        Dictionary with global metrics:
        - 'mean_jacobian_norm': Mean deviation from identity
        - 'mean_off_diagonal_norm': Mean off-diagonal norm
        - 'mean_diagonal_mean': Mean diagonal value
        - 'identity_preservation_score': Fraction of layers with diagonal_mean > 0.5
        - 'residual_strength': Average diagonal dominance
    """
    # Extract metrics
    jacobian_norms = [m["jacobian_norm"] for m in layer_jacobians.values()]
    off_diagonal_norms = [m["off_diagonal_norm"] for m in layer_jacobians.values()]
    diagonal_means = [m["diagonal_mean"] for m in layer_jacobians.values()]

    # Compute global metrics
    mean_jacobian_norm = np.mean(jacobian_norms)
    mean_off_diagonal_norm = np.mean(off_diagonal_norms)
    mean_diagonal_mean = np.mean(diagonal_means)

    # Identity preservation score: fraction of layers with strong identity mapping
    identity_preservation_score = sum(1 for dm in diagonal_means if dm > 0.5) / num_layers

    # Residual strength: average diagonal dominance
    residual_strength = np.mean(diagonal_means)

    return {
        "mean_jacobian_norm": float(mean_jacobian_norm),
        "mean_off_diagonal_norm": float(mean_off_diagonal_norm),
        "mean_diagonal_mean": float(mean_diagonal_mean),
        "identity_preservation_score": float(identity_preservation_score),
        "residual_strength": float(residual_strength),
    }


def plot_jacobian_metrics(layer_jacobians: Dict[int, Dict[str, float]], output_dir: Path):
    """
    Plot Jacobian metrics across layers.

    Creates multiple plots:
    1. Jacobian norm (deviation from identity)
    2. Off-diagonal norm (non-identity components)
    3. Diagonal mean (strength of identity mapping)
    4. Spectral norm

    Args:
        layer_jacobians: Dictionary of layer index to Jacobian metrics
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = sorted(layer_jacobians.keys())

    # Extract metrics
    jacobian_norms = [layer_jacobians[l]["jacobian_norm"] for l in layers]
    off_diagonal_norms = [layer_jacobians[l]["off_diagonal_norm"] for l in layers]
    diagonal_means = [layer_jacobians[l]["diagonal_mean"] for l in layers]
    spectral_norms = [layer_jacobians[l]["spectral_norm"] for l in layers]

    # Plot 1: Jacobian norm (deviation from identity)
    plt.figure(figsize=(10, 6))
    plt.plot(layers, jacobian_norms, marker="o", linewidth=2, markersize=6)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("||J - I|| (Frobenius Norm)", fontsize=12)
    plt.title("Jacobian Deviation from Identity Across Layers", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "jacobian_norm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Off-diagonal norm
    plt.figure(figsize=(10, 6))
    plt.plot(layers, off_diagonal_norms, marker="o", linewidth=2, markersize=6, color="orange")
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("||J_off_diag|| (Frobenius Norm)", fontsize=12)
    plt.title("Off-Diagonal Jacobian Norm Across Layers", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "off_diagonal_norm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Diagonal mean (identity strength)
    plt.figure(figsize=(10, 6))
    plt.plot(layers, diagonal_means, marker="o", linewidth=2, markersize=6, color="green")
    plt.axhline(y=1.0, color="r", linestyle="--", label="Perfect Identity")
    plt.axhline(y=0.5, color="orange", linestyle="--", label="Threshold")
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Mean Diagonal Value", fontsize=12)
    plt.title("Identity Mapping Strength Across Layers", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "diagonal_mean.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 4: Spectral norm
    plt.figure(figsize=(10, 6))
    plt.plot(layers, spectral_norms, marker="o", linewidth=2, markersize=6, color="purple")
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Spectral Norm (Max Singular Value)", fontsize=12)
    plt.title("Spectral Norm Across Layers", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "spectral_norm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 5: Combined metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Jacobian Metrics Across Layers", fontsize=16, fontweight="bold")

    axes[0, 0].plot(layers, jacobian_norms, marker="o", linewidth=2, markersize=6)
    axes[0, 0].set_ylabel("||J - I||", fontsize=11)
    axes[0, 0].set_title("Deviation from Identity", fontsize=12, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(layers, off_diagonal_norms, marker="o", linewidth=2, markersize=6, color="orange")
    axes[0, 1].set_ylabel("||J_off_diag||", fontsize=11)
    axes[0, 1].set_title("Off-Diagonal Norm", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(layers, diagonal_means, marker="o", linewidth=2, markersize=6, color="green")
    axes[1, 0].axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Identity")
    axes[1, 0].set_xlabel("Layer Index", fontsize=11)
    axes[1, 0].set_ylabel("Mean Diagonal", fontsize=11)
    axes[1, 0].set_title("Identity Strength", fontsize=12, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(layers, spectral_norms, marker="o", linewidth=2, markersize=6, color="purple")
    axes[1, 1].set_xlabel("Layer Index", fontsize=11)
    axes[1, 1].set_ylabel("Spectral Norm", fontsize=11)
    axes[1, 1].set_title("Spectral Norm", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "jacobian_metrics_combined.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compute Jacobian matrices for residual identity mapping analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--output_dir", type=str, default="./jacobian_residual_mapping_results", help="Directory to save results"
    )
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to use for evaluation")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument(
        "--method",
        type=str,
        default="rowwise",
        choices=["rowwise", "elementwise", "both"],
        help="Method to compute Jacobian: 'rowwise' (default), 'elementwise', or 'both'",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Jacobian Analysis for Residual Identity Mapping")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer, is_moe = load_model_and_tokenizer(args.model_path, args.device)

    num_layers = len(model.model.layers)
    print(f"Number of layers: {num_layers}")

    if is_moe:
        print("Warning: Jacobian analysis for MoE models may be less interpretable")

    # Create sample data
    sample_data = create_sample_data(tokenizer, args.num_samples, args.seq_length)

    # Compute Jacobian metrics for all layers
    layer_jacobians = compute_all_layer_jacobians(
        model, sample_data["input_ids"], sample_data["attention_mask"], args.device, num_layers, method=args.method
    )

    # Compute global metrics
    global_metrics = compute_global_jacobian_metrics(layer_jacobians, num_layers)

    # Print results
    print("\n" + "=" * 60)
    print("Global Jacobian Metrics")
    print("=" * 60)
    print(f"Mean Jacobian Norm (||J - I||): {global_metrics['mean_jacobian_norm']:.4f}")
    print(f"Mean Off-Diagonal Norm: {global_metrics['mean_off_diagonal_norm']:.4f}")
    print(f"Mean Diagonal Value: {global_metrics['mean_diagonal_mean']:.4f}")
    print(f"Identity Preservation Score: {global_metrics['identity_preservation_score']:.4f}")
    print(f"Residual Strength: {global_metrics['residual_strength']:.4f}")

    print("\n" + "=" * 60)
    print("Layer-wise Jacobian Metrics (Row-wise)")
    print("=" * 60)
    print(f"{'Layer':<8} {'||J-I||':<12} {'Off-Diag':<12} {'Diag Mean':<12} {'Spectral':<12}")
    print("-" * 60)
    for layer_idx in sorted(layer_jacobians.keys()):
        m = layer_jacobians[layer_idx]
        print(
            f"{layer_idx:<8} {m['jacobian_norm']:<12.4f} {m['off_diagonal_norm']:<12.4f} "
            f"{m['diagonal_mean']:<12.4f} {m['spectral_norm']:<12.4f}"
        )

    # Print element-wise metrics if computed
    if args.method in ["elementwise", "both"]:
        print("\n" + "=" * 60)
        print("Layer-wise Jacobian Metrics (Element-wise)")
        print("=" * 60)
        print(f"{'Layer':<8} {'||J-I||':<12} {'Off-Diag':<12} {'Diag Mean':<12} {'Spectral':<12}")
        print("-" * 60)
        for layer_idx in sorted(layer_jacobians.keys()):
            m = layer_jacobians[layer_idx]
            if "jacobian_norm_elementwise" in m:
                print(
                    f"{layer_idx:<8} {m['jacobian_norm_elementwise']:<12.4f} "
                    f"{m['off_diagonal_norm_elementwise']:<12.4f} "
                    f"{m['diagonal_mean_elementwise']:<12.4f} "
                    f"{m['spectral_norm_elementwise']:<12.4f}"
                )

    # Save results
    results = {
        "num_layers": num_layers,
        "global_metrics": global_metrics,
        "layer_jacobians": layer_jacobians,
    }

    output_file = output_dir / "jacobian_residual_mapping.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Save as CSV
    csv_file = output_dir / "jacobian_residual_mapping.csv"
    with open(csv_file, "w") as f:
        # Write header
        header = "layer,jacobian_norm,off_diagonal_norm,diagonal_mean,diagonal_std,"
        header += "frobenius_norm,spectral_norm"
        if args.method in ["elementwise", "both"]:
            header += ",jacobian_norm_elementwise,off_diagonal_norm_elementwise,"
            header += "diagonal_mean_elementwise,diagonal_std_elementwise,"
            header += "frobenius_norm_elementwise,spectral_norm_elementwise"
        header += "\n"
        f.write(header)

        # Write data
        for layer_idx in sorted(layer_jacobians.keys()):
            m = layer_jacobians[layer_idx]
            row = (
                f"{layer_idx},{m['jacobian_norm']},{m['off_diagonal_norm']},"
                f"{m['diagonal_mean']},{m['diagonal_std']},{m['frobenius_norm']},"
                f"{m['spectral_norm']}"
            )
            if args.method in ["elementwise", "both"]:
                if "jacobian_norm_elementwise" in m:
                    row += (
                        f",{m['jacobian_norm_elementwise']},"
                        f"{m['off_diagonal_norm_elementwise']},"
                        f"{m['diagonal_mean_elementwise']},"
                        f"{m['diagonal_std_elementwise']},"
                        f"{m['frobenius_norm_elementwise']},"
                        f"{m['spectral_norm_elementwise']}"
                    )
            row += "\n"
            f.write(row)
    print(f"CSV saved to: {csv_file}")

    # Generate plots
    if args.plot:
        plot_jacobian_metrics(layer_jacobians, output_dir)

    print("\n" + "=" * 60)
    print("Jacobian Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
