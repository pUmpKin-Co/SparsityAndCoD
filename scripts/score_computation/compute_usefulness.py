import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
            model_path, config=config, torch_dtype=torch.bfloat16, device_map=device
        )
    else:
        print("Loaded as standard (dense) model")
        model = CustomOlmo2ForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=torch.bfloat16, device_map=device
        )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, is_moe


def create_sample_data(tokenizer, num_samples: int = 100, seq_length: int = 512) -> Dict:
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


def compute_baseline_loss(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: str) -> float:
    """Compute baseline loss with all layers intact."""
    print("Computing baseline loss...")

    model.eval()
    batch_size = input_ids.shape[0]
    micro_batch_size = min(2, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    total_loss = 0.0

    with torch.no_grad():
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)
            micro_labels = input_ids[start_idx:end_idx].to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(
                    input_ids=micro_input_ids,
                    attention_mask=micro_attention_mask,
                    labels=micro_labels,
                    use_cache=False,
                    return_dict=True,
                )

            total_loss += outputs.loss.item() * (end_idx - start_idx)

            if device == "cuda":
                torch.cuda.empty_cache()

    return total_loss / batch_size


def collect_layer_input_output(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: str, layer_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect input and output data for a specific layer.

    Returns:
        inputs: Tensor of shape [N, D] where N is total tokens, D is hidden dimension
        outputs: Tensor of shape [N, D]
    """
    model.eval()
    batch_size = input_ids.shape[0]
    micro_batch_size = min(2, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    all_inputs = []
    all_outputs = []

    with torch.no_grad():
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(
                    input_ids=micro_input_ids,
                    attention_mask=micro_attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )

            # Get input (previous layer output) and output (current layer output)
            layer_input = outputs.hidden_states[layer_idx]  # Shape: [batch, seq_len, d_model]
            layer_output = outputs.hidden_states[layer_idx + 1]

            # Handle tuples (decoder layers return tuples)
            if isinstance(layer_input, tuple):
                layer_input = layer_input[0] if layer_input and len(layer_input) > 0 else None
            if isinstance(layer_output, tuple):
                layer_output = layer_output[0] if layer_output and len(layer_output) > 0 else None

            if layer_input is not None and layer_output is not None:
                # Flatten: [batch, seq_len, d_model] -> [batch*seq_len, d_model]
                all_inputs.append(layer_input.flatten(0, 1).float())
                all_outputs.append(layer_output.flatten(0, 1).float())

            if device == "cuda":
                torch.cuda.empty_cache()

    # Concatenate all micro-batches
    inputs = torch.cat(all_inputs, dim=0)  # [N, D]
    outputs = torch.cat(all_outputs, dim=0)  # [N, D]

    print(
        f"Layer {layer_idx}: Collected {inputs.shape[0]} samples, input_dim={inputs.shape[1]}, output_dim={outputs.shape[1]}"
    )

    return inputs, outputs


def fit_linear_mapping(inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit linear mapping outputs = W * inputs + b using least squares.

    Returns:
        W: Weight matrix of shape [D, D]
        b: Bias vector of shape [D]
    """
    # Convert to float32 for numerical stability in lstsq
    inputs = inputs.float()
    outputs = outputs.float()

    # Add bias term: [N, D] -> [N, D+1]
    inputs_with_bias = torch.cat(
        [inputs, torch.ones(inputs.shape[0], 1, device=inputs.device, dtype=inputs.dtype)], dim=1
    )

    # Solve least squares: minimize ||outputs - inputs_with_bias @ params||^2
    # params = [W; b] of shape [D+1, D]
    params = torch.linalg.lstsq(inputs_with_bias, outputs).solution

    # Split into W and b
    W = params[:-1, :]  # [D, D]
    b = params[-1, :]  # [D]

    return W, b


class LinearLayerWrapper(nn.Module):
    """Wrapper for nn.Linear that accepts attention_mask and other decoder layer arguments."""

    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.linear = linear_layer

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """
        Forward pass that accepts decoder layer arguments.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, d_model]
            attention_mask: Attention mask (ignored)
            **kwargs: Other arguments (ignored)

        Returns:
            Tuple of (hidden_states, None) to match decoder layer output format
        """
        # Ensure hidden_states is a tensor (not a tuple)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        # Apply linear transformation
        output = self.linear(hidden_states)

        # Return tuple to match decoder layer format: (hidden_states, attn_weights)
        return (output, None)


def create_linear_layer(W: torch.Tensor, b: torch.Tensor, device: str) -> nn.Module:
    """Create a linear layer from fitted parameters."""
    d_model = W.shape[0]
    # Create linear layer without specifying dtype initially
    linear_layer = nn.Linear(d_model, d_model, bias=True).to(device)

    with torch.no_grad():
        # W and b are float32, convert to bfloat16 and copy
        linear_layer.weight.data = W.to(torch.bfloat16)
        linear_layer.bias.data = b.to(torch.bfloat16)

    linear_layer.eval()
    # Wrap in a layer that accepts attention_mask
    return LinearLayerWrapper(linear_layer)


def compute_layer_usefulness_linear_approximation(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: str, baseline_loss: float, num_layers: int
) -> Dict[int, Dict[str, float]]:
    """
    Compute layer usefulness by replacing each layer with a linear approximation.

    For each layer l:
    1. Collect input-output data from the layer
    2. Fit linear mapping W*x + b
    3. Replace layer with this linear mapping
    4. Compute loss increase
    5. Restore original layer

    Global usefulness score = ratio of layers with >10% loss increase

    Returns:
        Dictionary mapping layer index to metrics:
        - 'loss_increase': Loss increase when replaced with linear approximation
        - 'loss_ratio': Loss ratio (linear_loss / baseline_loss)
        - 'significant_increase': Whether loss increased by >10%
    """
    print("Computing layer usefulness via linear approximation...")

    layer_metrics = {}

    for layer_idx in tqdm(range(num_layers), desc="Computing layer usefulness"):
        model.eval()

        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()

        # Store original layer
        original_layer = model.model.layers[layer_idx]
        original_params = {name: param.data.clone() for name, param in original_layer.named_parameters()}

        # Collect input-output data for this layer
        inputs, outputs = collect_layer_input_output(model, input_ids, attention_mask, device, layer_idx)

        # Fit linear mapping
        W, b = fit_linear_mapping(inputs, outputs)

        # Create linear layer
        linear_layer = create_linear_layer(W, b, device)

        # Replace original layer with linear layer
        model.model.layers[layer_idx] = linear_layer

        # Verify the layer replacement with a small test forward pass
        with torch.no_grad():
            test_input = torch.randn(1, 10, W.shape[0], dtype=torch.bfloat16, device=device)
            test_output = linear_layer(test_input)
            print(
                f"Layer {layer_idx}: Test forward pass successful, output shape: {test_output[0].shape if isinstance(test_output, tuple) else test_output.shape}"
            )

        # Clear all caches after replacement
        if device == "cuda":
            torch.cuda.empty_cache()

        # Verify the layer replacement
        print(f"Layer {layer_idx}: Replaced with linear layer")
        print(f"  Input dim: {W.shape[0]}, Output dim: {W.shape[1]}")

        # Compute loss with linear approximation
        batch_size = input_ids.shape[0]
        micro_batch_size = min(2, batch_size)
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

        linear_loss_accum = 0.0

        with torch.no_grad():
            for mb_idx in range(num_micro_batches):
                start_idx = mb_idx * micro_batch_size
                end_idx = min(start_idx + micro_batch_size, batch_size)

                micro_input_ids = input_ids[start_idx:end_idx].to(device)
                micro_attention_mask = attention_mask[start_idx:end_idx].to(device)
                micro_labels = input_ids[start_idx:end_idx].to(device)

                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=micro_input_ids,
                        attention_mask=micro_attention_mask,
                        labels=micro_labels,
                        use_cache=False,
                        output_hidden_states=False,  # Don't collect hidden_states to avoid format issues
                        return_dict=True,
                    )

                linear_loss_accum += outputs.loss.item() * (end_idx - start_idx)

                if device == "cuda":
                    torch.cuda.empty_cache()

        # Compute average loss
        linear_loss = linear_loss_accum / batch_size

        # Restore original layer
        model.model.layers[layer_idx] = original_layer

        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()

        # Compute metrics
        loss_increase = linear_loss - baseline_loss
        loss_ratio = linear_loss / baseline_loss
        significant_increase = loss_ratio > 1.1  # >10% increase

        layer_metrics[layer_idx] = {
            "loss_increase": loss_increase,
            "loss_ratio": loss_ratio,
            "linear_loss": linear_loss,
            "significant_increase": significant_increase,
        }

    return layer_metrics


def compute_global_usefulness_score(layer_metrics: Dict[int, Dict[str, float]], num_layers: int) -> Dict[str, float]:
    """
    Compute global usefulness score.

    Global usefulness score = ratio of layers with >10% loss increase.

    Returns:
        Dictionary with global metrics:
        - 'global_usefulness_score': Ratio of layers with >10% loss increase
        - 'num_significant_layers': Number of layers with >10% loss increase
        - 'mean_loss_increase': Mean loss increase across all layers
        - 'mean_loss_ratio': Mean loss ratio across all layers
    """
    significant_count = sum(1 for metrics in layer_metrics.values() if metrics["significant_increase"])
    global_usefulness_score = significant_count / num_layers if num_layers > 0 else 0.0

    mean_loss_increase = np.mean([m["loss_increase"] for m in layer_metrics.values()])
    mean_loss_ratio = np.mean([m["loss_ratio"] for m in layer_metrics.values()])

    return {
        "global_usefulness_score": global_usefulness_score,
        "num_significant_layers": significant_count,
        "total_layers": num_layers,
        "mean_loss_increase": mean_loss_increase,
        "mean_loss_ratio": mean_loss_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute layer usefulness via linear approximation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--output_dir", type=str, default="./layer_usefulness_linear_results", help="Directory to save results"
    )
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to use for evaluation")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Layer Usefulness via Linear Approximation")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer, is_moe = load_model_and_tokenizer(args.model_path, args.device)

    num_layers = len(model.model.layers)
    print(f"Number of layers: {num_layers}")

    if is_moe:
        print("Warning: This method is designed for dense models. MoE models may not work correctly.")

    # Create sample data
    sample_data = create_sample_data(tokenizer, args.num_samples, args.seq_length)

    # Compute baseline loss
    baseline_loss = compute_baseline_loss(model, sample_data["input_ids"], sample_data["attention_mask"], args.device)
    print(f"\nBaseline loss: {baseline_loss:.4f}")

    # Compute layer usefulness via linear approximation
    layer_metrics = compute_layer_usefulness_linear_approximation(
        model, sample_data["input_ids"], sample_data["attention_mask"], args.device, baseline_loss, num_layers
    )

    # Compute global usefulness score
    global_metrics = compute_global_usefulness_score(layer_metrics, num_layers)

    # Print results
    print("\n" + "=" * 60)
    print("Global Usefulness Score")
    print("=" * 60)
    print(f"Global Usefulness Score: {global_metrics['global_usefulness_score']:.4f}")
    print(
        f"Layers with >10% loss increase: {global_metrics['num_significant_layers']}/{global_metrics['total_layers']}"
    )
    print(f"Mean Loss Increase: {global_metrics['mean_loss_increase']:.4f}")
    print(f"Mean Loss Ratio: {global_metrics['mean_loss_ratio']:.4f}")

    print("\n" + "=" * 60)
    print("Layer-wise Results")
    print("=" * 60)
    print(f"{'Layer':<10} {'Loss Increase':<15} {'Loss Ratio':<15} {'Significant':<15}")
    print("-" * 60)
    for layer_idx in sorted(layer_metrics.keys()):
        metrics = layer_metrics[layer_idx]
        sig_str = "Yes" if metrics["significant_increase"] else "No"
        print(f"{layer_idx:<10} {metrics['loss_increase']:<15.4f} {metrics['loss_ratio']:<15.4f} {sig_str:<15}")

    # Save results
    results = {
        "baseline_loss": baseline_loss,
        "num_layers": num_layers,
        "global_metrics": global_metrics,
        "layer_metrics": layer_metrics,
    }

    output_file = output_dir / "layer_usefulness_linear.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Also save as CSV for easy viewing
    csv_file = output_dir / "layer_usefulness_linear.csv"
    with open(csv_file, "w") as f:
        f.write("layer,loss_increase,loss_ratio,linear_loss,significant_increase\n")
        for layer_idx in sorted(layer_metrics.keys()):
            m = layer_metrics[layer_idx]
            f.write(
                f"{layer_idx},{m['loss_increase']},{m['loss_ratio']},{m['linear_loss']},{m['significant_increase']}\n"
            )
    print(f"CSV saved to: {csv_file}")

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
