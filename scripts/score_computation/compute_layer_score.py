import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

HAS_MATPLOTLIB = True

from transformers import AutoTokenizer

from olmo.custom_hf_model import CustomOlmo2ForCausalLM, CustomOlmo2MoEForCausalLM
from olmo.custom_hf_model_config import CustomOlmo2Config, CustomOlmo2MoEConfig
from olmo.data.memmap_dataset import MemMapDataset


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer, automatically detecting if it's a MoE model."""
    print(f"Loading model from {model_path}...")

    # Try to determine if it's a MoE model by checking the config
    config = CustomOlmo2Config.from_pretrained(model_path)

    # Check if this is a MoE model by looking at the moe_type parameter
    is_moe = hasattr(config, "model_type") and config.model_type == "customolmo2moe"

    if is_moe:
        print(f"Detected MoE model (type: {config.moe_type})")
        print(f"  Number of experts: {config.num_experts}")
        print(f"  Experts per token: {config.num_experts_per_tok}")
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

    # Use some sample texts
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


def compute_baseline_forward(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: str
) -> Tuple[torch.Tensor, List[torch.Tensor], float]:
    """
    Compute baseline forward pass and collect all hidden states.

    Uses batch-wise processing to reduce memory usage and avoid CUDA OOM errors.

    Returns:
        - loss: Cross-entropy loss
        - all_hidden_states: List of hidden states for each layer
        - perplexity: Perplexity score
    """
    model.eval()

    batch_size = input_ids.shape[0]
    micro_batch_size = min(4, batch_size)  # Use small micro-batches to reduce memory
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    all_hidden_states_accum = []
    loss_accum = 0.0

    with torch.no_grad():
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            # Get micro-batch
            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)
            micro_labels = input_ids[start_idx:end_idx].to(device)

            # Forward pass
            outputs = model(
                input_ids=micro_input_ids,
                attention_mask=micro_attention_mask,
                labels=micro_labels,
                output_hidden_states=True,
                return_dict=True,
            )

            # Accumulate loss
            loss_accum += outputs.loss.item() * (end_idx - start_idx)

            # Process hidden states
            if mb_idx == 0:
                # First micro-batch: initialize hidden states list
                for h in outputs.hidden_states:
                    if isinstance(h, tuple):
                        h = h[0] if h else None
                    if h is not None:
                        all_hidden_states_accum.append(h.detach().cpu() if h.is_cuda else h.detach())
                    else:
                        all_hidden_states_accum.append(None)
            else:
                # Subsequent micro-batches: accumulate hidden states
                for i, h in enumerate(outputs.hidden_states):
                    if isinstance(h, tuple):
                        h = h[0] if h else None
                    if h is not None:
                        h_cpu = h.detach().cpu() if h.is_cuda else h.detach()
                        if all_hidden_states_accum[i] is None:
                            all_hidden_states_accum[i] = h_cpu
                        else:
                            # Concatenate along batch dimension
                            all_hidden_states_accum[i] = torch.cat([all_hidden_states_accum[i], h_cpu], dim=0)

            # Clean up
            del outputs, micro_input_ids, micro_attention_mask, micro_labels
            if device == "cuda":
                torch.cuda.empty_cache()

    # Compute average loss
    loss = loss_accum / batch_size
    perplexity = math.exp(loss)

    # Replace None values with zeros
    all_hidden_states = []
    for h in all_hidden_states_accum:
        if h is None:
            all_hidden_states.append(torch.zeros_like(all_hidden_states_accum[0]))
        else:
            all_hidden_states.append(h)

    return loss, all_hidden_states, perplexity


def compute_causal_effect_on_future(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    baseline_hidden_states: List[torch.Tensor],
    layer_to_skip: int,
) -> Dict[int, float]:
    """
    Compute causal effect by skipping a specific layer.

    For each future layer l > s, computes:
    ||(h_{l+1} - h_l) - (bar{h}_{l+1} - bar{h}_l)||_2 / ||h_{l+1} - h_l||_2

    Returns:
        Dictionary mapping future layer index to causal effect score
    """
    model.eval()

    # Set the layer to skip
    original_skip_ids = getattr(model.model.config, "skip_layer_ids", None)
    model.model.config.skip_layer_ids = [layer_to_skip]

    batch_size = input_ids.shape[0]
    micro_batch_size = min(4, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    modified_hidden_states_accum = []

    with torch.no_grad():
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            # Get micro-batch
            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)

            # Forward pass
            outputs = model(
                input_ids=micro_input_ids,
                attention_mask=micro_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # Process hidden states
            if mb_idx == 0:
                # First micro-batch: initialize
                for h in outputs.hidden_states:
                    if isinstance(h, tuple):
                        h = h[0] if h else None
                    if h is not None:
                        modified_hidden_states_accum.append(h.detach().cpu() if h.is_cuda else h.detach())
                    else:
                        modified_hidden_states_accum.append(None)
            else:
                # Subsequent micro-batches: accumulate
                for i, h in enumerate(outputs.hidden_states):
                    if isinstance(h, tuple):
                        h = h[0] if h else None
                    if h is not None:
                        h_cpu = h.detach().cpu() if h.is_cuda else h.detach()
                        if modified_hidden_states_accum[i] is None:
                            modified_hidden_states_accum[i] = h_cpu
                        else:
                            modified_hidden_states_accum[i] = torch.cat(
                                [modified_hidden_states_accum[i], h_cpu], dim=0
                            )

            # Clean up
            del outputs, micro_input_ids, micro_attention_mask
            if device == "cuda":
                torch.cuda.empty_cache()

    # Restore original skip configuration
    model.model.config.skip_layer_ids = original_skip_ids

    # Replace None values with zeros
    modified_hidden_states = []
    for h in modified_hidden_states_accum:
        if h is None:
            modified_hidden_states.append(torch.zeros_like(baseline_hidden_states[0]))
        else:
            modified_hidden_states.append(h)

    # Compute causal effect for each future layer
    causal_effects = {}

    # Use minimum length to avoid index out of bounds
    max_layer = min(len(baseline_hidden_states), len(modified_hidden_states)) - 1

    # For each future layer l > layer_to_skip
    for l in range(layer_to_skip + 1, max_layer):
        # Get baseline differences
        h_l = baseline_hidden_states[l]
        h_l_plus_1 = baseline_hidden_states[l + 1]
        baseline_diff = h_l_plus_1 - h_l
        baseline_norm = torch.norm(baseline_diff, p=2, dim=-1).mean().item()

        # Get modified differences
        bar_h_l = modified_hidden_states[l]
        bar_h_l_plus_1 = modified_hidden_states[l + 1]
        modified_diff = bar_h_l_plus_1 - bar_h_l

        # Compute relative change
        diff_diff = modified_diff - baseline_diff
        relative_change = torch.norm(diff_diff, p=2, dim=-1).mean().item()

        # Compute causal effect score
        if baseline_norm > 1e-8:
            causal_effect = relative_change / baseline_norm
        else:
            causal_effect = 0.0

        causal_effects[l] = causal_effect

    return causal_effects


def swap_layer_weights(model, layer1_idx: int, layer2_idx: int):
    """Swap the weights of two layers."""
    layer1 = model.model.layers[layer1_idx]
    layer2 = model.model.layers[layer2_idx]

    # Swap all parameters between the two layers
    for (name1, param1), (name2, param2) in zip(layer1.named_parameters(), layer2.named_parameters()):
        # Swap parameters
        temp = param1.data.clone()
        param1.data.copy_(param2.data)
        param2.data.copy_(temp)


def compute_permutation_score(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    baseline_loss: float,
    layer1_idx: int,
    layer2_idx: int,
) -> float:
    """
    Compute permutation score by swapping two layers.

    Formula: P_score(l1, l2) = |L(M) - L(M_swap(l1, l2))| / |L(M)|

    This measures the normalized absolute change in loss when swapping two layers.

    Returns:
        Permutation score (non-negative, higher means layers are more important/different)
    """
    model.eval()

    # Store original weights
    layer1 = model.model.layers[layer1_idx]
    layer2 = model.model.layers[layer2_idx]

    original_params_1 = {name: param.data.clone() for name, param in layer1.named_parameters()}
    original_params_2 = {name: param.data.clone() for name, param in layer2.named_parameters()}

    # Swap layers
    swap_layer_weights(model, layer1_idx, layer2_idx)

    # Compute loss with swapped layers using batch-wise processing
    batch_size = input_ids.shape[0]
    micro_batch_size = min(4, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    swapped_loss_accum = 0.0

    with torch.no_grad():
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            # Get micro-batch
            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)
            micro_labels = input_ids[start_idx:end_idx].to(device)

            # Forward pass
            outputs = model(
                input_ids=micro_input_ids, attention_mask=micro_attention_mask, labels=micro_labels, return_dict=True
            )
            swapped_loss_accum += outputs.loss.item() * (end_idx - start_idx)

            # Clean up
            del outputs, micro_input_ids, micro_attention_mask, micro_labels
            if device == "cuda":
                torch.cuda.empty_cache()

    # Compute average loss
    swapped_loss = swapped_loss_accum / batch_size

    # Restore original weights
    for name, param in layer1.named_parameters():
        param.data.copy_(original_params_1[name])
    for name, param in layer2.named_parameters():
        param.data.copy_(original_params_2[name])

    # Compute permutation score
    if baseline_loss > 1e-8:
        permutation_score = abs(baseline_loss - swapped_loss) / baseline_loss
    else:
        permutation_score = 0.0

    return permutation_score


def compute_all_causal_effects(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: str, num_layers: int
) -> np.ndarray:
    """
    Compute causal effects for all layer pairs.

    Returns:
        Matrix where causal_effects[s, l] is the effect of skipping layer s on layer l
    """
    print("Computing causal effects...")

    # Get baseline
    _, baseline_hidden_states, _ = compute_baseline_forward(model, input_ids, attention_mask, device)

    causal_effect_matrix = np.zeros((num_layers, num_layers))

    for s in tqdm(range(num_layers), desc="Computing causal effects"):
        causal_effects = compute_causal_effect_on_future(
            model, input_ids, attention_mask, device, baseline_hidden_states, s
        )

        for l, effect in causal_effects.items():
            causal_effect_matrix[s, l] = effect

    return causal_effect_matrix


def compute_all_permutation_scores(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    num_layers: int,
    layer_pairs: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], float]:
    """
    Compute permutation scores for specified layer pairs.

    Returns:
        Dictionary mapping (l1, l2) to permutation score
    """
    print("Computing permutation scores...")

    # Get baseline loss
    baseline_loss, _, _ = compute_baseline_forward(model, input_ids, attention_mask, device)

    permutation_scores = {}

    for l1, l2 in tqdm(layer_pairs, desc="Computing permutation scores"):
        score = compute_permutation_score(model, input_ids, attention_mask, device, baseline_loss, l1, l2)
        permutation_scores[(l1, l2)] = score

    return permutation_scores


def compute_global_permutation_score(
    permutation_scores: Dict[Tuple[int, int], float], num_layers: int, method: str = "mean"
) -> Dict[str, float]:
    """
    Compute a scalar global permutation score comparable across different networks.

    Multiple normalization methods to handle different network depths:

    Args:
        permutation_scores: Dictionary of (l1, l2) -> permutation score
        num_layers: Total number of layers in the network
        method: Normalization method ('mean', 'normalized_by_depth', 'weighted')

    Returns:
        Dictionary with different global scores:
        - 'global_score_mean': Simple mean of all permutation scores
        - 'global_score_normalized': Normalized by maximum possible pairs
        - 'global_score_weighted': Weighted by layer position (earlier layers weighted more)
        - 'layer_avg_scores': Average score per layer
    """
    if not permutation_scores:
        return {
            "global_score_mean": 0.0,
            "global_score_normalized": 0.0,
            "global_score_weighted": 0.0,
            "layer_avg_scores": [],
        }

    # Extract all scores
    scores = list(permutation_scores.values())

    # Method 1: Simple mean (unnormalized)
    global_score_mean = np.mean(scores)

    # Method 2: Normalized by maximum possible pairs
    # Maximum possible pairs = num_layers * (num_layers - 1) / 2
    max_pairs = num_layers * (num_layers - 1) / 2
    global_score_normalized = np.sum(scores) / max_pairs if max_pairs > 0 else 0.0

    # Method 3: Weighted by layer position
    # Earlier layers (closer to input) get higher weights
    # This accounts for depth: deeper networks naturally have more pairs
    layer_counts = np.zeros(num_layers)
    layer_sums = np.zeros(num_layers)
    layer_weights = np.zeros(num_layers)

    # Assign weights based on layer position (exponential decay from input to output)
    # Layer 0 has weight 1.0, layer N-1 has weight decay_factor^(N-1)
    decay_factor = 0.9
    for i in range(num_layers):
        layer_weights[i] = decay_factor**i

    for (l1, l2), score in permutation_scores.items():
        layer_counts[l1] += 1
        layer_sums[l1] += score * layer_weights[l1]
        layer_counts[l2] += 1
        layer_sums[l2] += score * layer_weights[l2]

    # Compute weighted average
    layer_avg_scores = []
    for i in range(num_layers):
        if layer_counts[i] > 0:
            avg = layer_sums[i] / layer_counts[i]
            layer_avg_scores.append(avg)
        else:
            layer_avg_scores.append(0.0)

    # Global weighted score (average of weighted layer scores)
    global_score_weighted = np.mean(layer_avg_scores) if layer_avg_scores else 0.0

    return {
        "global_score_mean": global_score_mean,
        "global_score_normalized": global_score_normalized,
        "global_score_weighted": global_score_weighted,
        "layer_avg_scores": layer_avg_scores,
    }


def compute_layer_similarity(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: str, num_layers: int
) -> np.ndarray:
    """
    Compute layer-wise similarity matrix using cosine similarity of layer activations/outputs.

    This measures how similar each layer's output is to every other layer's output
    based on their activations on the same input. High similarity indicates redundant
    computations.

    Args:
        model: The model to evaluate
        input_ids: Input token IDs
        attention_mask: Attention mask
        device: Device to run on
        num_layers: Number of layers in the model

    Returns:
        Similarity matrix where similarity[i, j] is the cosine similarity between
        activations of layer i and layer j. Diagonal is 1.0 (self-similarity).
    """
    print("Computing layer similarity matrix (based on activations)...")

    model.eval()

    batch_size = input_ids.shape[0]
    micro_batch_size = min(8, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    layer_activations_accum = []

    with torch.no_grad():
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            # Get micro-batch
            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)

            # Forward pass
            outputs = model(
                input_ids=micro_input_ids,
                attention_mask=micro_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # Get activations for each layer (excluding embeddings)
            for layer_idx in range(num_layers):
                activation = outputs.hidden_states[layer_idx + 1]  # +1 to skip embeddings

                # Handle tuples (can occur when layers are skipped)
                if isinstance(activation, tuple):
                    activation = activation[0] if activation else None

                if activation is None:
                    continue

                # Flatten activations: [batch, seq_len, hidden_dim] -> [batch * seq_len, hidden_dim]
                mb_batch_size, seq_len, hidden_dim = activation.shape
                flattened = activation.reshape(-1, hidden_dim)

                # Compute mean activation across all tokens and batch
                mean_activation = flattened.mean(dim=0)

                if mb_idx == 0:
                    layer_activations_accum.append(mean_activation.cpu())
                else:
                    # Accumulate activations (weighted average)
                    layer_activations_accum[layer_idx] = (
                        layer_activations_accum[layer_idx] * mb_idx + mean_activation.cpu()
                    ) / (mb_idx + 1)

            # Clean up
            del outputs, micro_input_ids, micro_attention_mask
            if device == "cuda":
                torch.cuda.empty_cache()

    # Compute pairwise cosine similarity
    similarity_matrix = np.zeros((num_layers, num_layers))

    for i in tqdm(range(num_layers), desc="Computing activation similarities"):
        for j in range(i, num_layers):
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                layer_activations_accum[i].unsqueeze(0), layer_activations_accum[j].unsqueeze(0)
            ).item()

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric

    return similarity_matrix


def plot_causal_effects(causal_effect_matrix: np.ndarray, output_path: Path):
    """Plot causal effect matrix as a heatmap."""
    if not HAS_MATPLOTLIB:
        print("Skipping causal effects plot - matplotlib not available")
        return

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        causal_effect_matrix,
        cmap="viridis",
        cbar_kws={"label": "Causal Effect Score"},
        xticklabels=range(causal_effect_matrix.shape[1]),
        yticklabels=range(causal_effect_matrix.shape[0]),
    )
    plt.xlabel("Future Layer (l)")
    plt.ylabel("Skipped Layer (s)")
    plt.title("Causal Effect on Future Computations\n(Skipping layer s on layer l)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved causal effects plot to {output_path}")


def plot_permutation_scores(permutation_scores: Dict[Tuple[int, int], float], num_layers: int, output_path: Path):
    """Plot permutation scores as an upper triangular matrix."""
    if not HAS_MATPLOTLIB:
        print("Skipping permutation scores plot - matplotlib not available")
        return

    # Create matrix
    score_matrix = np.zeros((num_layers, num_layers))
    score_matrix[:] = np.nan  # Fill with NaN for lower triangle

    for (l1, l2), score in permutation_scores.items():
        score_matrix[l1, l2] = score

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        score_matrix,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Permutation Score"},
        xticklabels=range(num_layers),
        yticklabels=range(num_layers),
        mask=np.tril(np.ones_like(score_matrix, dtype=bool)),
    )
    plt.xlabel("Layer 2")
    plt.ylabel("Layer 1")
    plt.title("Layer Permutation Scores\n(Higher = More Permutable/Independent)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved permutation scores plot to {output_path}")


def plot_layer_importance_summary(
    causal_effect_matrix: np.ndarray,
    permutation_scores: Dict[Tuple[int, int], float],
    num_layers: int,
    output_path: Path,
):
    """Create a summary plot showing both metrics."""
    if not HAS_MATPLOTLIB:
        print("Skipping summary plot - matplotlib not available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Causal effect: average effect on future layers
    avg_causal_effect = np.mean(causal_effect_matrix, axis=1)

    axes[0].bar(range(num_layers), avg_causal_effect)
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Average Causal Effect on Future Layers")
    axes[0].set_title("Layer Importance: Causal Effect")
    axes[0].grid(True, alpha=0.3)

    # Permutation score: average with other layers
    avg_permutation_score = np.zeros(num_layers)
    count = np.zeros(num_layers)

    for (l1, l2), score in permutation_scores.items():
        avg_permutation_score[l1] += score
        count[l1] += 1

    # Avoid division by zero
    avg_permutation_score[count > 0] /= count[count > 0]

    axes[1].bar(range(num_layers), avg_permutation_score)
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("Average Permutation Score")
    axes[1].set_title("Layer Importance: Permutation Score")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot to {output_path}")


def plot_layer_similarity_matrix(
    similarity_matrix: np.ndarray, output_path: Path, title: str = "Layer Similarity Matrix"
):
    """Plot layer similarity matrix as a heatmap."""
    if not HAS_MATPLOTLIB:
        print("Skipping similarity plot - matplotlib not available")
        return

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        cbar_kws={"label": "Cosine Similarity"},
        xticklabels=range(similarity_matrix.shape[1]),
        yticklabels=range(similarity_matrix.shape[0]),
        square=True,
    )
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved similarity plot to {output_path}")


def compute_moe_expert_utilization(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: str, num_layers: int
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute expert utilization statistics for MoE models.

    This function records which experts are used during forward pass and computes:
    - Expert selection frequency: How often each expert is selected
    - Load balancing score: How evenly tokens are distributed across experts
    - Expert importance: How much each expert contributes to the loss

    Args:
        model: The MoE model to evaluate
        input_ids: Input token IDs
        attention_mask: Attention mask
        device: Device to run on
        num_layers: Number of layers in the model

    Returns:
        Dictionary mapping layer index to expert statistics:
        - 'expert_selections': Count of how many times each expert was selected
        - 'expert_utilization': Percentage of tokens each expert processed
        - 'load_balance_score': Load balancing score (higher = more balanced)
    """
    print("Computing MoE expert utilization...")

    model.eval()

    # Hook to capture expert selections
    expert_selections = {layer_idx: [] for layer_idx in range(num_layers)}

    def create_expert_hook(layer_idx):
        def hook(module, input, output):
            # Try to extract expert selection information from the MoE module
            # The MoE module should have routing information
            if hasattr(module, "moe"):
                moe = module.moe
                # Try to get expert assignments from the router
                if hasattr(moe, "router"):
                    with torch.no_grad():
                        # Get routing decisions
                        x = input[0]
                        routing_output = moe.router(x)
                        # Get top-k expert indices
                        if hasattr(routing_output, "indices"):
                            expert_indices = routing_output.indices  # [batch, seq, top_k]
                            expert_selections[layer_idx].append(expert_indices.cpu())
                        elif isinstance(routing_output, tuple) and len(routing_output) > 0:
                            # routing_output might be (dispatch_idx, combine_weights, expert_mask)
                            if hasattr(routing_output[0], "expert_idx"):
                                expert_indices = routing_output[0].expert_idx
                                expert_selections[layer_idx].append(expert_indices.cpu())

        return hook

    # Register hooks for all layers
    hooks = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        hook = layer.register_forward_hook(create_expert_hook(layer_idx))
        hooks.append(hook)

    # Forward pass to collect expert selections using batch-wise processing
    batch_size = input_ids.shape[0]
    micro_batch_size = min(4, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    with torch.no_grad():
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            # Get micro-batch
            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)

            # Forward pass
            outputs = model(input_ids=micro_input_ids, attention_mask=micro_attention_mask, return_dict=True)

            # Clean up
            del outputs, micro_input_ids, micro_attention_mask
            if device == "cuda":
                torch.cuda.empty_cache()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Process expert selections
    expert_stats = {}

    for layer_idx in range(num_layers):
        selections = expert_selections[layer_idx]

        if not selections:
            # No expert selections recorded, create empty stats
            num_experts = model.config.num_experts
            expert_stats[layer_idx] = {
                "expert_selections": np.zeros(num_experts),
                "expert_utilization": np.zeros(num_experts),
                "load_balance_score": 0.0,
            }
            continue

        # Concatenate all selections
        all_selections = torch.cat(selections, dim=0)  # [total_tokens, top_k]

        # Count selections per expert
        num_experts = model.config.num_experts
        expert_counts = torch.zeros(num_experts, dtype=torch.long)

        for expert_idx in all_selections.flatten():
            if 0 <= expert_idx < num_experts:
                expert_counts[expert_idx] += 1

        total_selections = expert_counts.sum()

        # Compute utilization (percentage)
        if total_selections > 0:
            expert_utilization = (expert_counts.float() / total_selections).numpy()
        else:
            expert_utilization = np.zeros(num_experts)

        # Compute load balancing score
        # Using coefficient of variation: 1 - (std / mean)
        if expert_utilization.mean() > 0:
            cv = expert_utilization.std() / expert_utilization.mean()
            load_balance_score = max(0.0, 1.0 - cv)
        else:
            load_balance_score = 0.0

        expert_stats[layer_idx] = {
            "expert_selections": expert_counts.numpy(),
            "expert_utilization": expert_utilization,
            "load_balance_score": load_balance_score,
        }

    return expert_stats


def compute_moe_expert_importance(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: str, baseline_loss: float, num_layers: int
) -> Dict[int, Dict[str, float]]:
    """
    Compute expert importance scores through expert ablation study.

    This function measures how much each expert contributes to model performance
    by removing each expert and measuring the performance impact.

    Args:
        model: The MoE model to evaluate
        input_ids: Input token IDs
        attention_mask: Attention mask
        device: Device to run on
        baseline_loss: Baseline loss with all experts
        num_layers: Number of layers in the model

    Returns:
        Dictionary mapping layer index to expert importance:
        - 'layer_importance': Overall importance of this layer's expert block
        - 'expert_importance': Importance of individual experts (if available)
        - 'loss_increase': How much loss increases when experts are ablated
    """
    print("Computing MoE expert importance (ablation study)...")

    model.eval()

    expert_importance = {}

    for layer_idx in tqdm(range(num_layers), desc="Computing expert importance"):
        model.eval()

        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()

        # Store original MoE weights
        layer = model.model.layers[layer_idx]
        original_moe = layer.moe

        # Zero out the MoE block (simulate removal)
        original_params = {}
        for name, param in original_moe.named_parameters():
            original_params[name] = param.data.clone()
            param.data.zero_()

        # Compute loss with MoE removed using batch-wise processing
        batch_size = input_ids.shape[0]
        micro_batch_size = min(4, batch_size)
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

        ablated_loss_accum = 0.0

        with torch.no_grad():
            for mb_idx in range(num_micro_batches):
                start_idx = mb_idx * micro_batch_size
                end_idx = min(start_idx + micro_batch_size, batch_size)

                # Get micro-batch
                micro_input_ids = input_ids[start_idx:end_idx].to(device)
                micro_attention_mask = attention_mask[start_idx:end_idx].to(device)
                micro_labels = input_ids[start_idx:end_idx].to(device)

                # Forward pass
                outputs = model(
                    input_ids=micro_input_ids,
                    attention_mask=micro_attention_mask,
                    labels=micro_labels,
                    return_dict=True,
                )
                ablated_loss_accum += outputs.loss.item() * (end_idx - start_idx)

                # Clean up
                del outputs, micro_input_ids, micro_attention_mask, micro_labels
                if device == "cuda":
                    torch.cuda.empty_cache()

        # Compute average loss
        ablated_loss = ablated_loss_accum / batch_size

        # Restore original weights
        for name, param in original_moe.named_parameters():
            param.data.copy_(original_params[name])

        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()

        # Compute loss increase
        loss_increase = ablated_loss - baseline_loss

        # Normalize loss increase
        normalized_loss_increase = min(loss_increase / (10 * baseline_loss), 1.0)

        expert_importance[layer_idx] = {"layer_importance": normalized_loss_increase, "loss_increase": loss_increase}

    return expert_importance


def compute_moe_global_metrics(
    expert_utilization: Dict[int, Dict[str, np.ndarray]],
    expert_importance: Dict[int, Dict[str, float]],
    num_layers: int,
) -> Dict[str, float]:
    """
    Compute global MoE metrics comparable across different models.

    Args:
        expert_utilization: Expert utilization statistics per layer
        expert_importance: Expert importance scores per layer
        num_layers: Number of layers

    Returns:
        Dictionary with global MoE metrics:
        - 'avg_load_balance': Average load balancing score across layers
        - 'avg_expert_utilization': Average expert utilization across layers
        - 'expert_importance_score': Average expert importance across layers
        - 'expert_diversity': How diverse expert usage is across layers
    """
    if not expert_utilization:
        return {
            "avg_load_balance": 0.0,
            "avg_expert_utilization": 0.0,
            "expert_importance_score": 0.0,
            "expert_diversity": 0.0,
        }

    # Average load balancing score
    load_balance_scores = [expert_utilization[l]["load_balance_score"] for l in range(num_layers)]
    avg_load_balance = np.mean(load_balance_scores)

    # Average expert utilization (mean utilization across all experts)
    all_utilizations = []
    for l in range(num_layers):
        all_utilizations.extend(expert_utilization[l]["expert_utilization"])
    avg_expert_utilization = np.mean(all_utilizations)

    # Average expert importance
    importance_scores = [expert_importance[l]["layer_importance"] for l in range(num_layers)]
    expert_importance_score = np.mean(importance_scores)

    # Expert diversity: entropy of expert selection patterns across layers
    # Higher diversity means different experts are used in different layers
    num_experts = len(expert_utilization[0]["expert_utilization"])
    layer_patterns = []

    for l in range(num_layers):
        pattern = expert_utilization[l]["expert_utilization"]
        layer_patterns.append(pattern)

    # Compute pairwise diversity
    diversity_scores = []
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            # Use 1 - cosine similarity as diversity measure
            sim = np.dot(layer_patterns[i], layer_patterns[j]) / (
                np.linalg.norm(layer_patterns[i]) * np.linalg.norm(layer_patterns[j]) + 1e-8
            )
            diversity_scores.append(1.0 - sim)

    expert_diversity = np.mean(diversity_scores) if diversity_scores else 0.0

    return {
        "avg_load_balance": avg_load_balance,
        "avg_expert_utilization": avg_expert_utilization,
        "expert_importance_score": expert_importance_score,
        "expert_diversity": expert_diversity,
    }


def plot_moe_expert_utilization(expert_utilization: Dict[int, Dict[str, np.ndarray]], output_path: Path):
    """Plot MoE expert utilization statistics."""
    if not HAS_MATPLOTLIB:
        print("Skipping MoE utilization plot - matplotlib not available")
        return

    num_layers = len(expert_utilization)
    num_experts = len(expert_utilization[0]["expert_utilization"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Expert utilization heatmap
    utilization_matrix = np.zeros((num_layers, num_experts))
    for l in range(num_layers):
        utilization_matrix[l] = expert_utilization[l]["expert_utilization"]

    im = axes[0, 0].imshow(utilization_matrix, cmap="YlOrRd", aspect="auto")
    axes[0, 0].set_xlabel("Expert Index")
    axes[0, 0].set_ylabel("Layer Index")
    axes[0, 0].set_title("Expert Utilization per Layer")
    plt.colorbar(im, ax=axes[0, 0], label="Utilization %")

    # 2. Load balancing score per layer
    load_balance_scores = [expert_utilization[l]["load_balance_score"] for l in range(num_layers)]
    axes[0, 1].bar(range(num_layers), load_balance_scores, color="steelblue")
    axes[0, 1].set_xlabel("Layer Index")
    axes[0, 1].set_ylabel("Load Balance Score")
    axes[0, 1].set_title("Load Balancing Score per Layer\n(Higher = More Balanced)")
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Expert selection counts
    selection_matrix = np.zeros((num_layers, num_experts))
    for l in range(num_layers):
        selection_matrix[l] = expert_utilization[l]["expert_selections"]

    im2 = axes[1, 0].imshow(selection_matrix, cmap="Blues", aspect="auto")
    axes[1, 0].set_xlabel("Expert Index")
    axes[1, 0].set_ylabel("Layer Index")
    axes[1, 0].set_title("Expert Selection Counts per Layer")
    plt.colorbar(im2, ax=axes[1, 0], label="Selection Count")

    # 4. Average utilization per expert (across all layers)
    avg_util_per_expert = np.mean(utilization_matrix, axis=0)
    axes[1, 1].bar(range(num_experts), avg_util_per_expert, color="coral")
    axes[1, 1].set_xlabel("Expert Index")
    axes[1, 1].set_ylabel("Average Utilization %")
    axes[1, 1].set_title("Average Expert Utilization (Across All Layers)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved MoE expert utilization plot to {output_path}")


def plot_moe_expert_importance(expert_importance: Dict[int, Dict[str, float]], output_path: Path):
    """Plot MoE expert importance scores."""
    if not HAS_MATPLOTLIB:
        print("Skipping MoE importance plot - matplotlib not available")
        return

    layers = sorted(expert_importance.keys())
    importance_values = [expert_importance[l]["layer_importance"] for l in layers]
    loss_increases = [expert_importance[l]["loss_increase"] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Expert importance score
    axes[0].bar(layers, importance_values, color="steelblue")
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Expert Block Importance Score")
    axes[0].set_title("MoE Expert Block Importance per Layer")
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)

    # Loss increase
    axes[1].bar(layers, loss_increases, color="coral")
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("Loss Increase")
    axes[1].set_title("Loss Increase When Expert Block is Removed")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved MoE expert importance plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute layer importance metrics")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument(
        "--output_dir", type=str, default="./layer_importance_results", help="Directory to save results"
    )
    parser.add_argument("--num_samples", type=int, default=100, help="Number of sample sequences to use")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length for samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--compute_causal", action="store_true", help="Compute causal effect metric")
    parser.add_argument("--compute_permutation", action="store_true", help="Compute permutation score metric")
    parser.add_argument(
        "--compute_similarity", action="store_true", help="Compute layer similarity matrix (based on activations)"
    )
    parser.add_argument(
        "--compute_moe_utilization", action="store_true", help="Compute MoE expert utilization (MoE models only)"
    )
    parser.add_argument(
        "--compute_moe_importance", action="store_true", help="Compute MoE expert importance (MoE models only)"
    )
    parser.add_argument("--skip_layer_1_with_next", action="store_true", help="Test layer 1 with its next layer only")
    parser.add_argument("--skip_layer_2_with_rest", action="store_true", help="Test layer 2 with layers 2-n")
    parser.add_argument("--skip_layer_3_with_rest", action="store_true", help="Test layer 3 with layers 3-n")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer, is_moe = load_model_and_tokenizer(args.model_path, args.device)
    num_layers = len(model.model.layers)

    print(f"Model has {num_layers} layers")
    if is_moe:
        print(f"Model type: MoE (num_experts={model.config.num_experts})")
    else:
        print(f"Model type: Dense")

    # Create sample data
    sample_data = create_sample_data(tokenizer, num_samples=args.num_samples, seq_length=args.seq_length)

    input_ids = sample_data["input_ids"]
    attention_mask = sample_data["attention_mask"]

    # Compute baseline
    print("\nComputing baseline...")
    baseline_loss, baseline_hidden_states, baseline_ppl = compute_baseline_forward(
        model, input_ids, attention_mask, args.device
    )
    print(f"Baseline Loss: {baseline_loss:.4f}")
    print(f"Baseline Perplexity: {baseline_ppl:.4f}")

    results = {
        "baseline_loss": baseline_loss,
        "baseline_perplexity": baseline_ppl,
        "num_layers": num_layers,
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "model_type": "moe" if is_moe else "dense",
    }

    if is_moe:
        results["num_experts"] = model.config.num_experts
        results["experts_per_token"] = model.config.num_experts_per_tok

    # Compute causal effects
    if args.compute_causal:
        print("\n" + "=" * 60)
        print("Computing Causal Effect Metric")
        print("=" * 60)

        causal_effect_matrix = compute_all_causal_effects(model, input_ids, attention_mask, args.device, num_layers)

        # Save results
        np.save(output_dir / "causal_effect_matrix.npy", causal_effect_matrix)
        results["causal_effect_matrix"] = causal_effect_matrix.tolist()

        # Plot
        plot_causal_effects(causal_effect_matrix, output_dir / "causal_effects_heatmap.png")

        print(f"\nCausal effect matrix shape: {causal_effect_matrix.shape}")
        print(f"Average causal effect per layer:")
        for i in range(num_layers):
            avg_effect = np.mean(causal_effect_matrix[i, i + 1 :])
            print(f"  Layer {i}: {avg_effect:.4f}")

    # Compute permutation scores
    if args.compute_permutation:
        print("\n" + "=" * 60)
        print("Computing Permutation Score Metric")
        print("=" * 60)

        # Determine which layer pairs to test
        layer_pairs = []

        # Test all pairs by default
        if not (args.skip_layer_1_with_next or args.skip_layer_2_with_rest or args.skip_layer_3_with_rest):
            for l1 in range(num_layers):
                for l2 in range(l1 + 1, num_layers):
                    layer_pairs.append((l1, l2))
        else:
            # Test specific pairs
            if args.skip_layer_1_with_next:
                if 1 < num_layers - 1:
                    layer_pairs.append((1, 2))

            if args.skip_layer_2_with_rest:
                for l2 in range(2, num_layers):
                    if 2 != l2:
                        layer_pairs.append((2, l2))

            if args.skip_layer_3_with_rest:
                for l2 in range(3, num_layers):
                    if 3 != l2:
                        layer_pairs.append((3, l2))

        print(f"Testing {len(layer_pairs)} layer pairs...")

        permutation_scores = compute_all_permutation_scores(
            model, input_ids, attention_mask, args.device, num_layers, layer_pairs
        )

        # Compute global permutation scores (comparable across different networks)
        global_scores = compute_global_permutation_score(permutation_scores, num_layers, method="mean")

        # Save results
        results["permutation_scores"] = {f"{l1}_{l2}": score for (l1, l2), score in permutation_scores.items()}
        results["global_permutation_scores"] = global_scores

        # Plot
        plot_permutation_scores(permutation_scores, num_layers, output_dir / "permutation_scores_heatmap.png")

        print(f"\nPermutation scores:")
        for (l1, l2), score in sorted(permutation_scores.items()):
            print(f"  Layers ({l1}, {l2}): {score:.4f}")

        print(f"\n" + "=" * 60)
        print("Global Permutation Scores (Comparable Across Networks)")
        print("=" * 60)
        print(f"  Mean Score: {global_scores['global_score_mean']:.4f}")
        print(f"  Normalized Score (by depth): {global_scores['global_score_normalized']:.4f}")
        print(f"  Weighted Score (early layers weighted more): {global_scores['global_score_weighted']:.4f}")
        print(f"\n  Average score per layer:")
        for i, avg_score in enumerate(global_scores["layer_avg_scores"]):
            print(f"    Layer {i}: {avg_score:.4f}")

    # Compute layer similarity matrix
    if args.compute_similarity:
        print("\n" + "=" * 60)
        print("Computing Layer Similarity Matrix (Based on Activations)")
        print("=" * 60)

        similarity_matrix = compute_layer_similarity(model, input_ids, attention_mask, args.device, num_layers)

        # Save results
        np.save(output_dir / "layer_similarity_matrix.npy", similarity_matrix)
        results["similarity_matrix"] = similarity_matrix.tolist()

        # Plot
        plot_layer_similarity_matrix(
            similarity_matrix,
            output_dir / "layer_similarity_heatmap.png",
            title="Layer Similarity Matrix (Based on Activations)",
        )

        print(f"\nSimilarity matrix shape: {similarity_matrix.shape}")
        print(
            f"Average similarity between layers: {np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]):.4f}"
        )

        # Find most similar layer pairs (excluding self-similarity)
        similar_pairs = []
        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                similar_pairs.append(((i, j), similarity_matrix[i, j]))

        similar_pairs.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 5 most similar layer pairs:")
        for i, (pair, sim) in enumerate(similar_pairs[:5], 1):
            print(f"  {i}. Layers {pair[0]} and {pair[1]}: {sim:.4f}")

        print(f"\nBottom 5 least similar layer pairs:")
        for i, (pair, sim) in enumerate(similar_pairs[-5:], 1):
            print(f"  {i}. Layers {pair[0]} and {pair[1]}: {sim:.4f}")

    # Compute MoE expert utilization (MoE models only)
    if args.compute_moe_utilization:
        if not is_moe:
            print("\nWarning: --compute_moe_utilization specified but model is not a MoE model. Skipping.")
        else:
            print("\n" + "=" * 60)
            print("Computing MoE Expert Utilization")
            print("=" * 60)

            expert_utilization = compute_moe_expert_utilization(
                model, input_ids, attention_mask, args.device, num_layers
            )

            # Save results
            results["moe_expert_utilization"] = {
                str(l): {
                    "expert_selections": expert_utilization[l]["expert_selections"].tolist(),
                    "expert_utilization": expert_utilization[l]["expert_utilization"].tolist(),
                    "load_balance_score": expert_utilization[l]["load_balance_score"],
                }
                for l in range(num_layers)
            }

            # Plot
            plot_moe_expert_utilization(expert_utilization, output_dir / "moe_expert_utilization.png")

            print(f"\nMoE Expert Utilization Summary:")
            for layer_idx in range(num_layers):
                print(f"  Layer {layer_idx}:")
                print(f"    Load Balance Score: {expert_utilization[layer_idx]['load_balance_score']:.4f}")
                print(f"    Top 3 Most Used Experts:")
                util = expert_utilization[layer_idx]["expert_utilization"]
                top_experts = np.argsort(util)[-3:][::-1]
                for expert_idx in top_experts:
                    print(f"      Expert {expert_idx}: {util[expert_idx]*100:.2f}%")

    # Compute MoE expert importance (MoE models only)
    if args.compute_moe_importance:
        if not is_moe:
            print("\nWarning: --compute_moe_importance specified but model is not a MoE model. Skipping.")
        else:
            print("\n" + "=" * 60)
            print("Computing MoE Expert Importance")
            print("=" * 60)

            expert_importance = compute_moe_expert_importance(
                model, input_ids, attention_mask, args.device, baseline_loss, num_layers
            )

            # Save results
            results["moe_expert_importance"] = {
                str(l): {
                    "layer_importance": expert_importance[l]["layer_importance"],
                    "loss_increase": expert_importance[l]["loss_increase"],
                }
                for l in range(num_layers)
            }

            # Plot
            plot_moe_expert_importance(expert_importance, output_dir / "moe_expert_importance.png")

            print(f"\nMoE Expert Importance Summary:")
            for layer_idx, metrics in sorted(expert_importance.items()):
                print(f"  Layer {layer_idx}:")
                print(f"    Importance Score: {metrics['layer_importance']:.4f}")
                print(f"    Loss Increase: {metrics['loss_increase']:.4f}")

            # Sort by importance
            sorted_layers = sorted(expert_importance.items(), key=lambda x: x[1]["layer_importance"], reverse=True)
            print(f"\n  Top 3 Most Important Expert Blocks:")
            for i, (layer_idx, metrics) in enumerate(sorted_layers[:3], 1):
                print(f"    {i}. Layer {layer_idx}: {metrics['layer_importance']:.4f}")
            print(f"  Bottom 3 Least Important Expert Blocks:")
            for i, (layer_idx, metrics) in enumerate(sorted_layers[-3:], 1):
                print(f"    {i}. Layer {layer_idx}: {metrics['layer_importance']:.4f}")

    # Compute global MoE metrics if any MoE metrics were computed
    if is_moe and (args.compute_moe_utilization or args.compute_moe_importance):
        print("\n" + "=" * 60)
        print("Global MoE Metrics")
        print("=" * 60)

        # Get expert utilization and importance
        expert_utilization = results.get("moe_expert_utilization", {})
        expert_importance = results.get("moe_expert_importance", {})

        # Convert back to dict format if needed
        if expert_utilization:
            expert_utilization_dict = {}
            for l, stats in expert_utilization.items():
                expert_utilization_dict[int(l)] = {
                    "expert_selections": np.array(stats["expert_selections"]),
                    "expert_utilization": np.array(stats["expert_utilization"]),
                    "load_balance_score": stats["load_balance_score"],
                }
        else:
            expert_utilization_dict = {
                l: {
                    "expert_selections": np.zeros(model.config.num_experts),
                    "expert_utilization": np.zeros(model.config.num_experts),
                    "load_balance_score": 0.0,
                }
                for l in range(num_layers)
            }

        if expert_importance:
            expert_importance_dict = {}
            for l, stats in expert_importance.items():
                expert_importance_dict[int(l)] = {
                    "layer_importance": stats["layer_importance"],
                    "loss_increase": stats["loss_increase"],
                }
        else:
            expert_importance_dict = {l: {"layer_importance": 0.0, "loss_increase": 0.0} for l in range(num_layers)}

        global_moe_metrics = compute_moe_global_metrics(expert_utilization_dict, expert_importance_dict, num_layers)

        results["global_moe_metrics"] = global_moe_metrics

        print(f"  Average Load Balance Score: {global_moe_metrics['avg_load_balance']:.4f}")
        print(f"  Average Expert Utilization: {global_moe_metrics['avg_expert_utilization']*100:.2f}%")
        print(f"  Expert Importance Score: {global_moe_metrics['expert_importance_score']:.4f}")
        print(f"  Expert Diversity: {global_moe_metrics['expert_diversity']:.4f}")

    # Create summary plot if both metrics computed
    if args.compute_causal and args.compute_permutation:
        plot_layer_importance_summary(
            causal_effect_matrix, permutation_scores, num_layers, output_dir / "layer_importance_summary.png"
        )

    # Save results as JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    # fix the seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    main()
