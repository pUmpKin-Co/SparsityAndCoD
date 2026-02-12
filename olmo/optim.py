import logging
import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, replace
from math import cos, pi, sqrt
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.optimizer import Optimizer as OptimizerBase

from olmo.exceptions import OLMoConfigurationError

from . import LayerNormBase
from .config import (
    OptimizerType,
    SchedulerConfig,
    SchedulerType,
    TrainConfig,
)
from .torch_util import get_default_device, is_distributed

# Try to import matplotlib, but handle gracefully if not available
try:
    import matplotlib
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

__all__ = [
    "Optimizer",
    "LionW",
    "AdamW",
    "Scheduler",
    "CosWithWarmup",
    "LinearWithWarmup",
    "InvSqrtWithWarmup",
    "MaxScheduler",
    "ConstantScheduler",
    "CosLinearEnvelope",
    "BoltOnWarmupScheduler",
    "build_optimizer",
    "build_scheduler",
]


log = logging.getLogger(__name__)


class Optimizer(OptimizerBase):
    def __init__(
        self,
        *args,
        record_update_metrics: bool = False,
        selective_updates: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._record_update_metrics = record_update_metrics
        self._collecting_metrics = False
        self._selective_updates = selective_updates
        self._gss_threshold = kwargs.get("gss_threshold", None)

        # Spike detection attributes
        self._spike_threshold = 100.0  # Default threshold for grad^2 / v
        self._global_spike_count = (
            0  # Number of steps that had spike parameters
        )
        self._step_spike_counts = (
            []
        )  # Number of spike parameters per step (can be disabled for memory)
        self._layer_spike_ratios = (
            {}
        )  # Spike ratios per layer across all steps
        self._layer_param_counts = (
            {}
        )  # Total parameter counts per layer for ratio calculation
        self._track_step_history = kwargs.get(
            "track_step_history", True
        )  # Whether to store step counts history

    def set_spike_threshold(self, threshold: float):
        """Set the threshold for spike detection."""
        self._spike_threshold = threshold

    def get_spike_statistics(self) -> Dict[str, Any]:
        """
        Get current spike statistics.

        Returns:
            Dictionary containing:
            - global_spike_count: Number of training steps that had spike parameters
            - step_spike_counts: List of spike parameter counts per step
            - layer_spike_ratios: Spike ratios per layer across all steps
        """
        return {
            "global_spike_count": self._global_spike_count,
            "step_spike_counts": self._step_spike_counts.copy(),
            "layer_spike_ratios": self._layer_spike_ratios.copy(),
        }

    def reset_spike_statistics(self):
        """Reset spike statistics."""
        self._global_spike_count = 0
        self._step_spike_counts = []
        self._layer_spike_ratios = {}
        self._layer_param_counts = {}

    def _clean_param_name(self, name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "")

    def _get_layer_name(self, param_name: str) -> str:

        if "module" in param_name:
            param_name = param_name.replace("module.", "")

        """Extract layer name from parameter name for OLMo model structure."""
        parts = param_name.split(".")

        # Handle OLMo-specific parameter naming
        if len(parts) >= 3 and parts[0] == "transformer":
            if parts[1] == "blocks":
                # transformer.blocks.{layer_id}.{component} -> transformer.blocks.{layer_id}
                if len(parts) >= 3:
                    return ".".join(parts[:3])
            elif parts[1] == "block_groups":
                # transformer.block_groups.{group_id}.{block_id}.{component} -> transformer.block_groups.{group_id}.{block_id}
                if len(parts) >= 4:
                    return ".".join(parts[:4])
            elif parts[1] in ["wte", "wpe", "emb_norm", "emb_drop"]:
                # Embedding components
                return "embedding"
            elif parts[1] in ["ln_f", "ff_out"]:
                # Output head components
                return "lm_head"

        # Fallback for other parameter structures
        if len(parts) >= 2:
            return ".".join(parts[:2])
        else:
            return parts[0] if parts else "unknown"

    def _get_layer_order_key(self, layer_name: str) -> tuple:
        """Get a tuple for sorting layer names in model structure order."""
        if layer_name == "embedding":
            return (0, 0, 0)
        elif layer_name == "lm_head":
            return (2, 0, 0)
        elif layer_name.startswith("transformer.blocks."):
            # Extract block number: transformer.blocks.{layer_id}
            parts = layer_name.split(".")
            if len(parts) >= 3:
                try:
                    block_id = int(parts[2])
                    return (1, block_id, 0)
                except ValueError:
                    pass
        elif layer_name.startswith("transformer.block_groups."):
            # Extract group and block numbers: transformer.block_groups.{group_id}.{block_id}
            parts = layer_name.split(".")
            if len(parts) >= 4:
                try:
                    group_id = int(parts[2])
                    block_id = int(parts[3])
                    return (
                        1,
                        group_id * 100 + block_id,
                        0,
                    )  # Simple ordering for block groups
                except ValueError:
                    pass

        # Default ordering for unknown layers
        return (3, 0, 0)

    def _update_global_spike_count(self) -> bool:
        """
        Lightweight spike detection to update global spike count.
        Returns True if current step has spike parameters, False otherwise.
        Only updates the global count, doesn't collect detailed metrics.
        """
        if not hasattr(self, "_spike_threshold"):
            return False

        current_step_has_spikes = False

        for group in self.param_groups:
            for name, p in zip(group["param_names"], group["params"]):
                if p.grad is None:
                    continue

                # Get optimizer state
                state = self.get_state_for_param(p)

                # Calculate grad^2 / v for supported optimizers
                spike_metric = None
                if (
                    "exp_avg_sq" in state
                    and state["exp_avg_sq"] is not None
                ):
                    # For AdamW-like optimizers
                    grad_squared = p.grad**2
                    v = state["exp_avg_sq"]
                    spike_metric = grad_squared / (v + 1e-8)
                elif hasattr(self, "_compute_spike_metric_for_optimizer"):
                    # For custom optimizers
                    spike_metric = self._compute_spike_metric_for_optimizer(
                        p, state
                    )

                if spike_metric is not None:
                    # Check if any parameters have spikes above threshold
                    if (spike_metric > self._spike_threshold).any():
                        current_step_has_spikes = True
                        break

            if current_step_has_spikes:
                break

        # Update global spike count (number of steps with spikes)
        if current_step_has_spikes:
            self._global_spike_count += 1

        return current_step_has_spikes

    def _compute_grad_squared_over_v_statistics(
        self,
        collect_param_metrics: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        is_fsdp: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute grad^2 / v statistics for spike detection.

        This method computes the ratio of gradient squared to the exponential moving average
        of squared gradients (second moment) for each parameter. Parameters with ratios above
        the threshold are counted as "spikes".

        The global spike count tracks the number of training steps that had at least one
        spike parameter, while step_spike_counts tracks the number of spike parameters per step.

        Returns:
            Dictionary containing spike detection metrics including counts and ratios.
        """
        if not collect_param_metrics:
            return {}

        device = get_default_device() if device is None else device

        # Track spike statistics
        current_step_spikes = 0
        total_params = 0
        layer_spikes = {}
        layer_param_counts = {}
        spike_ratios = {}

        dst_rank = 0
        if process_group is not None:
            dst_rank = dist.get_global_rank(process_group, 0)

        # Per-parameter spike metrics
        per_param_spike_metrics = []
        per_param_spike_names = []

        for group in self.param_groups:
            for name, p in zip(group["param_names"], group["params"]):
                clean_name = self._clean_param_name(name)
                layer_name = self._get_layer_name(clean_name)

                if p.grad is None:
                    continue

                # Get optimizer state
                state = self.get_state_for_param(p)

                # Calculate grad^2 / v for supported optimizers
                spike_metric = None
                if (
                    "exp_avg_sq" in state
                    and state["exp_avg_sq"] is not None
                ):
                    # For AdamW-like optimizers
                    grad_squared = p.grad**2
                    v = state["exp_avg_sq"]
                    # Add small epsilon to avoid division by zero
                    spike_metric = grad_squared / (v + 1e-8)
                elif hasattr(self, "_compute_spike_metric_for_optimizer"):
                    # For custom optimizers
                    spike_metric = self._compute_spike_metric_for_optimizer(
                        p, state
                    )

                if spike_metric is not None:
                    # Count spikes (parameters above threshold)
                    spikes_mask = spike_metric > self._spike_threshold
                    param_spike_count = spikes_mask.sum().item()
                    param_total = spike_metric.numel()

                    current_step_spikes += param_spike_count
                    total_params += param_total

                    # Track per-layer spikes and parameter counts
                    if layer_name not in layer_spikes:
                        layer_spikes[layer_name] = 0
                        layer_param_counts[layer_name] = 0
                    layer_spikes[layer_name] += param_spike_count
                    layer_param_counts[layer_name] += param_total

                    # Store max spike metric for this parameter
                    max_spike = spike_metric.max().item()
                    per_param_spike_metrics.append(
                        torch.tensor([max_spike], device=device)
                    )
                    per_param_spike_names.append(f"spike_max/{clean_name}")

        # Reduce metrics across ranks if distributed
        if is_distributed() and process_group is not None:
            # Reduce spike counts
            current_step_spikes_tensor = torch.tensor(
                [current_step_spikes], device=device, dtype=torch.long
            )
            total_params_tensor = torch.tensor(
                [total_params], device=device, dtype=torch.long
            )

            dist.reduce(
                current_step_spikes_tensor,
                dst_rank,
                op=dist.ReduceOp.SUM,
                group=process_group,
            )
            dist.reduce(
                total_params_tensor,
                dst_rank,
                op=dist.ReduceOp.SUM,
                group=process_group,
            )

            current_step_spikes = current_step_spikes_tensor.item()
            total_params = total_params_tensor.item()

            # Reduce per-parameter max spike metrics
            # Note: In FSDP, different ranks have different subsets of parameters,
            # so we can't directly reduce per-parameter metrics across ranks.
            # We skip this reduction to avoid tensor shape mismatches.
            if per_param_spike_metrics and not is_fsdp:
                all_spike_metrics = torch.cat(per_param_spike_metrics).to(
                    device
                )
                dist.reduce(
                    all_spike_metrics,
                    dst_rank,
                    op=dist.ReduceOp.MAX,
                    group=process_group,
                )
                per_param_spike_metrics = all_spike_metrics.split(1)

        if self._track_step_history:
            self._step_spike_counts.append(current_step_spikes)

        # Calculate and update layer spike ratios
        for layer_name in layer_spikes:
            spike_count = layer_spikes[layer_name]
            param_count = layer_param_counts[layer_name]
            layer_ratio = spike_count / max(param_count, 1)

            # Update cumulative layer spike ratios (using exponential moving average)
            if layer_name not in self._layer_spike_ratios:
                self._layer_spike_ratios[layer_name] = layer_ratio
                self._layer_param_counts[layer_name] = param_count
            else:
                # Update with exponential moving average (alpha = 0.1)
                alpha = 0.1
                self._layer_spike_ratios[layer_name] = (
                    1 - alpha
                ) * self._layer_spike_ratios[
                    layer_name
                ] + alpha * layer_ratio
                self._layer_param_counts[layer_name] = max(
                    self._layer_param_counts[layer_name], param_count
                )

        # Calculate spike ratio
        spike_ratio = current_step_spikes / max(total_params, 1)

        # Collect metrics
        metrics = {
            "spike_detection/current_step_spikes": torch.tensor(
                current_step_spikes, device=device
            ),
            "spike_detection/spike_ratio": torch.tensor(
                spike_ratio, device=device
            ),
            "spike_detection/global_spike_count": torch.tensor(
                self._global_spike_count, device=device
            ),
        }

        # Add layer-wise spike ratio metrics
        for layer_name, layer_ratio in self._layer_spike_ratios.items():
            # Create clean metric name
            clean_layer_name = layer_name.replace(".", "_")
            metrics[f"spike_detection/layer_ratio_{clean_layer_name}"] = (
                torch.tensor(layer_ratio, device=device)
            )

        # Add per-parameter max spike metrics
        for name, metric in zip(
            per_param_spike_names, per_param_spike_metrics
        ):
            metrics[name] = metric.squeeze(0)

        return metrics

    def generate_spike_plot(
        self, save_path: str, step: int
    ) -> Optional[str]:
        """
        Generate a bar plot showing spike ratios per layer.
        Returns the path to the saved plot or None if no spikes to plot.
        """
        if not self._layer_spike_ratios:
            return None

        if not MATPLOTLIB_AVAILABLE:
            log.warning(
                "Matplotlib not available, cannot generate spike plot"
            )
            return None

        try:
            matplotlib.use("Agg")  # Use non-interactive backend

            fig, ax = plt.subplots(figsize=(12, 8))

            # Sort layers by model structure order (embedding, blocks, lm_head)
            layer_items = list(self._layer_spike_ratios.items())
            layer_items.sort(key=lambda x: self._get_layer_order_key(x[0]))

            layers = [item[0] for item in layer_items]
            ratios = [item[1] for item in layer_items]

            # Create more readable layer names for x-axis
            readable_names = []
            for layer_name in layers:
                if layer_name == "embedding":
                    readable_names.append("Embedding")
                elif layer_name == "lm_head":
                    readable_names.append("LM Head")
                elif layer_name.startswith("transformer.blocks."):
                    # Extract block number
                    parts = layer_name.split(".")
                    if len(parts) >= 3:
                        readable_names.append(f"Block {parts[2]}")
                    else:
                        readable_names.append(layer_name)
                elif layer_name.startswith("transformer.block_groups."):
                    # Extract group and block numbers
                    parts = layer_name.split(".")
                    if len(parts) >= 4:
                        readable_names.append(
                            f"Group {parts[2]}.{parts[3]}"
                        )
                    else:
                        readable_names.append(layer_name)
                else:
                    readable_names.append(layer_name)

            bars = ax.bar(
                range(len(layers)), ratios, alpha=0.7, color="red"
            )
            ax.set_xlabel("Layer")
            ax.set_ylabel("Spike Ratio")
            ax.set_title(f"Parameter Spike Ratios by Layer (Step {step})")

            # Set x-axis labels
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(readable_names, rotation=45, ha="right")

            # Add value labels on bars
            for bar, ratio in zip(bars, ratios):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{ratio:.4f}",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            return save_path
        except Exception as e:
            log.warning(f"Failed to generate spike plot: {e}")
            return None

    @torch.no_grad()
    def clip_grads_and_collect_metrics(
        self,
        global_step: int,
        collect_param_metrics: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        collect_spike_metrics: bool = True,
        is_fsdp: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Clips gradients for every group that has the field `max_grad_norm`.
        At the same time collect metrics for each parameter and its gradient.
        Now also includes spike detection metrics.
        """
        self._collecting_metrics = collect_param_metrics
        device = get_default_device() if device is None else device

        # NOTE: during distributed training we're making an assumption that the order of
        # the param groups and the params within each group are the same across all ranks.
        # This is justified since we initialize the parameter groups in every rank by iterating over
        # `module.parameters()` or `module.named_modules()` / `module.named_parameters()`, each of which
        # provides a consistent order.
        #  For each parameter (with a gradient) we'll collect:
        # - min, max, avg, norm of the param itself
        # - min, max, avg, norm of the param's gradient
        # - min, max, avg, norm of any additional per-parameter optimizer state metrics returned from
        #   `self.get_state_for_param()`.
        # Afterwards we'll reduce these all over all ranks.
        per_param_min_metrics: List[torch.Tensor] = []
        per_param_max_metrics: List[torch.Tensor] = []
        per_param_sum_metrics: List[torch.Tensor] = []
        per_param_norm_metrics: List[torch.Tensor] = []
        per_param_numel_metrics: List[torch.Tensor] = []

        per_param_min_metric_names: List[str] = []
        per_param_max_metric_names: List[str] = []
        per_param_avg_metric_names: List[str] = []
        per_param_norm_metric_names: List[str] = []

        dst_rank = 0
        if process_group is not None:
            dst_rank = dist.get_global_rank(process_group, 0)

        #######################################################################
        # part 1: collect metrics locally
        #######################################################################
        for group in self.param_groups:
            for name, p in zip(group["param_names"], group["params"]):
                name = self._clean_param_name(name)
                # Always need to collect the norm of gradients for clipping, even if we're not collecting
                # other metrics.
                tensors: List[Optional[torch.Tensor]] = [p.grad]
                prefixes: List[str] = [f"grad/{name}"]
                if collect_param_metrics:
                    state = self.get_state_for_param(p)
                    sorted_state_keys = sorted([k for k in state.keys()])
                    tensors.extend(
                        [p] + [state[key] for key in sorted_state_keys]
                    )
                    prefixes.extend(
                        [f"param/{name}"]
                        + [f"{key}/{name}" for key in sorted_state_keys]
                    )
                assert len(tensors) == len(prefixes)

                # Get min, max, avg, and norm for all `tensors` associated with the parameter.
                for x, prefix in zip(tensors, prefixes):
                    # grad or state tensors could be none for params that have their shards completely on
                    # other ranks.
                    if x is not None and x.numel() > 0:
                        if collect_param_metrics:
                            x_abs = x.abs()
                            per_param_min_metrics.append(
                                x_abs.min()
                                .unsqueeze(0)
                                .to(dtype=torch.float32)
                            )
                            per_param_max_metrics.append(
                                x_abs.max()
                                .unsqueeze(0)
                                .to(dtype=torch.float32)
                            )
                            per_param_sum_metrics.append(
                                x.sum().unsqueeze(0).to(dtype=torch.float32)
                            )
                            per_param_numel_metrics.append(
                                torch.tensor(
                                    [x.numel()],
                                    device=device,
                                    dtype=torch.float32,
                                )
                            )
                        per_param_norm_metrics.append(
                            torch.linalg.vector_norm(
                                x, 2.0, dtype=torch.float32
                            ).unsqueeze(0)
                        )
                    else:
                        if collect_param_metrics:
                            per_param_min_metrics.append(
                                torch.tensor(
                                    [float("inf")],
                                    device=device,
                                    dtype=torch.float32,
                                )
                            )
                            per_param_max_metrics.append(
                                torch.tensor(
                                    [0.0],
                                    device=device,
                                    dtype=torch.float32,
                                )
                            )
                            per_param_sum_metrics.append(
                                torch.tensor(
                                    [0.0],
                                    device=device,
                                    dtype=torch.float32,
                                )
                            )
                            per_param_numel_metrics.append(
                                torch.tensor(
                                    [0.0],
                                    device=device,
                                    dtype=torch.float32,
                                )
                            )
                        per_param_norm_metrics.append(
                            torch.tensor(
                                [0.0], device=device, dtype=torch.float32
                            )
                        )
                    if collect_param_metrics:
                        per_param_min_metric_names.append(f"{prefix}.min")
                        per_param_max_metric_names.append(f"{prefix}.max")
                        per_param_avg_metric_names.append(f"{prefix}.avg")
                    per_param_norm_metric_names.append(f"{prefix}.norm")

        assert (
            len(per_param_min_metrics)
            == len(per_param_min_metric_names)
            == len(per_param_max_metrics)
            == len(per_param_max_metric_names)
            == len(per_param_sum_metrics)
            == len(per_param_numel_metrics)
            == len(per_param_avg_metric_names)
        )
        assert len(per_param_norm_metrics) == len(
            per_param_norm_metric_names
        )

        def is_grad_norm_metric(metric_name: str) -> bool:
            return metric_name.startswith("grad/") and metric_name.endswith(
                ".norm"
            )

        #######################################################################
        # part 2: reduce metrics over ranks
        #######################################################################
        param_group_sharded = False
        for group in self.param_groups:
            param_group_sharded = param_group_sharded or group.get(
                "sharded", False
            )

        total_grad_norm: torch.Tensor
        per_param_avg_metrics: List[torch.Tensor] = []
        if is_distributed() and param_group_sharded:
            # Reduce metrics across all ranks. Note that we can use a `reduce` for most cases
            # instead of an `all_reduce`, but we need `all_reduce` for norms so that all ranks
            # get the right value for gradient norms so they can clip correctly.
            # Reduce mins.
            if per_param_min_metrics:
                all_mins = torch.cat(per_param_min_metrics).to(device)
                dist.reduce(
                    all_mins,
                    dst_rank,
                    op=dist.ReduceOp.MIN,
                    group=process_group,
                )
                per_param_min_metrics = all_mins.split(1)
            # Reduce maxs.
            if per_param_max_metrics:
                all_maxs = torch.cat(per_param_max_metrics).to(device)
                dist.reduce(
                    all_maxs,
                    dst_rank,
                    op=dist.ReduceOp.MAX,
                    group=process_group,
                )
                per_param_max_metrics = all_maxs.split(1)
            # Reduce sums or just norms.
            all_norms = torch.cat(per_param_norm_metrics).to(device) ** 2.0
            if per_param_sum_metrics and per_param_numel_metrics:
                all_sums = torch.cat(per_param_sum_metrics).to(device)
                all_numels = torch.cat(per_param_numel_metrics).to(device)
                all_sums_norms_numels = torch.cat(
                    [
                        all_sums.unsqueeze(0),
                        all_norms.unsqueeze(0),
                        all_numels.unsqueeze(0),
                    ],
                    dim=0,
                )
                dist.all_reduce(
                    all_sums_norms_numels,
                    op=dist.ReduceOp.SUM,
                    group=process_group,
                )
                all_sums, all_norms, all_numels = (
                    all_sums_norms_numels.split(1)
                )
                # Get averages.
                # NOTE: could get infs for non-rank0 processes but that's okay.
                per_param_avg_metrics = (
                    (all_sums / all_numels).squeeze(0).split(1)
                )
            else:
                dist.all_reduce(
                    all_norms, op=dist.ReduceOp.SUM, group=process_group
                )
            grad_norm_metric_mask = torch.tensor(
                [
                    float(is_grad_norm_metric(n))
                    for n in per_param_norm_metric_names
                ],
                device=all_norms.device,
            )
            total_grad_norm = (
                all_norms * grad_norm_metric_mask
            ).sum() ** 0.5
            per_param_norm_metrics = (
                (all_norms ** (0.5)).squeeze(0).split(1)
            )
        else:
            total_grad_norm = (
                torch.cat(
                    [
                        m
                        for m, n in zip(
                            per_param_norm_metrics,
                            per_param_norm_metric_names,
                        )
                        if is_grad_norm_metric(n)
                    ]
                )
                ** 2.0
            ).sum() ** 0.5
            per_param_avg_metrics = [
                x / n
                for x, n in zip(
                    per_param_sum_metrics, per_param_numel_metrics
                )
            ]

        assert len(per_param_avg_metrics) == len(per_param_avg_metric_names)

        # Collect all metrics into a single dict.
        all_metrics: Dict[str, torch.Tensor] = {}
        if collect_param_metrics:
            for metric_name, metric in zip(
                per_param_min_metric_names, per_param_min_metrics
            ):
                all_metrics[metric_name] = metric.squeeze(0)
            for metric_name, metric in zip(
                per_param_max_metric_names, per_param_max_metrics
            ):
                all_metrics[metric_name] = metric.squeeze(0)
            for metric_name, metric in zip(
                per_param_avg_metric_names, per_param_avg_metrics
            ):
                all_metrics[metric_name] = metric.squeeze(0)

        for metric_name, metric in zip(
            per_param_norm_metric_names, per_param_norm_metrics
        ):
            all_metrics[metric_name] = metric.squeeze(0)
        all_metrics["total_grad_norm"] = total_grad_norm

        # Collect spike detection metrics
        if collect_spike_metrics and collect_param_metrics:
            spike_metrics = self._compute_grad_squared_over_v_statistics(
                collect_param_metrics=collect_param_metrics,
                process_group=process_group,
                device=device,
                is_fsdp=is_fsdp,
            )
            all_metrics.update(spike_metrics)

        #######################################################################
        # part 3: clip grads
        #######################################################################
        num_grads_clipped = 0
        num_eligible_grads = 0
        for group in self.param_groups:
            if (
                max_norm_ratio := group.get("max_grad_norm_ratio")
            ) is not None:
                num_clipped = self._do_adaptive_clipping(
                    group,
                    max_norm_ratio,
                    global_step,
                    all_metrics,
                    collect_param_metrics=collect_param_metrics,
                )
            elif (max_norm := group.get("max_grad_norm")) is not None:
                num_clipped = self._do_global_fixed_clipping(
                    group,
                    max_norm,
                    all_metrics,
                    collect_param_metrics=collect_param_metrics,
                )
            elif group.get("adaptive_gradient_clipping") is True:
                num_clipped = self._do_adaptive_clipping_spam(
                    group, all_metrics
                )
            else:
                # No clipping needed.
                continue
            num_eligible_grads += len(group["params"])
            if num_clipped is not None:
                num_grads_clipped += num_clipped

        if collect_param_metrics:
            if num_eligible_grads > 0:
                clipping_rate = torch.tensor(
                    num_grads_clipped / num_eligible_grads, device="cpu"
                )
            else:
                clipping_rate = torch.tensor(0.0, device="cpu")
            all_metrics["clipping_rate"] = clipping_rate

        # total_grad_norm is computed at all steps, even when collect_param_metrics is set to False
        return all_metrics

    @torch.no_grad()
    def _do_adaptive_clipping_spam(
        self,
        group: Dict[str, Any],
        all_metrics: Dict[str, torch.Tensor],
    ) -> Optional[int]:
        num_grads_clipped = 0
        for name, param in zip(group["param_names"], group["params"]):
            name = self._clean_param_name(name)
            grad_norm = all_metrics.get(f"grad/{name}.norm")
            if grad_norm is None:
                continue

            max_gradient = all_metrics.get(f"grad/{name}.max")
            if max_gradient is None:
                max_gradient = torch.max(param.grad.abs())

            state = self.state[param]
            m_max_t = state.get("m_max_t")
            step_t = state.get("step")

            if m_max_t is None:
                m_max_t = 0

            if step_t is None:
                step = 1
            else:
                step = step_t + 1

            m_max_t = self.theta * m_max_t + (1 - self.theta) * max_gradient
            m_max_hat = m_max_t / (1 - self.theta**step)
            mask_grad = param.grad.abs() > m_max_hat
            if mask_grad.sum() > 0:
                param.grad[mask_grad] = (
                    param.grad[mask_grad] / max_gradient * m_max_hat
                )

            num_grads_clipped += mask_grad.sum()
            state["m_max_t"] = m_max_t

        return num_grads_clipped

    @torch.no_grad()
    def _do_adaptive_clipping(
        self,
        group: Dict[str, Any],
        max_norm_ratio: float,
        global_step: int,
        all_metrics: Dict[str, torch.Tensor],
        collect_param_metrics: bool = True,
        device: Optional[torch.device] = None,
    ) -> Optional[int]:
        """
        Do adaptive gradient clipping on a param group.

        If ``collect_param_metrics`` is ``True`` this will return the total number of gradients clipped.
        """
        device = get_default_device() if device is None else device
        num_grads_clipped = 0
        # We'll use the bigger of beta1 and beta2 to update the exponential average of the norm of
        # the gradient (a scalar), not to be confused with the exponential average of the gradient.
        # TODO: handle optimizers that don't have betas.
        beta1, beta2 = group["betas"]
        beta = max(beta1, beta2)
        for name, p in zip(group["param_names"], group["params"]):
            name = self._clean_param_name(name)
            grad_norm = all_metrics.get(f"grad/{name}.norm")
            if grad_norm is None:
                continue

            # Get or initialize the exponential average of grad norm.
            # TODO: The way we have it right now, every rank tracks the `grad_norm_exp_avg` of every parameter,
            # even parameters for which the corresponding local shard is empty. This has the potential to
            # cause some issues with the optimizer, as we ran into with https://github.com/allenai/LLM/pull/372.
            # So we should consider changing how we do this at some point so that we don't add any state
            # to parameters for which the local shard is empty. That would probably add extra distributed
            # communication, at least on steps where we have to log (i.e. when `collect_param_metrics=True`).
            state = self.state[p]
            grad_norm_exp_avg = state.get("grad_norm_exp_avg")
            if grad_norm_exp_avg is None:
                grad_norm_exp_avg = grad_norm.clone().to(device)
                # We don't want to add anything to `state` until `state` has been initialized, otherwise
                # this will crash some optimizers which rely on checking `len(state)`. The downside here
                # is that we won't start tracking `grad_norm_exp_avg` until the 2nd training step.
                if global_step > 1:
                    state["grad_norm_exp_avg"] = grad_norm_exp_avg

            max_allowed_norm = max_norm_ratio * grad_norm_exp_avg
            clip_coef = max_allowed_norm / (grad_norm + 1e-6)

            # Clip the gradients and update the exponential average.
            # Note that multiplying by the clamped coefficient is meaningless when it is
            # equal to 1, but it avoids the host-device sync that would result from `if clip_coef_clamped < 1`.
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            if p.grad is not None:
                # p.grad could be none for some ranks when using FSDP.
                p.grad.detach().mul_(
                    clip_coef_clamped.to(p.grad.device, p.grad.dtype)
                )

            # Update the exponential average of the norm of the gradient with the clipped norm of the gradient.
            grad_norm_exp_avg.lerp_(
                (grad_norm * clip_coef_clamped).to(
                    grad_norm_exp_avg.device
                ),
                1 - beta,
            )
            # Alternative: update with the *unclipped* norm of the gradient.
            #  grad_norm_exp_avg.lerp_(grad_norm.to(grad_norm_exp_avg.device), 1 - beta)

            if collect_param_metrics:
                # Can't avoid host-device sync here.
                if clip_coef_clamped < 1.0:
                    num_grads_clipped += 1
                all_metrics[f"grad_norm_exp_avg/{name}"] = grad_norm_exp_avg
        return num_grads_clipped if collect_param_metrics else None

    @torch.no_grad()
    def _do_global_fixed_clipping(
        self,
        group: Dict[str, Any],
        max_norm: float,
        all_metrics: Dict[str, torch.Tensor],
        collect_param_metrics: bool = True,
        device: Optional[torch.device] = None,
    ) -> Optional[int]:
        """
        Do global fixed gradient clipping on a param group.

        If ``collect_param_metrics`` is ``True`` this will return the total number of gradients clipped.
        """
        device = get_default_device() if device is None else device
        total_grad_norm = all_metrics["total_grad_norm"]
        clip_coef = max_norm / (total_grad_norm.to(device) + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        num_grads_clipped: Optional[int] = None
        if collect_param_metrics:
            # Can't avoid host-device sync here.
            if clip_coef_clamped < 1.0:
                num_grads_clipped = len(group["params"])
        for p in group["params"]:
            # Clip the gradients.
            # Note that multiplying by the clamped coefficient is meaningless when it is
            # equal to 1, but it avoids the host-device sync that would result from `if clip_coef_clamped < 1`.
            if p.grad is not None:
                # p.grad could be none for some ranks when using FSDP.
                p.grad.detach().mul_(
                    clip_coef_clamped.to(p.grad.device, p.grad.dtype)
                )
        return num_grads_clipped

    def get_post_step_metrics(
        self,
        module: nn.Module,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, torch.Tensor]:
        del module, process_group
        return {}

    def get_state_for_param(
        self, param: nn.Parameter
    ) -> Dict[str, Optional[torch.Tensor]]:
        del param
        return {}


class LionW(Optimizer):
    """
    Adapted from https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        record_update_metrics: bool = False,
        selective_updates: bool = False,
        device: Optional[torch.device] = None,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(
            params,
            defaults,
            record_update_metrics=record_update_metrics,
            selective_updates=selective_updates,
        )
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]
        self._update_total_dot_prod: Optional[torch.Tensor] = None
        self._update_total_norm: Optional[torch.Tensor] = None
        self._signed_update_total_norm: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = device

    def get_post_step_metrics(
        self,
        module: nn.Module,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(
            module, FSDP
        ), "`get_post_step_metrics` expects module to be FSDP and will not work with other `distributed_strategy`."

        update_total_dot_prod = self._update_total_dot_prod
        update_total_norm = self._update_total_norm
        signed_update_total_norm = self._signed_update_total_norm
        if (
            update_total_dot_prod is None
            or update_total_norm is None
            or signed_update_total_norm is None
        ):
            return {}

        self._update_total_dot_prod = None
        self._update_total_norm = None
        self._signed_update_total_norm = None

        if is_distributed() and isinstance(
            module, FullyShardedDataParallel
        ):
            # Reduce total dot prod and norms across all ranks.
            update_total_norm = update_total_norm**2.0
            signed_update_total_norm = signed_update_total_norm**2.0
            # Reduce all together to avoid multiple communication calls.
            all_together = torch.stack(
                [
                    update_total_dot_prod,
                    update_total_norm,
                    signed_update_total_norm,
                ]
            )
            # Only need the final result on rank0, since that's where we log from.
            dist.reduce(
                all_together,
                (
                    0
                    if process_group is None
                    else dist.get_global_rank(process_group, 0)
                ),
                group=process_group,
            )
            (
                update_total_dot_prod,
                update_total_norm,
                signed_update_total_norm,
            ) = all_together
            update_total_norm = update_total_norm**0.5
            signed_update_total_norm = signed_update_total_norm**0.5

        update_cos_sim = update_total_dot_prod / torch.max(
            update_total_norm * signed_update_total_norm,
            torch.tensor(
                1e-8,
                device=(
                    get_default_device()
                    if self._device is None
                    else self._device
                ),
            ),
        )
        return {"update_cos_sim": update_cos_sim}

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        update_total_dot_prod: Optional[torch.Tensor] = None
        update_norms: Optional[List[torch.Tensor]] = None
        signed_update_norms: Optional[List[torch.Tensor]] = None
        if self._collecting_metrics and self._record_update_metrics:
            update_total_dot_prod = torch.tensor(0.0, dtype=torch.float32)
            update_norms = []
            signed_update_norms = []

        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]

                # Perform step weight decay
                mask: Union[torch.Tensor, int] = (
                    grad != 0 if self._selective_updates else 1
                )
                p.data.mul_(
                    1 - mask * (group["lr"] * group["weight_decay"])
                )

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                if isinstance(mask, torch.Tensor):
                    # When mask isn't a tensor it's just a literal `1` (python int), so there's
                    # no point in calling this op.
                    update.mul_(mask)
                signed_update = torch.sign(update)
                p.add_(signed_update, alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(1 - mask * (1 - beta2)).add_(
                    grad, alpha=1 - beta2
                )

                # Track dot product and norms of update vs signed update in order to calculate
                # their cosine similarity.
                if (
                    update_total_dot_prod is not None
                    and update_norms is not None
                    and signed_update_norms is not None
                ):
                    update_total_dot_prod = update_total_dot_prod.to(
                        update.device
                    )
                    update_total_dot_prod += torch.tensordot(
                        update, signed_update, dims=len(update.shape)
                    )
                    update_norms.append(
                        torch.linalg.vector_norm(
                            update, 2.0, dtype=torch.float32
                        )
                    )
                    signed_update_norms.append(
                        torch.linalg.vector_norm(
                            signed_update, 2.0, dtype=torch.float32
                        )
                    )

        # Compute cosine similarity between update and signed update.
        if (
            update_total_dot_prod is not None
            and update_norms is not None
            and signed_update_norms is not None
        ):
            device = (
                get_default_device()
                if self._device is None
                else self._device
            )
            self._update_total_dot_prod = update_total_dot_prod.to(device)
            self._update_total_norm = torch.linalg.vector_norm(
                torch.stack(update_norms),
                2.0,
                dtype=torch.float32,
            ).to(device)
            self._signed_update_total_norm = torch.linalg.vector_norm(
                torch.stack(signed_update_norms),
                2.0,
                dtype=torch.float32,
            ).to(device)

    def _compute_spike_metric_for_optimizer(
        self, param: torch.nn.Parameter, state: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """
        Compute spike metric for LionW optimizer.
        For LionW, we use grad^2 / exp_avg as a proxy since there's no second moment.
        """
        if param.grad is None:
            return None

        exp_avg = state.get("exp_avg")
        if exp_avg is None:
            return None

        grad_squared = param.grad**2
        # Use exp_avg as a proxy for second moment
        exp_avg_abs = exp_avg.abs()
        spike_metric = grad_squared / (exp_avg_abs + 1e-8)
        return spike_metric


class AdamW(torch.optim.AdamW, Optimizer):
    def __init__(
        self,
        *args,
        record_update_metrics: bool = False,
        selective_updates: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Need to set these here just like in our base `Optimizer` class since our `Optimizer.__init__`
        # won't be called.
        self._record_update_metrics = record_update_metrics
        self._collecting_metrics = False
        self._selective_updates = selective_updates

        self._step_size_param_names: Optional[List[str]] = None
        self._step_size_norms: Optional[List[torch.Tensor]] = None
        self._step_size_maxs: Optional[List[torch.Tensor]] = None

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if (
            not (self._record_update_metrics and self._collecting_metrics)
            and not self._selective_updates
        ):
            return super().step(closure=closure)

        device = get_default_device()
        param_names = []
        step_size_norms = []
        step_size_maxs = []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            amsgrad = group["amsgrad"]
            for name, param in zip(group["param_names"], group["params"]):
                name = self._clean_param_name(name)
                param_names.append(name)
                grad = param.grad
                if grad is None:
                    step_size_norms.append(
                        torch.tensor([0.0], device=device)
                    )
                    step_size_maxs.append(
                        torch.tensor([0.0], device=device)
                    )
                    continue

                state = self.state[param]
                # init state if needed
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros(
                            (), dtype=torch.float32, device=param.device
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0, dtype=torch.float32)
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step_t = state["step"]

                # Update step.
                step_t += 1

                # Perform step weight decay.
                mask: Union[torch.Tensor, int] = (
                    grad != 0 if self._selective_updates else 1
                )
                param.mul_(1 - mask * (lr * weight_decay))

                # Decay the first and second moment running average coefficient.
                exp_avg.lerp_(grad, mask * (1 - beta1))
                exp_avg_sq.mul_(1 - mask * (1 - beta2)).addcmul_(
                    grad, grad, value=1 - beta2
                )

                step = step_t.item()

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = sqrt(bias_correction2)

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(
                        max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq
                    )

                    # Use the max. for normalizing running avg. of gradient
                    denom = (
                        max_exp_avg_sq.sqrt() / bias_correction2_sqrt
                    ).add_(eps)
                else:
                    denom = (
                        exp_avg_sq.sqrt() / bias_correction2_sqrt
                    ).add_(eps)

                update = -step_size * torch.div(exp_avg, denom)
                if isinstance(mask, torch.Tensor):
                    # When mask isn't a tensor it's just a literal `1` (python int), so there's
                    # no point in calling this op.
                    update.mul_(mask)
                param.add_(update)
                step_size_norms.append(
                    torch.linalg.vector_norm(
                        update, 2.0, dtype=torch.float32
                    ).unsqueeze(0)
                )
                step_size_maxs.append(update.abs().max().unsqueeze(0))

        self._step_size_param_names = param_names
        self._step_size_norms = step_size_norms
        self._step_size_maxs = step_size_maxs

    def get_state_for_param(
        self, param: nn.Parameter
    ) -> Dict[str, Optional[torch.Tensor]]:
        return {key: self.state[param].get(key) for key in ("exp_avg", "exp_avg_sq")}  # type: ignore

    def get_post_step_metrics(
        self,
        module: nn.Module,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, torch.Tensor]:
        if not (self._record_update_metrics and self._collecting_metrics):
            return {}
        else:
            device = get_default_device()
            dst_rank = 0
            if process_group is not None:
                dst_rank = dist.get_global_rank(process_group, 0)
            param_names = self._step_size_param_names
            step_size_norms = self._step_size_norms
            step_size_maxs = self._step_size_maxs
            assert param_names is not None
            assert step_size_norms is not None
            assert step_size_maxs is not None

            # Reduce metrics if needed.
            if is_distributed() and isinstance(
                module, FullyShardedDataParallel
            ):
                # Reduce norms.
                all_norms = torch.cat(step_size_norms).to(device) ** 2.0
                dist.reduce(
                    all_norms,
                    dst_rank,
                    op=dist.ReduceOp.SUM,
                    group=process_group,
                )
                step_size_norms = (all_norms ** (0.5)).squeeze(0).split(1)

                # Reduce maxs.
                all_maxs = torch.cat(step_size_maxs).to(device)
                dist.reduce(
                    all_maxs,
                    dst_rank,
                    op=dist.ReduceOp.MAX,
                    group=process_group,
                )
                step_size_maxs = all_maxs.split(1)

            metrics = {}
            for param_name, step_size_norm, step_size_max in zip(param_names, step_size_norms, step_size_maxs):  # type: ignore[arg-type]
                metrics[f"step/{param_name}.norm"] = step_size_norm.squeeze(
                    0
                )
                metrics[f"step/{param_name}.max"] = step_size_max.squeeze(0)

            self._step_size_param_names = None
            self._step_size_norms = None
            self._step_size_maxs = None
            return metrics


class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.5, last_epoch=-1):
        self.sgd = torch.optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]),
            lr=death_rate,
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.sgd, T_max + 1, eta_min, last_epoch
        )
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, current_step):
        self.cosine_stepper.step(current_step)

    def get_dr(self, current_step):
        self.step(current_step)
        return self.sgd.param_groups[0]["lr"]


class StableSPAM(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        gamma1: float = 0.7,
        gamma2: float = 0.9,
        gamma3: float = 0.999,
        total_T: Optional[int] = None,
        eta_min: float = 0.5,
        update_proj_gap: int = 1000,
        record_update_metrics: bool = False,
        selective_updates: bool = False,
    ):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )

        # Fix: Move the assert before super().__init__()
        assert (
            not selective_updates
        ), "selective_updates is not supported for StableSPAM"

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            gamma1=gamma1,
            gamma2=gamma2,
            gamma3=gamma3,
            total_T=total_T,
            eta_min=eta_min,
            update_proj_gap=update_proj_gap,
        )
        super().__init__(
            params,
            defaults,
            record_update_metrics=record_update_metrics,
            selective_updates=selective_updates,
        )

        self.gamma1 = gamma1  # 0.85 & 0.5 & 0.8,0.9
        self.gamma2 = gamma2  # 0.99999 # 0.999,0.9999
        self.theta = gamma3  # 0.999
        self.total_T = total_T
        if self.total_T is not None:
            self.warmup = CosineDecay(
                1.0, total_T, eta_min=eta_min
            )  # total_T is the totoal number of update steps
        self.total_steps = 0

        if self.gamma1 == -1:
            self.gamma1 = betas[0]
        self.update_proj_gap = update_proj_gap
        self._device = get_default_device()

        self._step_size_param_names: Optional[List[str]] = None
        self._step_size_norms: Optional[List[torch.Tensor]] = None
        self._step_size_maxs: Optional[List[torch.Tensor]] = None

    def __setstate__(self, state):
        super(StableSPAM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None) -> None:

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.total_steps += 1
        if self.total_T is not None:
            scale = self.warmup.get_dr(self.total_steps)
        else:
            scale = 1.0
        # print("scales:",scale,self.update_proj_gap)
        for group in self.param_groups:

            # if "rank" in group:
            #     self.update_proj_gap=group["update_proj_gap"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Perform optimization step
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if "exp_avg" not in state:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["m_norm_t"] = 0
                    state["v_norm_t"] = 0
                    state["m_max_t"] = 0
                    # state['m_min_t']=0
                    # # state["c_norm_t"]=0

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                max_gradient = torch.max(grad.abs())
                # min_gradient=torch.min(grad)
                m_max_t = state["m_max_t"]
                # m_min_t=state['m_min_t']

                state["step"] += 1

                m_max_t = (
                    self.theta * m_max_t + (1 - self.theta) * max_gradient
                )
                # m_min_t = self.theta* m_min_t + (1 - self.theta) * min_gradient

                m_max_hat = m_max_t / (1 - self.theta ** state["step"])
                # m_min_hat = m_min_t / (1 - self.theta**state['step'])

                mask = grad.abs() > m_max_hat
                # mask_neg=grad<m_min_hat
                if mask.sum() > 0:
                    grad[mask] = grad[mask] / max_gradient * m_max_hat

                state["m_max_t"] = m_max_t
                # state["m_min_t"]=m_min_t
                # ###### clipping
                grad_norm = torch.norm(grad)
                ####norm scaling
                m_norm_t, v_norm_t = state["m_norm_t"], state["v_norm_t"]
                # print("m_norm_t",m_norm_t,grad_norm)
                m_norm_t = (
                    self.gamma1 * scale * m_norm_t
                    + (1 - self.gamma1 * scale) * grad_norm
                )

                v_norm_t = (
                    self.gamma2 * v_norm_t
                    + (1 - self.gamma2) * grad_norm**2
                )

                m_norm_hat = m_norm_t / (
                    1 - (self.gamma1 * scale) ** state["step"]
                )
                v_norm_hat = v_norm_t / (1 - self.gamma2 ** state["step"])

                c_norm_t = m_norm_hat / (
                    torch.sqrt(v_norm_hat) + group["eps"]
                )
                # print("grad_nrom",grad_norm,"c_norm",c_norm_t,"st",s_t,m_norm_t)
                if grad_norm > 0:
                    grad = grad / grad_norm * c_norm_t

                # print(m_norm_t)
                state["m_norm_t"], state["v_norm_t"] = m_norm_t, v_norm_t

                ###############################norm scaling end#########################
                if self.update_proj_gap != 0:
                    if self.total_steps % self.update_proj_gap == 0:
                        state["exp_avg"] = torch.zeros_like(grad)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                        state["step"] = 1

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]
                beta1 = beta1 * scale

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(
                        max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq
                    )
                    # Use the max. for normalizing running avg. of gradient
                    denom = (
                        max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                    ).add_(group["eps"])
                else:
                    denom = (
                        exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                    ).add_(group["eps"])

                step_size = group["lr"] / bias_correction1

                norm_grad = exp_avg / denom

                # else:
                grad = norm_grad
                p.add_(grad, alpha=-step_size)
        return loss

    def get_state_for_param(
        self, param: nn.Parameter
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Return the optimizer state for a parameter, including StableSPAM-specific state variables."""
        state = self.state[param]
        return {
            "exp_avg": state.get("exp_avg"),
            "exp_avg_sq": state.get("exp_avg_sq"),
            "m_norm_t": state.get("m_norm_t"),
            "v_norm_t": state.get("v_norm_t"),
            "m_max_t": state.get("m_max_t"),
        }

    def _compute_spike_metric_for_optimizer(
        self, param: torch.nn.Parameter, state: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """
        Compute spike metric for StableSPAM optimizer.
        Uses grad^2 / exp_avg_sq like AdamW since StableSPAM has second moment.
        """
        if param.grad is None:
            return None

        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg_sq is None:
            return None

        grad_squared = param.grad**2
        spike_metric = grad_squared / (exp_avg_sq + 1e-8)
        return spike_metric

    def get_post_step_metrics(
        self,
        module: nn.Module,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, torch.Tensor]:
        """Collect post-step metrics for logging, similar to AdamW implementation."""
        return {}
        # if not (self._record_update_metrics and self._collecting_metrics):
        #     return {}
        # else:
        #     device = get_default_device() if self._device is None else self._device
        #     dst_rank = 0
        #     if process_group is not None:
        #         dst_rank = dist.get_global_rank(process_group, 0)

        #     param_names = self._step_size_param_names
        #     step_size_norms = self._step_size_norms
        #     step_size_maxs = self._step_size_maxs

        #     assert param_names is not None
        #     assert step_size_norms is not None
        #     assert step_size_maxs is not None

        #     # Reduce metrics if needed (for distributed training).
        #     if is_distributed() and isinstance(module, FullyShardedDataParallel):
        #         # Reduce norms.
        #         all_norms = torch.cat(step_size_norms).to(device) ** 2.0
        #         dist.reduce(all_norms, dst_rank, op=dist.ReduceOp.SUM, group=process_group)
        #         step_size_norms = (all_norms ** (0.5)).squeeze(0).split(1)

        #         # Reduce maxs.
        #         all_maxs = torch.cat(step_size_maxs).to(device)
        #         dist.reduce(all_maxs, dst_rank, op=dist.ReduceOp.MAX, group=process_group)
        #         step_size_maxs = all_maxs.split(1)

        #     # Collect metrics.
        #     metrics = {}
        #     for param_name, step_size_norm, step_size_max in zip(param_names, step_size_norms, step_size_maxs):  # type: ignore[arg-type]
        #         metrics[f"step/{param_name}.norm"] = step_size_norm.squeeze(0)
        #         metrics[f"step/{param_name}.max"] = step_size_max.squeeze(0)

        #     # Reset stored metrics.
        #     self._step_size_param_names = None
        #     self._step_size_norms = None
        #     self._step_size_maxs = None

        #     return metrics


@dataclass
class Scheduler(metaclass=ABCMeta):
    # NOTE: these fields are not given default values because otherwise dataclasses complains
    # about how the scheduler subclasses are defined.
    grad_clip_warmup_steps: Optional[int]
    grad_clip_warmup_factor: Optional[float]
    warmup_min_lr: Optional[float]

    @abstractmethod
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        raise NotImplementedError

    def _get_max_grad_norm_coeff(
        self, initial_value: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        del max_steps  # might need this in the future, but for now I just wanted to match the API of `get_lr()`.
        if initial_value is None:
            return None
        elif (
            self.grad_clip_warmup_steps is None
            or self.grad_clip_warmup_factor is None
            or step > self.grad_clip_warmup_steps
        ):
            return initial_value
        else:
            return self.grad_clip_warmup_factor * initial_value

    def get_max_grad_norm(
        self,
        initial_max_grad_norm: Optional[float],
        step: int,
        max_steps: int,
    ) -> Optional[float]:
        return self._get_max_grad_norm_coeff(
            initial_max_grad_norm, step, max_steps
        )

    def get_max_grad_norm_ratio(
        self,
        initial_max_grad_norm_ratio: Optional[float],
        step: int,
        max_steps: int,
    ) -> Optional[float]:
        return self._get_max_grad_norm_coeff(
            initial_max_grad_norm_ratio, step, max_steps
        )

    def _linear_warmup(
        self, initial_lr: float, step: int, warmup_steps: int = 2000
    ) -> float:
        warmup_min_lr = (
            self.warmup_min_lr
            if self.warmup_min_lr is not None
            else initial_lr * 0.10
        )
        assert 0 <= warmup_min_lr < initial_lr
        return (
            warmup_min_lr
            + (initial_lr - warmup_min_lr)
            * min(step, warmup_steps)
            / warmup_steps
        )


@dataclass
class CosWithWarmup(Scheduler):
    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            return (
                eta_min
                + (initial_lr - eta_min)
                * (1 + cos(pi * step / max_steps))
                / 2
            )


@dataclass
class LinearWithWarmup(Scheduler):
    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            return initial_lr - (initial_lr - eta_min) * (step / max_steps)


@dataclass
class WSD(Scheduler):
    """
    Warmup-stable-decay scheduler
    """

    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    decay: Optional[int] = None
    decay_steps: Optional[int] = None  # deprecated, use 'decay' instead.
    decay_fraction: Optional[float] = 0.1
    warmup_min_lr: float = 0.0
    decay_min_lr: float = 0.0

    def __post_init__(self):
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError(
                "Either 'warmup_fraction' or 'warmup' must be specified."
            )

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError(
                "warmup_fraction must be between 0 and 1."
            )

        if self.decay is None and self.decay_steps is not None:
            self.decay = self.decay_steps
            self.decay_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.decay_steps' is deprecated, please use '.decay' instead.",
                DeprecationWarning,
            )

        if (self.decay_fraction is None) == (self.decay is None):
            raise OLMoConfigurationError(
                "Either 'decay_fraction' or 'decay' must be specified. Never both."
            )

        if self.decay_fraction is not None and (
            self.decay_fraction < 0 or self.decay_fraction > 1
        ):
            raise OLMoConfigurationError(
                "decay_fraction must be between 0 and 1."
            )

    def get_lr(
        self,
        initial_lr: Union[float, torch.Tensor],
        step: int,
        max_steps: int,
    ) -> float:
        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(max_steps * self.warmup_fraction)
        else:
            warmup = self.warmup

        if step <= warmup:
            return _linear_warmup(
                initial_lr, step, warmup, self.warmup_min_lr
            )

        if self.decay is None:
            assert self.decay_fraction is not None
            decay = round(max_steps * self.decay_fraction)
        else:
            decay = self.decay

        if step >= max_steps - decay:
            return _linear_decay(
                initial_lr, max_steps - step, decay, self.decay_min_lr
            )

        return initial_lr


@dataclass
class InvSqrtWithWarmup(Scheduler):
    warmup_steps: int

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        del max_steps
        return initial_lr * sqrt(
            self.warmup_steps / max(self.warmup_steps, step)
        )


@dataclass
class MaxScheduler(Scheduler):
    sched1: Scheduler
    sched2: Scheduler

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        return max(
            self.sched1.get_lr(initial_lr, step, max_steps),
            self.sched2.get_lr(initial_lr, step, max_steps),
        )


@dataclass
class BoltOnWarmupScheduler(Scheduler):
    inner: Scheduler
    warmup_start: int
    warmup_end: int

    @classmethod
    def wrap(
        cls, scheduler: Scheduler, warmup_start: int, warmup_end: int
    ) -> "BoltOnWarmupScheduler":
        return cls(
            grad_clip_warmup_steps=None,
            grad_clip_warmup_factor=None,
            inner=scheduler,
            warmup_start=warmup_start,
            warmup_end=warmup_end,
            warmup_min_lr=None,
        )

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        if step < self.warmup_start:
            return 0.0
        if step < self.warmup_end:
            lr_at_intercept = self.inner.get_lr(
                initial_lr, self.warmup_end, max_steps
            )
            return (
                lr_at_intercept
                * (step - self.warmup_start)
                / (self.warmup_end - self.warmup_start)
            )
        else:
            return self.inner.get_lr(initial_lr, step, max_steps)

    def _get_max_grad_norm_coeff(
        self, initial_value: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self.inner._get_max_grad_norm_coeff(
            initial_value, step, max_steps
        )


@dataclass
class ConstantScheduler(Scheduler):
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        del step, max_steps
        return initial_lr


@dataclass
class CosLinearEnvelope(Scheduler):
    "Pointwise product of cosine schedule and linear decay; useful during annealing."

    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        if step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            linear_envelope = 1 - (step / max_steps)
            cosine_schedule = (
                (initial_lr - eta_min)
                * (1 + cos(pi * step / max_steps))
                / 2
            )
            return eta_min + linear_envelope * cosine_schedule


@dataclass
class ConstantWithWarmupScheduler(Scheduler):
    warmup_steps: int

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        del max_steps
        return initial_lr


PARAM_GROUP_FIELDS = (
    "sharded",
    "max_grad_norm",
    "max_grad_norm_ratio",
    "param_names",
)


def get_param_groups(
    cfg: TrainConfig, model: nn.Module
) -> List[Dict[str, Any]]:
    """
    Separate parameters into weight decay and non weight decay groups.
    """
    param_groups: List[Dict[str, Any]]
    param_group_defaults = {
        "sharded": isinstance(model, FullyShardedDataParallel),
        "max_grad_norm": cfg.max_grad_norm,
        "max_grad_norm_ratio": cfg.max_grad_norm_ratio,
        "adaptive_gradient_clipping": cfg.adaptive_gradient_clipping,
    }

    # Separate out parameters that we don't want to apply weight decay to, like norms and biases.
    decay = set()
    no_decay = set()
    all_params = {}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            # NOTE: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times, but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if not p.requires_grad:
                continue

            fpn = f"{mn}.{pn}" if mn else pn
            all_params[fpn] = p

            if pn.endswith("bias"):
                if (
                    cfg.optimizer.decay_norm_and_bias
                    and fpn not in no_decay
                ):
                    decay.add(fpn)
                elif fpn not in decay:
                    no_decay.add(fpn)
            elif (
                pn.endswith("weight")
                and isinstance(m, nn.Linear)
                and fpn not in no_decay
            ):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(
                m, (LayerNormBase, nn.LayerNorm)
            ):
                if (
                    cfg.optimizer.decay_norm_and_bias
                    and fpn not in no_decay
                ):
                    decay.add(fpn)
                elif fpn not in decay:
                    no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, nn.Embedding):
                if cfg.optimizer.decay_embeddings and fpn not in no_decay:
                    decay.add(fpn)
                elif fpn not in decay:
                    no_decay.add(fpn)
            else:
                if (
                    "bias" in pn
                    and not cfg.optimizer.decay_norm_and_bias
                    and fpn not in decay
                ):
                    no_decay.add(fpn)
                elif fpn not in no_decay:
                    decay.add(fpn)

    # Validate that we've considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(all_params.keys() - union_params) == 0
    ), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

    # Create the pytorch optimizer groups.
    decay_sorted = sorted(list(decay))
    no_decay_sorted = sorted(list(no_decay))
    param_groups = []
    if len(decay_sorted) > 0:
        param_groups.append(
            {
                "params": [all_params[pn] for pn in decay_sorted],
                "param_names": decay_sorted,
                **param_group_defaults,
            }
        )
    if len(no_decay_sorted) > 0:
        param_groups.append(
            {
                "params": [all_params[pn] for pn in no_decay_sorted],
                "param_names": no_decay_sorted,
                "weight_decay": 0.0,
                **param_group_defaults,
            }
        )

    # Validate fields.
    for group in param_groups:
        for key in PARAM_GROUP_FIELDS:
            assert key in group

    return param_groups


def fix_optim_state_dict(
    optimizer: Optimizer, state_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Make sure old optim state dicts are compatible with new versions.
    """
    if (
        len(state_dict["param_groups"]) == 1
        and len(optimizer.param_groups) == 2
    ):
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

        # Decay
        decay_param_group = {
            k: v
            for k, v in state_dict["param_groups"][0].items()
            if k != "params"
        }
        decay_param_group["params"] = optimizer.state_dict()[
            "param_groups"
        ][0]["params"]

        # No decay.
        no_decay_param_group = {
            k: v
            for k, v in state_dict["param_groups"][0].items()
            if k != "params"
        }
        no_decay_param_group["weight_decay"] = 0.0
        no_decay_param_group["params"] = optimizer.state_dict()[
            "param_groups"
        ][1]["params"]

        state_dict["param_groups"] = [
            decay_param_group,
            no_decay_param_group,
        ]

    assert len(optimizer.param_groups) == len(state_dict["param_groups"])

    # Make sure:
    #  - All required fields are included in the state dict,
    #  - And that the values of those fields doesn't change from what's currently set in the optimizer,
    #    since we might have changed those fields on purpose after a restart.
    for group, sd_group in zip(
        optimizer.param_groups, state_dict["param_groups"]
    ):
        for key in PARAM_GROUP_FIELDS:
            sd_group[key] = group[key]

    return state_dict


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> Optimizer:
    param_groups = get_param_groups(cfg, model)
    log.info(
        f"Constructing optimizer with {len(param_groups)} param groups"
    )

    optimizer = None
    if cfg.optimizer.name == OptimizerType.lionw:
        optimizer = LionW(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
            record_update_metrics=cfg.optimizer.record_update_metrics,
            selective_updates=cfg.optimizer.selective_updates,
        )
    elif cfg.optimizer.name == OptimizerType.adamw:
        optimizer = AdamW(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
            record_update_metrics=cfg.optimizer.record_update_metrics,
            selective_updates=cfg.optimizer.selective_updates,
            eps=cfg.optimizer.eps,
        )
    elif cfg.optimizer.name == OptimizerType.stablespam:
        optimizer = StableSPAM(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
            gamma1=cfg.optimizer.gamma1,
            gamma2=cfg.optimizer.gamma2,
            gamma3=cfg.optimizer.gamma3,
            total_T=cfg.optimizer.total_T,
            eta_min=cfg.optimizer.eta_min,
            update_proj_gap=cfg.optimizer.update_proj_gap,
            record_update_metrics=cfg.optimizer.record_update_metrics,
        )
    else:
        raise NotImplementedError

    # Configure spike detection if enabled
    if cfg.optimizer.spike_detection:
        optimizer.set_spike_threshold(cfg.optimizer.spike_threshold)
        log.info(
            f"Spike detection configured with threshold: {cfg.optimizer.spike_threshold}"
        )

    return optimizer


def build_scheduler(
    cfg: TrainConfig, sched_cfg: Optional[SchedulerConfig] = None
) -> Scheduler:
    sched_cfg = sched_cfg if sched_cfg is not None else cfg.scheduler
    if sched_cfg.name == SchedulerType.cosine_with_warmup:
        return CosWithWarmup(
            grad_clip_warmup_steps=(
                None
                if sched_cfg.grad_clip_warmup_steps is None
                else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            alpha_f=sched_cfg.alpha_f,
            t_max=None if sched_cfg.t_max is None else int(sched_cfg.t_max),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.linear_with_warmup:
        return LinearWithWarmup(
            grad_clip_warmup_steps=(
                None
                if sched_cfg.grad_clip_warmup_steps is None
                else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            alpha_f=sched_cfg.alpha_f,
            t_max=None if sched_cfg.t_max is None else int(sched_cfg.t_max),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.inverse_sqrt_with_warmup:
        return InvSqrtWithWarmup(
            grad_clip_warmup_steps=(
                None
                if sched_cfg.grad_clip_warmup_steps is None
                else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.max_scheduler:
        return MaxScheduler(
            grad_clip_warmup_steps=(
                None
                if sched_cfg.grad_clip_warmup_steps is None
                else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            sched1=build_scheduler(
                cfg,
                replace(sched_cfg, name=SchedulerType.cosine_with_warmup),
            ),
            sched2=build_scheduler(
                cfg,
                replace(
                    sched_cfg, name=SchedulerType.inverse_sqrt_with_warmup
                ),
            ),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.constant:
        return ConstantScheduler(
            grad_clip_warmup_steps=(
                None
                if sched_cfg.grad_clip_warmup_steps is None
                else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.cosine_linear_envelope:
        return CosLinearEnvelope(
            grad_clip_warmup_steps=(
                None
                if sched_cfg.grad_clip_warmup_steps is None
                else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            alpha_f=sched_cfg.alpha_f,
            t_max=None if sched_cfg.t_max is None else int(sched_cfg.t_max),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.constant_with_warmup:
        return ConstantWithWarmupScheduler(
            grad_clip_warmup_steps=(
                None
                if sched_cfg.grad_clip_warmup_steps is None
                else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_min_lr=sched_cfg.warmup_min_lr,
            warmup_steps=int(sched_cfg.t_warmup),
        )
    elif sched_cfg.name == SchedulerType.wsd:
        return WSD(
            grad_clip_warmup_steps=(
                None
                if sched_cfg.grad_clip_warmup_steps is None
                else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_min_lr=float(sched_cfg.warmup_min_lr),
            decay_min_lr=float(sched_cfg.decay_min_lr),
            warmup_steps=int(sched_cfg.t_warmup),
            decay_fraction=float(sched_cfg.decay_fraction),
        )
    else:
        raise NotImplementedError


def _linear_decay(
    initial_lr: Union[float, torch.Tensor],
    step_from_end: int,
    decay: int,
    decay_min_lr: float = 0.0,
) -> float:
    if isinstance(
        initial_lr, float
    ):  # not worth the potential host-device sync if it's a tensor
        assert 0 <= decay_min_lr < initial_lr

    return (
        decay_min_lr
        + (initial_lr - decay_min_lr) * min(step_from_end, decay) / decay
    )


def _linear_warmup(
    initial_lr: Union[float, torch.Tensor],
    current: int,
    warmup: int,
    warmup_min_lr: float = 0.0,
) -> float:
    if isinstance(
        initial_lr, float
    ):  # not worth the potential host-device sync if it's a tensor
        assert 0 <= warmup_min_lr < initial_lr
    return (
        warmup_min_lr
        + (initial_lr - warmup_min_lr) * min(current, warmup) / warmup
    )
