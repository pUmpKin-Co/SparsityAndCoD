from __future__ import annotations

import cProfile
import functools
import gc
import json
import logging
import math
import os
import random
import shutil
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from pstats import SortKey
from typing import Any, Callable, Deque, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import swanlab
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils
import torch.utils.hooks
import wandb
from olmo_core.train.common import ReduceType
from packaging import version
from torch._C._distributed_c10d import ReduceOp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .aliases import PathOrStr
from .checkpoint import Checkpointer, FullCheckpointer, build_sharded_checkpointer
from .config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    LayerwiseMomentsConfig,
    OutlierDetectionMethod,
    SchedulerUnits,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
    TrainConfig,
)
from .data import IterableDataset
from .eval import Evaluator
from .exceptions import OLMoConfigurationError
from .model import OLMo
from .optim import Optimizer, Scheduler
from .torch_util import (
    SingleAccelerator,
    barrier,
    gc_cuda,
    get_fs_local_rank,
    get_global_rank,
    get_local_rank,
    get_local_tensor,
    get_world_size,
    move_to_device,
    peak_gpu_memory,
    synchronize_flag,
    synchronize_value,
)
from .util import upload

__all__ = ["SpeedMonitor", "LRMonitor", "Trainer"]

log = logging.getLogger(__name__)


@dataclass
class SpeedMonitor:
    cfg: SpeedMonitorConfig
    start_times: Deque[float] = field(default_factory=lambda: deque([]))
    global_total_tokens: int = 0
    total_training_Gflops: float = 0
    device_interval_tokens: Deque[int] = field(default_factory=lambda: deque([]))

    def batch_start(
        self,
        global_total_tokens: int,
        device_batch_num_tokens: int,
        num_fwd_flops: int,
        num_bck_flops: int,
        record: bool = True,
    ) -> None:
        self.global_total_tokens = global_total_tokens
        # num_fwd_flops and num_bck_flops from the OLMo model computes flops per token
        # converting to GFLOPs here prevents numerical issues while logging
        self.total_training_Gflops = (num_fwd_flops + num_bck_flops) * global_total_tokens / 1e9

        if record:
            if len(self.start_times) >= self.cfg.window_size:
                self.start_times.popleft()
                self.device_interval_tokens.popleft()
            self.start_times.append(time.monotonic())
            self.device_interval_tokens.append(device_batch_num_tokens)

    def reset(self) -> None:
        self.start_times.clear()
        self.device_interval_tokens.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"throughput/total_tokens": self.global_total_tokens}

        # plot flops related metrics
        metrics["throughput/total_training_Gflops"] = self.total_training_Gflops
        metrics["throughput/total_training_log_Gflops"] = math.log(self.total_training_Gflops)

        if self.start_times:
            interval_seconds = time.monotonic() - self.start_times[0]
            interval_batches = len(self.start_times)
            interval_tokens = sum(self.device_interval_tokens)
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        lrs = [group["lr"] for group in self.optim.param_groups]
        return {f"optim/learning_rate_group{idx}": lr for idx, lr in enumerate(lrs)}


def cross_entropy_loss(
    logits,
    labels,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss


fused_loss_fn: Optional[Callable]

try:
    import flash_attn
    from flash_attn.ops.triton.cross_entropy import (
        cross_entropy_loss as flash_cross_entropy_loss,  # type: ignore
    )

    def fused_loss_fn(
        logits,
        labels,
        ignore_index: int = -100,
        reduction: str = "mean",
        compute_z_loss: bool = False,
        z_loss_multiplier: float = 1e-4,
    ):
        # The `ignored_index` parameter of `cross_entropy_loss` was changed to `ignore_index` in v2.5.8 with commit https://github.com/Dao-AILab/flash-attention/commit/ec6d22143b5d375e253b2ebfc563b26a43f43684
        ce_loss_use_ignore_index_param = version.parse(flash_attn.__version__) >= version.parse("2.5.8")

        if ce_loss_use_ignore_index_param:
            ignore_index_kwarg = {"ignore_index": ignore_index}
        else:
            ignore_index_kwarg = {"ignored_index": ignore_index}

        loss, z_loss = flash_cross_entropy_loss(
            logits,
            labels,
            label_smoothing=0.0,
            logit_scale=1.0,
            lse_square_scale=z_loss_multiplier,
            inplace_backward=False,
            process_group=None,
            **ignore_index_kwarg,
        )

        mask = labels != ignore_index

        if reduction == "mean":
            loss = loss.sum() / mask.sum()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss

        if not compute_z_loss:
            return loss, None

        if reduction == "mean":
            z_loss = z_loss.sum() / mask.sum()
        elif reduction == "sum":
            z_loss = z_loss.sum()
        else:
            z_loss = z_loss

        return loss, z_loss

except ImportError:
    fused_loss_fn = None


def compute_higher_order_moments(
    tensor: torch.Tensor,
    sample_features: bool = False,
    sample_ratio: float = 0.1,
    compute_per_feature: bool = False,
    compute_channel_moments: bool = True,
) -> Dict[str, Any]:
    """
    Compute higher order moments of a tensor.

    Args:
        tensor: Input tensor of shape (batch_size, seq_len, d_model) or similar
        sample_features: Whether to sample features for efficiency
        sample_ratio: Ratio of features to sample
        compute_per_feature: If True, compute moments for each feature dimension separately
        compute_channel_moments: If True, compute skewness and kurtosis in channel dimension
                                     and average across tokens

    Returns:
        Dictionary containing skewness, kurtosis, and other statistics.
        If compute_per_feature=True, also includes per-feature moment arrays.
        If compute_channel_moments=True, includes channel-wise averaged moments.
    """
    # Flatten the tensor
    flat_tensor = tensor.flatten()

    # Optionally sample features
    if sample_features and flat_tensor.numel() > 10000:
        num_samples = max(1000, int(flat_tensor.numel() * sample_ratio))
        indices = torch.randperm(flat_tensor.numel(), device=tensor.device)[:num_samples]
        flat_tensor = flat_tensor[indices]

    # Convert to float32 for numerical stability
    if flat_tensor.dtype != torch.float32:
        flat_tensor = flat_tensor.float()

    # Compute basic statistics
    mean = flat_tensor.mean().item()
    variance = flat_tensor.var().item()
    std = math.sqrt(variance) if variance > 0 else 1e-8

    # Compute skewness (3rd moment)
    if variance > 0:
        centered = flat_tensor - mean
        skewness = ((centered ** 3).mean().item() / (std ** 3))
    else:
        skewness = 0.0

    # Compute kurtosis (4th moment)
    if variance > 0:
        centered = flat_tensor - mean
        kurtosis = ((centered ** 4).mean().item() / (std ** 4))
        excess_kurtosis = kurtosis - 3.0
    else:
        kurtosis = 0.0
        excess_kurtosis = -3.0

    # Compute robust statistics
    median = flat_tensor.median().item()
    mad = (flat_tensor - median).abs().median().item()

    # Compute coefficient of variation
    cv = std / abs(mean) if abs(mean) > 1e-8 else 0.0

    # Compute IQR
    q25 = flat_tensor.quantile(0.25).item()
    q75 = flat_tensor.quantile(0.75).item()
    iqr = q75 - q25

    result = {
        "mean": mean,
        "variance": variance,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "excess_kurtosis": excess_kurtosis,
        "median": median,
        "mad": mad,
        "cv": cv,
        "iqr": iqr,
        "q25": q25,
        "q75": q75,
    }

    # Compute per-feature moments if requested
    if compute_per_feature:
        # Reshape to (batch_size * seq_len, d_model)
        batch_size, seq_len, d_model = tensor.shape
        reshaped = tensor.reshape(-1, d_model)  # shape: (N, d_model)

        per_feature_moments = {
            'skewness_per_feature': [],
            'kurtosis_per_feature': [],
            'mean_per_feature': [],
            'std_per_feature': [],
        }

        for dim in range(d_model):
            feature_values = reshaped[:, dim]

            # Compute moments for this feature
            feature_mean = feature_values.mean().item()
            feature_std = feature_values.std().item()

            if feature_std > 0:
                centered = feature_values - feature_mean
                feature_skewness = ((centered ** 3).mean().item() / (feature_std ** 3))
                feature_kurtosis = ((centered ** 4).mean().item() / (feature_std ** 4))
            else:
                feature_skewness = 0.0
                feature_kurtosis = 3.0

            per_feature_moments['skewness_per_feature'].append(feature_skewness)
            per_feature_moments['kurtosis_per_feature'].append(feature_kurtosis)
            per_feature_moments['mean_per_feature'].append(feature_mean)
            per_feature_moments['std_per_feature'].append(feature_std)

        # Convert to numpy arrays
        for key in per_feature_moments:
            per_feature_moments[key] = np.array(per_feature_moments[key])

        result.update(per_feature_moments)

    # Compute channel-wise moments (skewness and kurtosis in channel dimension, averaged across tokens)
    if compute_channel_moments:
        # Reshape to (batch_size * seq_len, d_model)
        batch_size, seq_len, d_model = tensor.shape
        reshaped = tensor.reshape(-1, d_model)  # shape: (N, d_model)

        # Convert to float32 for numerical stability
        if reshaped.dtype != torch.float32:
            reshaped = reshaped.float()

        # Compute mean and std for each token across channels
        token_means = reshaped.mean(dim=1)  # shape: (N,)
        token_stds = reshaped.std(dim=1)    # shape: (N,)

        # Compute skewness for each token across channels
        token_skewness = []
        for i in range(reshaped.shape[0]):
            token_values = reshaped[i, :]
            token_mean = token_means[i].item()
            token_std = token_stds[i].item()

            if token_std > 0:
                centered = token_values - token_mean
                token_skew = ((centered ** 3).mean().item() / (token_std ** 3))
            else:
                token_skew = 0.0
            token_skewness.append(token_skew)

        # Compute kurtosis for each token across channels
        token_kurtosis = []
        for i in range(reshaped.shape[0]):
            token_values = reshaped[i, :]
            token_mean = token_means[i].item()
            token_std = token_stds[i].item()

            if token_std > 0:
                centered = token_values - token_mean
                token_kurt = ((centered ** 4).mean().item() / (token_std ** 4))
            else:
                token_kurt = 3.0
            token_kurtosis.append(token_kurt)

        # Average skewness and kurtosis across all tokens
        avg_skewness = np.mean(token_skewness)
        avg_kurtosis = np.mean(token_kurtosis)
        avg_excess_kurtosis = avg_kurtosis - 3.0

        # Also compute std of skewness and kurtosis across tokens
        std_skewness = np.std(token_skewness)
        std_kurtosis = np.std(token_kurtosis)

        result.update({
            'channel_avg_skewness': avg_skewness,
            'channel_avg_kurtosis': avg_kurtosis,
            'channel_avg_excess_kurtosis': avg_excess_kurtosis,
            'channel_std_skewness': std_skewness,
            'channel_std_kurtosis': std_kurtosis,
        })

    return result


def detect_outliers(
    tensor: torch.Tensor,
    method: OutlierDetectionMethod = OutlierDetectionMethod.zscore,
    threshold: float = 3.0,
    percentile: float = 0.01,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Detect outlier features in a tensor.

    Args:
        tensor: Input tensor of shape (batch_size, seq_len, d_model) or similar
        method: Outlier detection method ('zscore', 'quantile', or 'topk')
        threshold: Threshold for z-score detection
        percentile: Percentile for quantile detection
        top_k: Number of top features for top-k detection

    Returns:
        Dictionary containing outlier statistics
    """
    # Flatten the tensor
    flat_tensor = tensor.flatten()

    # Convert to float32 for numerical stability
    if flat_tensor.dtype != torch.float32:
        flat_tensor = flat_tensor.float()

    # Compute mean and std
    mean = flat_tensor.mean().item()
    std = flat_tensor.std().item()
    std = std if std > 0 else 1e-8

    # Compute absolute values
    abs_tensor = flat_tensor.abs()

    # Compute total norm
    total_norm = flat_tensor.norm().item()

    outlier_stats = {
        "outlier_count": 0,
        "outlier_ratio": 0.0,
        "top_magnitudes": [],
        "top_magnitude_ratio": 0.0,
    }

    if method == OutlierDetectionMethod.zscore:
        # Z-score based detection
        z_scores = ((flat_tensor - mean) / std).abs()
        outlier_mask = z_scores > threshold
        outlier_count = outlier_mask.sum().item()
        outlier_ratio = outlier_count / flat_tensor.numel()

        outlier_stats["outlier_count"] = int(outlier_count)
        outlier_stats["outlier_ratio"] = outlier_ratio

    elif method == OutlierDetectionMethod.quantile:
        # Quantile-based detection
        lower_threshold = flat_tensor.quantile(percentile).item()
        upper_threshold = flat_tensor.quantile(1.0 - percentile).item()
        outlier_mask = (flat_tensor < lower_threshold) | (flat_tensor > upper_threshold)
        outlier_count = outlier_mask.sum().item()
        outlier_ratio = outlier_count / flat_tensor.numel()

        outlier_stats["outlier_count"] = int(outlier_count)
        outlier_stats["outlier_ratio"] = outlier_ratio

    elif method == OutlierDetectionMethod.topk:
        # Top-K magnitude detection
        k = min(top_k, flat_tensor.numel())
        top_k_values, top_k_indices = torch.topk(abs_tensor, k)
        top_magnitudes = top_k_values.cpu().tolist()
        top_magnitude_norm = torch.norm(top_k_values).item()
        top_magnitude_ratio = top_magnitude_norm / total_norm if total_norm > 0 else 0.0

        outlier_stats["outlier_count"] = k
        outlier_stats["outlier_ratio"] = k / flat_tensor.numel()
        outlier_stats["top_magnitudes"] = top_magnitudes
        outlier_stats["top_magnitude_ratio"] = top_magnitude_ratio

    return outlier_stats


@dataclass
class Trainer:
    cfg: TrainConfig
    model: OLMo
    dist_model: Union[DDP, FSDP, SingleAccelerator]
    optim: Optimizer
    scheduler: Scheduler
    train_loader: DataLoader
    device: torch.device
    evaluators: List[Evaluator]
    epoch: Optional[int] = None
    global_step: int = 0
    global_train_examples_seen_this_epoch: int = 0
    """Tracks the global number of training examples seen in the current epoch for the purpose of restoring
    the data loader position on restarts."""
    global_train_tokens_seen: int = 0
    """Tracks the global total number of tokens trained on."""
    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)
    ephemeral_checkpoints: List[Path] = field(default_factory=list)
    min_train_loss: float = float("inf")
    cur_train_loss: float = float("inf")
    indices_file: Optional[TextIO] = None
    _start_time: float = 0.0
    _gc_init_state: bool = True
    loss_fn: Callable[..., torch.Tensor] = field(default_factory=lambda: cross_entropy_loss)  # type: ignore
    last_sharded_checkpoint_step: Optional[int] = None
    last_unsharded_checkpoint_step: Optional[int] = None
    # Spike detection attributes
    spike_detection_enabled: bool = False
    last_spike_plot_step: Optional[int] = None

    def __post_init__(self):
        if self.cfg.fused_loss:
            if fused_loss_fn is not None:
                self.loss_fn = fused_loss_fn
            else:
                raise NameError("`fused_loss_fn` is not defined. Please ensure that `flash_attn` is installed.")

        # Initialize spike detection
        if self.cfg.optimizer.spike_detection:
            self.spike_detection_enabled = True
            self.optim.set_spike_threshold(self.cfg.optimizer.spike_threshold)
            log.info(f"Spike detection enabled with threshold: {self.cfg.optimizer.spike_threshold}")
        else:
            self.spike_detection_enabled = False

        if self.cfg.layerwise_statis_collect_interval is not None:
            self.layerwise_statis_collect_interval = self.cfg.layerwise_statis_collect_interval
            self.layerwise_statis = {}
            for layer_id in range(self.cfg.model.n_layers):
                self.layerwise_statis[f"layer_{layer_id}"] = defaultdict(list)
            # Initialize layerwise stats save path (single JSONL file)
            self.layerwise_stats_save_path = Path(self.cfg.save_folder) / "layerwise_stats.jsonl"
            self.layerwise_stats_save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.layerwise_statis_collect_interval = None
            self.layerwise_stats_save_path = None

        # Initialize layerwise moments collection
        if self.cfg.layerwise_moments_config is not None:
            self.layerwise_moments_config = self.cfg.layerwise_moments_config
            self.layerwise_moments = {}
            for layer_id in range(self.cfg.model.n_layers):
                self.layerwise_moments[f"layer_{layer_id}"] = defaultdict(list)
            # Initialize layerwise moments save path
            self.layerwise_moments_save_path = Path(self.cfg.save_folder) / "layerwise_moments.jsonl"
            self.layerwise_moments_save_path.parent.mkdir(parents=True, exist_ok=True)
            # Flag to track if moments were collected in current step
            self._moments_collected_this_step = False
        else:
            self.layerwise_moments_config = None
            self.layerwise_moments_save_path = None
            self._moments_collected_this_step = False

    @property
    def dataset(self) -> IterableDataset:
        assert isinstance(self.train_loader.dataset, IterableDataset)
        return self.train_loader.dataset

    @property
    def tokens_per_batch(self) -> int:
        return self.cfg.global_train_batch_size * self.cfg.model.max_sequence_length

    @property
    def batches_per_epoch(self) -> int:
        return self.dataset.total_size // self.cfg.global_train_batch_size

    @property
    def max_epochs(self) -> int:
        return math.ceil(self.max_steps / self.batches_per_epoch)

    @property
    def max_steps(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return self.cfg.max_duration
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                max_tokens = int(float(self.cfg.max_duration[:-1].strip()))
                tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
                steps_remaining = math.ceil(tokens_remaining / self.tokens_per_batch)
                return self.global_step + steps_remaining
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch
            else:
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration))
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def max_tokens(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return (
                self.global_train_tokens_seen
                + max(self.cfg.max_duration - self.global_step, 0) * self.tokens_per_batch
            )
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration[:-1].strip()))
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch * self.tokens_per_batch
            else:
                # convert to float *first* to handle scientific notation
                return (
                    self.global_train_tokens_seen
                    + max(
                        int(float(self.cfg.max_duration)) - self.global_step,
                        0,
                    )
                    * self.tokens_per_batch
                )
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def scheduler_current(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.global_step
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.global_train_tokens_seen
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    @property
    def scheduler_max(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.max_steps
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.max_tokens
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    def trainer_state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch or 0,
            "global_step": self.global_step,
            "global_train_examples_seen_this_epoch": self.global_train_examples_seen_this_epoch,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "world_size": get_world_size(),
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "ephemeral_checkpoints": self.ephemeral_checkpoints,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
                "mps": (torch.mps.get_rng_state() if torch.backends.mps.is_available() else None),
            },
        }

    def load_trainer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Checkpoint paths.
        self.checkpoints = [
            path
            for path in state_dict["checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.unsharded_checkpoints = [
            path
            for path in state_dict["unsharded_checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.ephemeral_checkpoints = [
            path
            for path in state_dict.get("ephemeral_checkpoints", [])
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]

        # Dataset / dataloader position.
        checkpoint_epoch = state_dict.get("epoch") or 0
        self.global_step = state_dict["global_step"]
        self.global_train_examples_seen_this_epoch = state_dict.get(
            "global_train_examples_seen_this_epoch",
            state_dict.get(  # for backwards compatibility
                "global_train_examples_seen",
                state_dict.get("global_data_step", self.global_step) * self.cfg.global_train_batch_size,
            ),
        )
        self.global_train_tokens_seen = state_dict.get(
            "global_train_tokens_seen",
            state_dict.get("global_data_step", self.global_step)  # for backwards compatibility
            * self.cfg.global_train_batch_size
            * self.cfg.model.max_sequence_length,
        )

        if not self.cfg.restore_dataloader:
            self.epoch = 0
            self.global_step = 0
            self.global_train_tokens_seen = 0
            self.global_train_examples_seen_this_epoch = 0
        elif self.epoch is None:
            self.epoch = checkpoint_epoch
        elif checkpoint_epoch != self.epoch:
            log.info(f"Starting new epoch (epoch = {self.epoch})")
            self.global_train_examples_seen_this_epoch = 0

        assert self.epoch is not None
        # Reshuffle dataset if needed.
        if self.dataset.epoch != self.epoch:
            log.info(f"Reshuffling data loader for epoch {self.epoch}...")
            self.dataset.reshuffle(self.epoch)

        if self.cfg.fast_forward_batches:
            log.info(f"Fast-forwarding data loader by {self.cfg.fast_forward_batches:,d} steps")
            # Technically we don't "see" these batches that we fast-forward through, but we use
            # this variable to update the position of the dataset so we need to include them here.
            self.global_train_examples_seen_this_epoch += (
                self.cfg.fast_forward_batches * self.cfg.global_train_batch_size
            )
            # NOTE: on the other hand we don't add anything to 'self.global_train_tokens_seen' here because
            # that variable is meant to track the actual number of tokens trained on.

        if self.global_train_examples_seen_this_epoch > 0:
            assert isinstance(self.dataset, IterableDataset)
            log.info(f"Data loader will start at instance index {self.global_train_examples_seen_this_epoch:,d}")
            self.dataset.start_index = self.global_train_examples_seen_this_epoch

        # Reset learning rate and weight decay to the values from the config, not the checkpoint.
        log.info("Resetting learning rate...")
        new_learning_rate = self.scheduler.get_lr(
            self.cfg.optimizer.learning_rate,
            self.scheduler_current,
            self.scheduler_max,
        )
        for group in self.optim.param_groups:
            group["lr"] = new_learning_rate
            group["initial_lr"] = self.cfg.optimizer.learning_rate
            if "weight_decay" in group and group["weight_decay"] > 0.0:
                group["weight_decay"] = self.cfg.optimizer.weight_decay

        # RNG states.
        if "rng" in state_dict and state_dict.get("world_size", get_world_size()) == get_world_size():
            log.info("Restoring RNG states...")
            rng_state = state_dict["rng"]
            self.restore_rng_state(rng_state)
        else:
            log.warning(
                "Trainer will not restore RNG states since the RNG states in the checkpoint are missing or invalid. "
                "This typically happens when restoring from an unsharded checkpoint or a checkpoint that was saved "
                "with a different world size. If that's the case you can safely ignore this warning."
            )

    def restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available():
            if rng_state["cuda"] is not None:
                torch.cuda.set_rng_state(rng_state["cuda"])
            else:
                log.warning("CUDA is available, but no RNG state was provided.")
        if torch.backends.mps.is_available():
            if rng_state["mps"] is not None:
                torch.mps.set_rng_state(rng_state["mps"])
            else:
                log.warning("MPS is available, but no RNG state was provided.")

    def _save_checkpoint(
        self, checkpointer: Checkpointer, checkpoint_type: CheckpointType
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        if checkpoint_type == CheckpointType.sharded:
            suffix = ""
            current_checkpoints = self.checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.unsharded:
            suffix = "-unsharded"
            current_checkpoints = self.unsharded_checkpoints
            link_latest = get_global_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_unsharded_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            suffix = ""
            current_checkpoints = self.ephemeral_checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = 1
        else:
            raise NotImplementedError(checkpoint_type)

        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        # Flush data indices file.
        # TODO: upload the indices files?
        if self.indices_file is not None:
            self.indices_file.flush()

        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}{suffix}"

        log.info(f"Saving checkpoint to {checkpoint_dir}")
        remote_checkpoint_dir: Optional[str] = None
        if self.cfg.remote_save_folder is not None:
            remote_checkpoint_dir = f"{self.cfg.remote_save_folder.rstrip('/')}/{checkpoint_dir.name}"
        current_checkpoints.append(checkpoint_dir)

        # Save the checkpoint.
        try:
            checkpointer.save_checkpoint(
                checkpoint_dir,
                self.dist_model,
                self.optim,
                self.trainer_state_dict(),
                upload_to=remote_checkpoint_dir,
            )

        except FileExistsError:
            raise OLMoConfigurationError(
                f"Checkpoint for step {self.global_step} already exists, use --save_overwrite to overwrite it"
            )

        if link_latest:
            # Link to 'latest'.
            latest_path = Path(self.cfg.save_folder) / f"latest{suffix}"
            latest_path.unlink(missing_ok=True)
            try:
                latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)
                # Remove old checkpoints.
                # For DDP, checkpoint_type being passed to remove_checkpoint is always `unsharded`.
                if num_checkpoints_to_keep > 0:
                    while len(current_checkpoints) > num_checkpoints_to_keep:
                        self.remove_checkpoint(0, checkpoint_type)
            except FileExistsError:
                # Same as above, caught when another (file-system) local rank 0 has already made the 'latest' symlink.
                # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                # file-systems.
                if latest_path.resolve().name != checkpoint_dir.name:
                    raise
            except Exception as e:
                log.error(f"Error linking to 'latest': {e}")

        barrier()

        if remote_checkpoint_dir is not None:
            return remote_checkpoint_dir, checkpoint_dir
        else:
            return checkpoint_dir, None

    def save_sharded_checkpoint(
        self,
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def save_ephemeral_checkpoint(
        self,
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded_ephemeral)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def _remove_sharded_checkpoint(self, idx: int, checkpoints: List[Path]):
        oldest_checkpoint = checkpoints.pop(idx)
        barrier()
        if get_fs_local_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def remove_sharded_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.checkpoints)

    def remove_ephemeral_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.ephemeral_checkpoints)

    def restore_sharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = build_sharded_checkpointer(self.cfg, name=sharded_checkpointer)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.dist_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_unsharded_checkpoint(
        self,
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = FullCheckpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.unsharded)
        self.last_unsharded_checkpoint_step = self.global_step
        return result

    def remove_unsharded_checkpoint(self, idx: int = 0):
        barrier()
        oldest_checkpoint = self.unsharded_checkpoints.pop(idx)
        if get_global_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest-unsharded"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def restore_unsharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = FullCheckpointer(self.cfg)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.dist_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_checkpoint(
        self, checkpoint_type: CheckpointType = CheckpointType.sharded
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        result: Tuple[PathOrStr, Optional[PathOrStr]]
        if checkpoint_type == CheckpointType.sharded:
            result = self.save_sharded_checkpoint()
        elif checkpoint_type == CheckpointType.unsharded:
            result = self.save_unsharded_checkpoint()
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            result = self.save_ephemeral_checkpoint()
        else:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()
        return result

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        checkpoint_type: Optional[CheckpointType] = None,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        if checkpoint_type == CheckpointType.unsharded or (
            checkpoint_type is None and str(load_path).rstrip("/").endswith("-unsharded")
        ):
            self.restore_unsharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
            )
        elif checkpoint_type == CheckpointType.sharded or checkpoint_type is None:
            self.restore_sharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
                sharded_checkpointer=sharded_checkpointer,
            )
        elif checkpoint_type is not None:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()

    def remove_checkpoint(
        self,
        idx: int = 0,
        checkpoint_type: CheckpointType = CheckpointType.sharded,
    ):
        if checkpoint_type == CheckpointType.sharded:
            self.remove_sharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.unsharded:
            self.remove_unsharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            self.remove_ephemeral_checkpoint(idx=idx)
        else:
            raise NotImplementedError(checkpoint_type)

    def _setup_module_output_save_hooks(self, micro_batch_idx: int) -> List[torch.utils.hooks.RemovableHandle]:
        if self.cfg.module_outputs_save_steps is None or self.global_step not in self.cfg.module_outputs_save_steps:
            return []

        if micro_batch_idx != 0 or get_global_rank() != 0:
            # Hook is currently only used on the first microbatch of rank 0
            return []

        trace_save_folder = Path(self.cfg.save_folder) / f"traces/step{self.global_step}"
        if trace_save_folder.exists():
            if self.cfg.save_overwrite:
                shutil.rmtree(trace_save_folder)
            else:
                raise OLMoConfigurationError(
                    f"Attempting to overwrite traces at step {self.global_step} without --save_overwrite"
                )
        trace_save_folder.mkdir(parents=True)

        def trace_outputs_hook(
            module_name: str,
            _: torch.nn.Module,
            args: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if len(args) == 0:
                log.info(
                    "No input args for module %s, output %s",
                    module_name,
                    output,
                )

            module_input = args[0] if len(args) > 0 else torch.tensor(())
            trace_save_folder = Path(self.cfg.save_folder) / f"traces/step{self.global_step}"
            trace_save_folder.mkdir(parents=True, exist_ok=True)

            module_occurence_num = 0
            while (
                module_input_filepath := trace_save_folder / f"{module_name}_{module_occurence_num}_input.pt"
            ).exists():
                module_occurence_num += 1
            torch.save(module_input, module_input_filepath)

            module_output_filepath = trace_save_folder / f"{module_name}_{module_occurence_num}_output.pt"
            torch.save(output, module_output_filepath)

        output_hooks = []
        for module_name, module in self.model.named_modules(prefix="model"):
            output_hooks.append(module.register_forward_hook(functools.partial(trace_outputs_hook, module_name)))

        return output_hooks

    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
        return labels[..., 1:].contiguous()

    def model_forward(
        self,
        batch: Dict[str, Any],
        loss_reduction: str = "mean",
        compute_z_loss: bool = False,
        should_collect_layerwise_statis: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[Dict]]:
        # shape: (batch_size, seq_len, vocab_size)
        model_output = self.dist_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
            doc_lens=batch.get("doc_lens"),
            max_doc_lens=batch.get("max_doc_lens"),
            output_hidden_states=should_collect_layerwise_statis,
            output_sublayer_statis=should_collect_layerwise_statis,
        )

        layerwise_stats = None
        if should_collect_layerwise_statis:
            hidden_states = model_output.hidden_states[1:]
            sublayer_statis = model_output.layers_output

            layerwise_stats = {}
            with torch.no_grad():
                for layer_idx, (hidden_state, sublayer_state) in enumerate(zip(hidden_states, sublayer_statis)):
                    # hidden_state shape: (batch_size, seq_len, d_model)
                    # Compute token-level variance (variance along d_model dimension)
                    # This gives us variance for each token
                    token_variances = hidden_state.var(dim=-1)  # shape: (batch_size, seq_len)

                    # Sum all token variances and count tokens
                    total_variance = token_variances.sum().detach().cpu().item()
                    num_tokens = token_variances.numel()

                    # Average variance per token
                    avg_variance = total_variance / num_tokens if num_tokens > 0 else 0.0

                    attn_out, ffn_out = sublayer_state

                    # Compute variance for attention output
                    attn_out_variances = attn_out.var(dim=-1)  # shape: (batch_size, seq_len)
                    attn_total_variance = attn_out_variances.sum().detach().cpu().item()
                    attn_avg_variance = attn_total_variance / num_tokens if num_tokens > 0 else 0.0

                    # Compute variance for FFN output
                    ffn_out_variances = ffn_out.var(dim=-1)  # shape: (batch_size, seq_len)
                    ffn_total_variance = ffn_out_variances.sum().detach().cpu().item()
                    ffn_avg_variance = ffn_total_variance / num_tokens if num_tokens > 0 else 0.0

                    attn_out_norm = attn_out.norm().detach().cpu().item()
                    ffn_out_norm = ffn_out.norm().detach().cpu().item()

                    layer_data = {
                        "out_var": avg_variance,  # Average variance per token (layer output)
                        "num_tokens": num_tokens,  # Number of tokens in this micro-batch
                        "attn_out_var": attn_avg_variance,  # Average variance per token (attention output)
                        "ffn_out_var": ffn_avg_variance,  # Average variance per token (FFN output)
                        "attn_out_norm": attn_out_norm,
                        "ffn_out_norm": ffn_out_norm,
                    }

                    # Only store tensors if we need them for moment calculation
                    if self.layerwise_moments_config is not None:
                        layer_data["hidden_state"] = hidden_state.detach()  # Decoder block output
                        layer_data["attn_output"] = attn_out.detach()
                        layer_data["ffn_output"] = ffn_out.detach()

                    layerwise_stats[f"layer_{layer_idx}"] = layer_data

        logits = model_output.logits
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels = self.get_labels(batch)
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss, z_loss = self.loss_fn(
            logits_for_loss,
            labels,
            ignore_index=-100,
            reduction=loss_reduction,
            compute_z_loss=compute_z_loss,
            z_loss_multiplier=(1e-4 if self.cfg.model.moe_config is None else self.cfg.model.moe_config.z_loss_weight),
        )
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)
        return ce_loss, z_loss, logits, layerwise_stats

    def train_micro_batch(
        self,
        micro_batch: Dict[str, Any],
        batch_size_in_tokens: int,
        should_collect_layerwise_statis: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        ce_loss, z_loss, logits, layerwise_stats = self.model_forward(
            micro_batch,
            compute_z_loss=self.cfg.softmax_auxiliary_loss,
            loss_reduction="sum",
            should_collect_layerwise_statis=should_collect_layerwise_statis,
        )
        ce_loss = ce_loss / batch_size_in_tokens

        # In case this helps with memory utilization.
        del micro_batch

        # Get loss to optimize for.
        if self.cfg.softmax_auxiliary_loss:
            assert z_loss is not None
            z_loss = z_loss / batch_size_in_tokens
            loss = ce_loss + z_loss
        else:
            loss = ce_loss

        del logits

        return loss, ce_loss, z_loss, layerwise_stats

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)
        batch_size_in_tokens = batch["input_ids"].numel()

        # In case this helps with memory utilization.
        del batch

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        z_batch_loss = None if not self.cfg.softmax_auxiliary_loss else torch.tensor(0.0, device=self.device)
        num_micro_batches = len(micro_batches)

        # Collect layerwise statistics across all micro-batches
        # We'll accumulate token-level variances and token counts
        layerwise_variance_sum = {}
        layerwise_token_count = {}
        layerwise_attn_var_sum = {}
        layerwise_ffn_var_sum = {}
        layerwise_attn_out_norm_sum = {}
        layerwise_ffn_out_norm_sum = {}

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            # setup sync context for DDP for all micro-batches except the last
            grad_sync_context = nullcontext
            if (
                self.cfg.distributed_strategy == DistributedStrategy.ddp
                and self.cfg.ddp is not None
                and self.cfg.ddp.grad_sync_mode == DDPGradSyncMode.batch
            ):
                if micro_batch_idx != num_micro_batches - 1:
                    grad_sync_context = self.dist_model.no_sync

            # Register output hooks
            output_hooks: List[torch.utils.hooks.RemovableHandle] = []
            output_hooks += self._setup_module_output_save_hooks(micro_batch_idx)

            if (
                self.layerwise_statis_collect_interval is not None
                and self.global_step % self.layerwise_statis_collect_interval == 0
            ):
                should_collect_layerwise_statis = True
            elif (
                self.layerwise_moments_config is not None
                and self.global_step % self.layerwise_moments_config.collect_interval == 0
            ):
                should_collect_layerwise_statis = True
            else:
                should_collect_layerwise_statis = False

            with grad_sync_context():
                autocast_device = "mps" if self.device.type == "mps" else "cuda"
                with torch.autocast(
                    autocast_device,
                    enabled=True,
                    dtype=self.cfg.autocast_precision,
                ):
                    # Run forward pass.
                    loss, ce_loss, z_loss, layerwise_stats = self.train_micro_batch(
                        micro_batch,
                        batch_size_in_tokens,
                        should_collect_layerwise_statis,
                    )

                    # Collect layerwise stats if enabled
                    if should_collect_layerwise_statis and layerwise_stats is not None:
                        for layer_name, stats in layerwise_stats.items():
                            # Initialize if first time
                            if layer_name not in layerwise_variance_sum:
                                layerwise_variance_sum[layer_name] = 0.0
                                layerwise_token_count[layer_name] = 0
                                layerwise_attn_var_sum[layer_name] = 0.0
                                layerwise_ffn_var_sum[layer_name] = 0.0
                                layerwise_attn_out_norm_sum[layer_name] = 0.0
                                layerwise_ffn_out_norm_sum[layer_name] = 0.0

                            # Accumulate statistics
                            layerwise_variance_sum[layer_name] += stats["out_var"] * stats["num_tokens"]
                            layerwise_token_count[layer_name] += stats["num_tokens"]
                            layerwise_attn_var_sum[layer_name] += stats["attn_out_var"] * stats["num_tokens"]
                            layerwise_ffn_var_sum[layer_name] += stats["ffn_out_var"] * stats["num_tokens"]
                            layerwise_attn_out_norm_sum[layer_name] += stats["attn_out_norm"]
                            layerwise_ffn_out_norm_sum[layer_name] += stats["ffn_out_norm"]

                    # Update overall CE batch loss.
                    ce_batch_loss += ce_loss.detach()

                    # Update overall Z batch loss.
                    if z_loss is not None:
                        assert z_batch_loss is not None
                        z_batch_loss += z_loss.detach()

                # Run backward pass.
                loss.backward()

            # Remove output hooks
            for hook in output_hooks:
                hook.remove()

        # Average layerwise statistics across all micro-batches
        if (
            self.layerwise_statis_collect_interval is not None
            and self.global_step % self.layerwise_statis_collect_interval == 0
            and len(layerwise_variance_sum) > 0
        ):
            for layer_name in layerwise_variance_sum.keys():
                # Average variance across all tokens in the global batch
                avg_out_var = layerwise_variance_sum[layer_name] / layerwise_token_count[layer_name]
                avg_attn_var = layerwise_attn_var_sum[layer_name] / layerwise_token_count[layer_name]
                avg_ffn_var = layerwise_ffn_var_sum[layer_name] / layerwise_token_count[layer_name]
                # Average norms across all micro-batches
                avg_attn_out_norm = layerwise_attn_out_norm_sum[layer_name] / num_micro_batches
                avg_ffn_out_norm = layerwise_ffn_out_norm_sum[layer_name] / num_micro_batches

                self.layerwise_statis[layer_name]["out_var"].append(avg_out_var)
                self.layerwise_statis[layer_name]["attn_out_var"].append(avg_attn_var)
                self.layerwise_statis[layer_name]["ffn_out_var"].append(avg_ffn_var)
                self.layerwise_statis[layer_name]["attn_out_norm"].append(avg_attn_out_norm)
                self.layerwise_statis[layer_name]["ffn_out_norm"].append(avg_ffn_out_norm)

        # Collect higher order moments and outlier detection
        if (
            self.layerwise_moments_config is not None
            and self.global_step % self.layerwise_moments_config.collect_interval == 0
            and layerwise_stats is not None
        ):
            self._moments_collected_this_step = True
            with torch.no_grad():
                for layer_name, stats in layerwise_stats.items():
                    # Compute higher order moments for decoder block output
                    if self.layerwise_moments_config.collect_full_moments:
                        hidden_state = stats.get("hidden_state")
                        if hidden_state is not None:
                            hidden_moments = compute_higher_order_moments(
                                hidden_state,
                                sample_features=self.layerwise_moments_config.sample_features,
                                sample_ratio=self.layerwise_moments_config.sample_ratio,
                                compute_per_feature=self.layerwise_moments_config.compute_per_feature_moments,
                                compute_channel_moments=self.layerwise_moments_config.compute_channel_moments,
                            )
                            for key, value in hidden_moments.items():
                                self.layerwise_moments[layer_name][f"out_{key}"].append(value)
                            log.debug(f"Step {self.global_step}: Collected {len(hidden_moments)} moments for {layer_name} decoder output")
                        else:
                            log.debug(f"Step {self.global_step}: No decoder output found for {layer_name}")

                        # Compute higher order moments for attention output
                        attn_output = stats.get("attn_output")
                        if attn_output is not None:
                            attn_moments = compute_higher_order_moments(
                                attn_output,
                                sample_features=self.layerwise_moments_config.sample_features,
                                sample_ratio=self.layerwise_moments_config.sample_ratio,
                                compute_per_feature=self.layerwise_moments_config.compute_per_feature_moments,
                                compute_channel_moments=self.layerwise_moments_config.compute_channel_moments,
                            )
                            for key, value in attn_moments.items():
                                self.layerwise_moments[layer_name][f"attn_{key}"].append(value)
                            log.debug(f"Step {self.global_step}: Collected {len(attn_moments)} moments for {layer_name} attention")
                        else:
                            log.debug(f"Step {self.global_step}: No attention output found for {layer_name}")

                        # Compute higher order moments for FFN output
                        ffn_output = stats.get("ffn_output")
                        if ffn_output is not None:
                            ffn_moments = compute_higher_order_moments(
                                ffn_output,
                                sample_features=self.layerwise_moments_config.sample_features,
                                sample_ratio=self.layerwise_moments_config.sample_ratio,
                                compute_per_feature=self.layerwise_moments_config.compute_per_feature_moments,
                                compute_channel_moments=self.layerwise_moments_config.compute_channel_moments,
                            )
                            for key, value in ffn_moments.items():
                                self.layerwise_moments[layer_name][f"ffn_{key}"].append(value)

                    # Detect outliers for decoder block output
                    hidden_state = stats.get("hidden_state")
                    if hidden_state is not None:
                        hidden_outliers = detect_outliers(
                            hidden_state,
                            method=self.layerwise_moments_config.outlier_detection_method,
                            threshold=self.layerwise_moments_config.outlier_threshold,
                            percentile=self.layerwise_moments_config.outlier_percentile,
                            top_k=self.layerwise_moments_config.top_k,
                        )
                        self.layerwise_moments[layer_name]["out_outlier_count"].append(hidden_outliers["outlier_count"])
                        self.layerwise_moments[layer_name]["out_outlier_ratio"].append(hidden_outliers["outlier_ratio"])
                        if "top_magnitudes" in hidden_outliers:
                            self.layerwise_moments[layer_name]["out_top_magnitudes"].append(hidden_outliers["top_magnitudes"])
                        if "top_magnitude_ratio" in hidden_outliers:
                            self.layerwise_moments[layer_name]["out_top_magnitude_ratio"].append(hidden_outliers["top_magnitude_ratio"])

                    # Detect outliers for attention output
                    attn_output = stats.get("attn_output")
                    if attn_output is not None:
                        attn_outliers = detect_outliers(
                            attn_output,
                            method=self.layerwise_moments_config.outlier_detection_method,
                            threshold=self.layerwise_moments_config.outlier_threshold,
                            percentile=self.layerwise_moments_config.outlier_percentile,
                            top_k=self.layerwise_moments_config.top_k,
                        )
                        self.layerwise_moments[layer_name]["attn_outlier_count"].append(attn_outliers["outlier_count"])
                        self.layerwise_moments[layer_name]["attn_outlier_ratio"].append(attn_outliers["outlier_ratio"])
                        if "top_magnitudes" in attn_outliers:
                            self.layerwise_moments[layer_name]["attn_top_magnitudes"].append(attn_outliers["top_magnitudes"])
                        if "top_magnitude_ratio" in attn_outliers:
                            self.layerwise_moments[layer_name]["attn_top_magnitude_ratio"].append(attn_outliers["top_magnitude_ratio"])

                    # Detect outliers for FFN output
                    ffn_output = stats.get("ffn_output")
                    if ffn_output is not None:
                        ffn_outliers = detect_outliers(
                            ffn_output,
                            method=self.layerwise_moments_config.outlier_detection_method,
                            threshold=self.layerwise_moments_config.outlier_threshold,
                            percentile=self.layerwise_moments_config.outlier_percentile,
                            top_k=self.layerwise_moments_config.top_k,
                        )
                        self.layerwise_moments[layer_name]["ffn_outlier_count"].append(ffn_outliers["outlier_count"])
                        self.layerwise_moments[layer_name]["ffn_outlier_ratio"].append(ffn_outliers["outlier_ratio"])
                        if "top_magnitudes" in ffn_outliers:
                            self.layerwise_moments[layer_name]["ffn_top_magnitudes"].append(ffn_outliers["top_magnitudes"])
                        if "top_magnitude_ratio" in ffn_outliers:
                            self.layerwise_moments[layer_name]["ffn_top_magnitude_ratio"].append(ffn_outliers["top_magnitude_ratio"])

        return ce_batch_loss, z_batch_loss

    def train_step(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Reset moments collected flag at the beginning of each step
        if self.layerwise_moments_config is not None:
            self._moments_collected_this_step = False

        # Write data-indices to file.
        if self.indices_file is not None and "index" in batch:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_file.write(f"{self.global_step}\t{indices}\n")

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward-backward pass.
        ce_batch_loss, z_batch_loss = self.train_batch(batch)

        # Collect loss, potentially reducing over all ranks.
        if reduce_global_loss and get_world_size() > 1:
            dist.reduce(ce_batch_loss, 0, op=ReduceOp.AVG)
            if z_batch_loss is not None:
                dist.reduce(z_batch_loss, 0, op=ReduceOp.AVG)

        # Clip gradient norms and collect param/gradient/optim metrics.
        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        should_log_spike_metrics_this_step = self.should_log_spike_metrics_this_step()

        # Always update global spike count if spike detection is enabled
        if self.spike_detection_enabled:
            # Lightweight spike detection to ensure global count is accurate for ALL steps
            self.optim._update_global_spike_count()

        if (
            self.layerwise_statis_collect_interval is not None
            and self.global_step % self.layerwise_statis_collect_interval == 0
        ):
            with torch.no_grad():
                for layer_idx, layer in enumerate(self.model.transformer.blocks):
                    # Compute layer-wise gradient norm using FSDP-compatible approach
                    layer_grad_norm_sq = 0.0
                    layer_param_norm_sq = 0.0
                    param_count = 0

                    for param in layer.parameters():
                        if param.grad is not None:
                            # Only compute for parameters that have gradients on this rank
                            layer_grad_norm_sq += param.grad.norm().item() ** 2

                        # For parameter norms, we need to use summon_full_params
                        layer_param_norm_sq += param.norm().item() ** 2
                        param_count += 1

                    # Aggregate across all ranks using all_reduce
                    if param_count > 0:
                        grad_norm_tensor = torch.tensor(layer_grad_norm_sq, device=self.device)
                        param_norm_tensor = torch.tensor(layer_param_norm_sq, device=self.device)

                        dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(param_norm_tensor, op=dist.ReduceOp.SUM)

                        layer_grad = grad_norm_tensor.sqrt().cpu().item()
                        layer_param = param_norm_tensor.sqrt().cpu().item()
                    else:
                        layer_grad = 0.0
                        layer_param = 0.0

                    # Only log on rank 0 to avoid duplicates
                    if get_local_rank() == 0:
                        self.layerwise_statis[f"layer_{layer_idx}"]["grad_norm"].append(layer_grad)
                        self.layerwise_statis[f"layer_{layer_idx}"]["param_norm"].append(layer_param)

        # Compute ΔW/W metrics (relative update size) for MuP analysis
        # This is computed as ||grad * lr|| / ||W||, which measures the relative update magnitude
        # We compute this before the optimizer step when gradients are still available
        should_compute_delta_w_metrics = (
            should_log_optim_metrics_this_step or self.global_step % self.cfg.console_log_interval == 0
        )

        if should_compute_delta_w_metrics:
            # Get current learning rate from scheduler (before it's updated in param_groups)
            current_lr = self.scheduler.get_lr(
                self.cfg.optimizer.learning_rate,
                self.scheduler_current,
                self.scheduler_max,
            )

            with torch.no_grad():
                # Compute global metrics: ||grad * lr||^2 and ||W||^2
                # Since lr is scalar: ||grad * lr|| = lr * ||grad||
                global_grad_norm_sq = 0.0
                global_w_norm_sq = 0.0

                for param in self.model.parameters():
                    if param.grad is not None:
                        global_grad_norm_sq += param.grad.norm().item() ** 2
                    global_w_norm_sq += param.norm().item() ** 2

                # Aggregate across ranks only for FSDP (DDP already has same values on all ranks)
                is_fsdp = isinstance(self.dist_model, FSDP)
                if is_fsdp and get_world_size() > 1:
                    grad_norm_tensor = torch.tensor(global_grad_norm_sq, device=self.device)
                    w_norm_tensor = torch.tensor(global_w_norm_sq, device=self.device)
                    dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(w_norm_tensor, op=dist.ReduceOp.SUM)
                    global_grad_norm_sq = grad_norm_tensor.item()
                    global_w_norm_sq = w_norm_tensor.item()

                # Compute ||grad * lr|| = lr * ||grad||
                global_grad_norm = math.sqrt(global_grad_norm_sq)
                global_w_norm = math.sqrt(global_w_norm_sq)
                global_delta_w_norm = current_lr * global_grad_norm

                # Compute relative update norm: ||grad * lr|| / ||W||
                if global_w_norm > 1e-12:
                    global_relative_update_norm = global_delta_w_norm / global_w_norm
                else:
                    global_relative_update_norm = 0.0

                metrics["mup/global_relative_update_norm"] = global_relative_update_norm

        optim_metrics = self.optim.clip_grads_and_collect_metrics(
            self.global_step,
            collect_param_metrics=should_log_optim_metrics_this_step,
            # passing this process group here ensures metrics are reduced correctly when we're using
            # HYBRID sharding.
            process_group=self.dist_model.process_group,
            collect_spike_metrics=should_log_spike_metrics_this_step,
            is_fsdp=isinstance(self.dist_model, FSDP),
        )

        # Adjust the learning rate.
        for group in self.optim.param_groups:
            # TODO: if we want to enable different LRs or gradient clipping settings per group
            # we should pass `group["initial_lr"]` or `group["initial_max_grad_norm"]` here instead of
            # the corresponding values from `self.cfg`.
            group["lr"] = self.scheduler.get_lr(
                self.cfg.optimizer.learning_rate,
                self.scheduler_current,
                self.scheduler_max,
            )
            group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                self.cfg.max_grad_norm,
                self.scheduler_current,
                self.scheduler_max,
            )
            group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                self.cfg.max_grad_norm_ratio,
                self.scheduler_current,
                self.scheduler_max,
            )

        # Optimizer step.
        self.optim.step()

        # Collect metrics and check for NaN loss.
        # NOTE: this involves a bunch of host-device syncs so we wait until the last moment to do this.
        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")
        if z_batch_loss is not None and torch.isnan(z_batch_loss):
            raise ValueError("nan loss encountered")
        for key, value in optim_metrics.items():
            if "step" in key:
                continue
            if "spike_detection" in key:
                metrics[f"{key}"] = value.item()
            else:
                metrics[f"optim/{key}"] = value.item()
        self.cur_train_loss = ce_batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        # Safe perplexity calculation to avoid overflow
        if self.cur_train_loss > 700:  # exp(700) is close to the float limit
            metrics["train/Perplexity"] = float("inf")
        else:
            metrics["train/Perplexity"] = math.exp(self.cur_train_loss)
        if z_batch_loss is not None:
            metrics["train/ZLoss"] = z_batch_loss.item()

        for metric_name, (
            metric_val,
            reduction,
        ) in self.model.compute_auxiliary_metrics(reset=True).items():
            # Skip None values
            if metric_val is None:
                log.warning(f"Skipping metric '{metric_name}' because value is None")
                continue

            if not isinstance(metric_val, torch.Tensor):
                metric_val = torch.tensor(metric_val)
            else:
                metric_val = get_local_tensor(metric_val.detach()).float()

            if get_world_size() > 1:
                if reduction == ReduceType.mean:
                    metric_val = dist.reduce(metric_val, 0, op=ReduceOp.AVG)
                elif reduction == ReduceType.sum:
                    metric_val = dist.reduce(metric_val, 0, op=ReduceOp.SUM)
                elif reduction == ReduceType.max:
                    metric_val = dist.reduce(metric_val, 0, op=ReduceOp.MAX)
                else:
                    raise NotImplementedError(reduction)
            if metric_val is None:
                log.warning(f"Skipping metric '{metric_name}' because value is None")
                continue
            metric_item = metric_val.item()
            if math.isnan(metric_item) or math.isinf(metric_item):
                log.warning(f"Metric '{metric_name}' has invalid value: {metric_item}, skipping")
                continue

            metrics[f"train/{metric_name}"] = metric_item

        lr = self.scheduler.get_lr(
            self.cfg.optimizer.learning_rate,
            self.scheduler_current,
            self.scheduler_max,
        )
        metrics["train/LR"] = lr

        # Maybe collect post-step optimizer-specific metrics.
        if should_log_optim_metrics_this_step or should_log_spike_metrics_this_step:
            optim_metrics = self.optim.get_post_step_metrics(
                self.dist_model, process_group=self.dist_model.process_group
            )

            for key, value in optim_metrics.items():
                if "step" in key:
                    continue

                if "spike_detection" in key and should_log_spike_metrics_this_step:
                    metrics[f"{key}"] = value.item()

                if should_log_optim_metrics_this_step:
                    metrics[f"optim/{key}"] = value.item()

        return metrics

    def eval_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            ce_loss, _, logits, _ = self.model_forward(batch, loss_reduction="none")
        return ce_loss.mean(dim=-1), logits

    def eval_step(self, batch: Dict[str, Any], evaluator: Evaluator) -> None:
        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            ce_loss, logits = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(
            batch, ce_loss, logits
        )  # batch includes all keys that the downstream evaluation needs

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        microbatch_size = self.cfg.device_train_microbatch_size
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def system_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self.global_step < 3 or self.global_step % 10 == 0:
            peak_gpu_mb = peak_gpu_memory()
            if peak_gpu_mb is not None:
                metrics["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        def format_float(value: float) -> str:
            if value < 0.0001:
                return str(value)  # scientific notation
            elif value > 1000:
                return f"{int(value):,d}"
            elif value > 100:
                return f"{value:.1f}"
            elif value > 10:
                return f"{value:.2f}"
            elif value > 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    if name == "optim/total_grad_norm"
                    or not name.startswith("optim/")  # there's too many optimizer metrics
                ]
            )
        )

    def save_layerwise_stats_to_file(self, metrics: Dict[str, float]) -> None:
        """
        Save layer-wise statistics to a JSONL file for offline analysis and plotting.

        Each line in the JSONL file contains a JSON object with statistics for one step.
        This format is efficient for streaming writes and easy to parse.

        Format:
        {"step": <global_step>, "timestamp": <iso_timestamp>, "metrics": {"layer_0": {...}, "layer_1": {...}, ...}}
        """
        if self.layerwise_stats_save_path is None or get_global_rank() != 0:
            return

        # Helper function to safely get the last value from a list
        def safe_last(lst: List[float], default: float = 0.0) -> float:
            return lst[-1] if lst else default

        # Prepare structured data for easy parsing
        layerwise_data = {"step": self.global_step, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": {}}

        for layer_name, layer_metrics in self.layerwise_statis.items():
            layerwise_data["metrics"][layer_name] = {
                "grad_norm": safe_last(layer_metrics["grad_norm"]),
                "param_norm": safe_last(layer_metrics["param_norm"]),
                "out_var": safe_last(layer_metrics["out_var"]),
                "attn_var": safe_last(layer_metrics["attn_out_var"]),
                "ffn_var": safe_last(layer_metrics["ffn_out_var"]),
                "attn_out_norm": safe_last(layer_metrics["attn_out_norm"]),
                "ffn_out_norm": safe_last(layer_metrics["ffn_out_norm"]),
            }

        # Append to JSONL file (one JSON object per line)
        with open(self.layerwise_stats_save_path, "a") as f:
            f.write(json.dumps(layerwise_data) + "\n")

        log.debug(f"Layer-wise statistics appended to {self.layerwise_stats_save_path}")

    def save_layerwise_moments(self) -> None:
        """
        Save layer-wise higher order moments and outlier detection results to a JSONL file.
        Each line in the JSONL file contains a JSON object with moments for one step.
        This format is efficient for streaming writes and easy to parse.

        Format:
        {"step": <global_step>, "timestamp": <iso_timestamp>, "metrics": {"layer_0": {...}, "layer_1": {...}, ...}}
        """
        if self.layerwise_moments_save_path is None or get_global_rank() != 0:
            return

        # Only save if moments were actually collected in this step
        if not self._moments_collected_this_step:
            return  # Skip saving if no moments were collected this step

        # Reset the flag after checking
        self._moments_collected_this_step = False

        # Helper function to safely get the last value from a list
        def safe_last(lst: List[float], default: float = 0.0) -> float:
            return lst[-1] if lst else default

        # Prepare structured data for easy parsing
        layerwise_data = {"step": self.global_step, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": {}}

        for layer_name, layer_metrics in self.layerwise_moments.items():
            layer_data = {}

            # Add skewness and kurtosis for decoder block output
            if "out_skewness" in layer_metrics:
                layer_data["out_skewness"] = safe_last(layer_metrics["out_skewness"])
            if "out_kurtosis" in layer_metrics:
                layer_data["out_kurtosis"] = safe_last(layer_metrics["out_kurtosis"])
            if "out_excess_kurtosis" in layer_metrics:
                layer_data["out_excess_kurtosis"] = safe_last(layer_metrics["out_excess_kurtosis"])

            # Add per-feature moments for decoder block output
            if "out_skewness_per_feature" in layer_metrics:
                layer_data["out_skewness_per_feature"] = layer_metrics["out_skewness_per_feature"][-1].tolist()
            if "out_kurtosis_per_feature" in layer_metrics:
                layer_data["out_kurtosis_per_feature"] = layer_metrics["out_kurtosis_per_feature"][-1].tolist()

            # Add robust statistics for decoder block output
            if "out_cv" in layer_metrics:
                layer_data["out_cv"] = safe_last(layer_metrics["out_cv"])
            if "out_mad" in layer_metrics:
                layer_data["out_mad"] = safe_last(layer_metrics["out_mad"])
            if "out_iqr" in layer_metrics:
                layer_data["out_iqr"] = safe_last(layer_metrics["out_iqr"])

            # Add skewness and kurtosis for attention output
            if "attn_skewness" in layer_metrics:
                layer_data["attn_skewness"] = safe_last(layer_metrics["attn_skewness"])
            if "attn_kurtosis" in layer_metrics:
                layer_data["attn_kurtosis"] = safe_last(layer_metrics["attn_kurtosis"])
            if "attn_excess_kurtosis" in layer_metrics:
                layer_data["attn_excess_kurtosis"] = safe_last(layer_metrics["attn_excess_kurtosis"])

            # Add per-feature moments for attention output
            if "attn_skewness_per_feature" in layer_metrics:
                layer_data["attn_skewness_per_feature"] = layer_metrics["attn_skewness_per_feature"][-1].tolist()
            if "attn_kurtosis_per_feature" in layer_metrics:
                layer_data["attn_kurtosis_per_feature"] = layer_metrics["attn_kurtosis_per_feature"][-1].tolist()

            # Add skewness and kurtosis for FFN output
            if "ffn_skewness" in layer_metrics:
                layer_data["ffn_skewness"] = safe_last(layer_metrics["ffn_skewness"])
            if "ffn_kurtosis" in layer_metrics:
                layer_data["ffn_kurtosis"] = safe_last(layer_metrics["ffn_kurtosis"])
            if "ffn_excess_kurtosis" in layer_metrics:
                layer_data["ffn_excess_kurtosis"] = safe_last(layer_metrics["ffn_excess_kurtosis"])

            # Add per-feature moments for FFN output
            if "ffn_skewness_per_feature" in layer_metrics:
                layer_data["ffn_skewness_per_feature"] = layer_metrics["ffn_skewness_per_feature"][-1].tolist()
            if "ffn_kurtosis_per_feature" in layer_metrics:
                layer_data["ffn_kurtosis_per_feature"] = layer_metrics["ffn_kurtosis_per_feature"][-1].tolist()

            # Add robust statistics for attention output
            if "attn_cv" in layer_metrics:
                layer_data["attn_cv"] = safe_last(layer_metrics["attn_cv"])
            if "attn_mad" in layer_metrics:
                layer_data["attn_mad"] = safe_last(layer_metrics["attn_mad"])
            if "attn_iqr" in layer_metrics:
                layer_data["attn_iqr"] = safe_last(layer_metrics["attn_iqr"])

            # Add robust statistics for FFN output
            if "ffn_cv" in layer_metrics:
                layer_data["ffn_cv"] = safe_last(layer_metrics["ffn_cv"])
            if "ffn_mad" in layer_metrics:
                layer_data["ffn_mad"] = safe_last(layer_metrics["ffn_mad"])
            if "ffn_iqr" in layer_metrics:
                layer_data["ffn_iqr"] = safe_last(layer_metrics["ffn_iqr"])

            # Add channel-wise moments for decoder block output
            if "out_channel_avg_skewness" in layer_metrics:
                layer_data["out_channel_avg_skewness"] = safe_last(layer_metrics["out_channel_avg_skewness"])
            if "out_channel_avg_kurtosis" in layer_metrics:
                layer_data["out_channel_avg_kurtosis"] = safe_last(layer_metrics["out_channel_avg_kurtosis"])
            if "out_channel_avg_excess_kurtosis" in layer_metrics:
                layer_data["out_channel_avg_excess_kurtosis"] = safe_last(layer_metrics["out_channel_avg_excess_kurtosis"])
            if "out_channel_std_skewness" in layer_metrics:
                layer_data["out_channel_std_skewness"] = safe_last(layer_metrics["out_channel_std_skewness"])
            if "out_channel_std_kurtosis" in layer_metrics:
                layer_data["out_channel_std_kurtosis"] = safe_last(layer_metrics["out_channel_std_kurtosis"])

            # Add channel-wise moments for attention output
            if "attn_channel_avg_skewness" in layer_metrics:
                layer_data["attn_channel_avg_skewness"] = safe_last(layer_metrics["attn_channel_avg_skewness"])
            if "attn_channel_avg_kurtosis" in layer_metrics:
                layer_data["attn_channel_avg_kurtosis"] = safe_last(layer_metrics["attn_channel_avg_kurtosis"])
            if "attn_channel_avg_excess_kurtosis" in layer_metrics:
                layer_data["attn_channel_avg_excess_kurtosis"] = safe_last(layer_metrics["attn_channel_avg_excess_kurtosis"])
            if "attn_channel_std_skewness" in layer_metrics:
                layer_data["attn_channel_std_skewness"] = safe_last(layer_metrics["attn_channel_std_skewness"])
            if "attn_channel_std_kurtosis" in layer_metrics:
                layer_data["attn_channel_std_kurtosis"] = safe_last(layer_metrics["attn_channel_std_kurtosis"])

            # Add channel-wise moments for FFN output
            if "ffn_channel_avg_skewness" in layer_metrics:
                layer_data["ffn_channel_avg_skewness"] = safe_last(layer_metrics["ffn_channel_avg_skewness"])
            if "ffn_channel_avg_kurtosis" in layer_metrics:
                layer_data["ffn_channel_avg_kurtosis"] = safe_last(layer_metrics["ffn_channel_avg_kurtosis"])
            if "ffn_channel_avg_excess_kurtosis" in layer_metrics:
                layer_data["ffn_channel_avg_excess_kurtosis"] = safe_last(layer_metrics["ffn_channel_avg_excess_kurtosis"])
            if "ffn_channel_std_skewness" in layer_metrics:
                layer_data["ffn_channel_std_skewness"] = safe_last(layer_metrics["ffn_channel_std_skewness"])
            if "ffn_channel_std_kurtosis" in layer_metrics:
                layer_data["ffn_channel_std_kurtosis"] = safe_last(layer_metrics["ffn_channel_std_kurtosis"])

            # Add outlier detection results for decoder block output
            if "out_outlier_count" in layer_metrics:
                layer_data["out_outlier_count"] = safe_last(layer_metrics["out_outlier_count"])
            if "out_outlier_ratio" in layer_metrics:
                layer_data["out_outlier_ratio"] = safe_last(layer_metrics["out_outlier_ratio"])
            if "out_top_magnitude_ratio" in layer_metrics:
                layer_data["out_top_magnitude_ratio"] = safe_last(layer_metrics["out_top_magnitude_ratio"])

            # Add outlier detection results for attention output
            if "attn_outlier_count" in layer_metrics:
                layer_data["attn_outlier_count"] = safe_last(layer_metrics["attn_outlier_count"])
            if "attn_outlier_ratio" in layer_metrics:
                layer_data["attn_outlier_ratio"] = safe_last(layer_metrics["attn_outlier_ratio"])
            if "attn_top_magnitude_ratio" in layer_metrics:
                layer_data["attn_top_magnitude_ratio"] = safe_last(layer_metrics["attn_top_magnitude_ratio"])

            # Add outlier detection results for FFN output
            if "ffn_outlier_count" in layer_metrics:
                layer_data["ffn_outlier_count"] = safe_last(layer_metrics["ffn_outlier_count"])
            if "ffn_outlier_ratio" in layer_metrics:
                layer_data["ffn_outlier_ratio"] = safe_last(layer_metrics["ffn_outlier_ratio"])
            if "ffn_top_magnitude_ratio" in layer_metrics:
                layer_data["ffn_top_magnitude_ratio"] = safe_last(layer_metrics["ffn_top_magnitude_ratio"])

            layerwise_data["metrics"][layer_name] = layer_data

        # Append to JSONL file (one JSON object per line)
        with open(self.layerwise_moments_save_path, "a") as f:
            f.write(json.dumps(layerwise_data) + "\n")

        log.debug(f"Layer-wise moments appended to {self.layerwise_moments_save_path}")

    def should_log_optim_metrics_this_step(self) -> bool:
        if self.cfg.wandb is None and self.cfg.swanlab is None:
            # We only log optimizer-specific metrics to W&B, since there are usually too many metrics
            # to log to the console.
            return False
        optim_log_interval = self.cfg.optimizer.metrics_log_interval
        if optim_log_interval is None:
            optim_log_interval = (
                self.cfg.wandb.log_interval if self.cfg.wandb is not None else self.cfg.swanlab.log_interval
            )
        else:
            optim_log_interval = max(
                optim_log_interval,
                (self.cfg.wandb.log_interval if self.cfg.wandb is not None else self.cfg.swanlab.log_interval),
            )
        return self.global_step % optim_log_interval == 0

    def should_log_spike_metrics_this_step(self) -> bool:
        if not self.spike_detection_enabled:
            return False
        if self.cfg.wandb is None and self.cfg.swanlab is None:
            # We only log spike metrics to W&B, since there are usually too many metrics
            # to log to the console.
            return False
        optim_log_interval = self.cfg.optimizer.metrics_log_interval
        if optim_log_interval is None:
            optim_log_interval = (
                self.cfg.wandb.log_interval if self.cfg.wandb is not None else self.cfg.swanlab.log_interval
            )
        else:
            optim_log_interval = max(
                optim_log_interval,
                (self.cfg.wandb.log_interval if self.cfg.wandb is not None else self.cfg.swanlab.log_interval),
            )
        return self.global_step % optim_log_interval == 0

    def should_log_this_step(self) -> bool:
        if self.global_step % self.cfg.console_log_interval == 0:
            return True
        elif self.cfg.wandb is not None and self.global_step % self.cfg.wandb.log_interval == 0:
            return True
        elif self.cfg.swanlab is not None and self.global_step % self.cfg.swanlab.log_interval == 0:
            return True
        else:
            return False

    def should_generate_spike_plot_this_step(self) -> bool:
        if not self.spike_detection_enabled:
            return False

        # Determine the interval for spike plots
        plot_interval = self.cfg.optimizer.spike_plot_interval
        if plot_interval is None:
            plot_interval = self.cfg.save_interval

        if plot_interval is None:
            return False

        return self.global_step % plot_interval == 0

    def generate_and_log_spike_plot(self) -> None:
        """Generate spike plot and log it to wandb if available."""
        if not self.spike_detection_enabled:
            return

        # Only generate plot on rank 0 to avoid multiple plots
        if get_global_rank() != 0:
            return

        try:
            # Create plots directory
            plots_dir = Path(self.cfg.save_folder) / "spike_plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Generate plot
            plot_path = plots_dir / f"spike_plot_step_{self.global_step}.png"
            saved_plot_path = self.optim.generate_spike_plot(str(plot_path), self.global_step)

            if saved_plot_path is not None:
                log.info(f"Spike plot saved to {saved_plot_path}")

                # Log to wandb if available
                if self.cfg.wandb is not None:
                    wandb.log(
                        {
                            "spike_detection/layer_spikes_plot": wandb.Image(saved_plot_path),
                        },
                        step=self.global_step,
                    )

                if self.cfg.swanlab is not None:
                    swanlab.log(
                        {
                            "spike_detection/layer_spikes_plot": swanlab.Image(saved_plot_path),
                        },
                        step=self.global_step,
                    )

                # Upload to remote if configured
                if self.cfg.remote_save_folder is not None:
                    remote_path = (
                        f"{self.cfg.remote_save_folder.rstrip('/')}/spike_plots/spike_plot_step_{self.global_step}.png"
                    )
                    upload(saved_plot_path, remote_path)

                self.last_spike_plot_step = self.global_step

        except Exception as e:
            log.warning(f"Failed to generate or log spike plot: {e}")

    def log_spike_metrics_to_console(self, spike_metrics: Dict[str, float]) -> None:
        """
        Log spike detection metrics to console.

        Metrics include:
        - current_step_spikes: Number of parameters with spikes in current step
        - spike_ratio: Ratio of spike parameters to total parameters in current step
        - global_spike_count: Number of training steps that had spike parameters
        """
        if not spike_metrics:
            return

        # Extract key spike metrics for console logging
        console_metrics = {}
        for key, value in spike_metrics.items():
            if key.startswith("spike_detection/"):
                short_key = key.replace("spike_detection/", "")
                console_metrics[short_key] = value

        if console_metrics:
            log.info(
                f"Spike Detection Metrics:\n"
                + "\n".join([f"    {name}={value:.4f}" for name, value in console_metrics.items()])
            )

    def eval(self) -> Dict[str, Any]:
        # Zero gradients and set model to 'eval' mode.
        self.optim.zero_grad(set_to_none=True)
        self.dist_model.eval()

        eval_metrics = {}
        for evaluator in self.evaluators:
            log.info(f"Running evaluation for '{evaluator.label}'...")

            # Reset metrics.
            evaluator.reset_metrics()

            # Initialize data loader iterator.
            eval_batches = iter(evaluator.eval_loader)

            # Adjust how many batches to evaluate on.
            num_eval_batches = (
                evaluator.subset_num_batches
                if evaluator.subset_num_batches is not None
                else self.cfg.eval_subset_num_batches
            )
            if num_eval_batches is not None and num_eval_batches > 0:
                num_eval_batches = min(num_eval_batches, len(evaluator.eval_loader))
            else:
                num_eval_batches = len(evaluator.eval_loader)

            # To prevent hanging with DDP, we synchronize the number of batches to evaluate on.
            # Use minimum across all ranks to ensure no rank runs out of data
            if get_world_size() > 1:
                num_eval_batches_tensor = torch.tensor(num_eval_batches, device=self.device)
                dist.all_reduce(num_eval_batches_tensor, op=dist.ReduceOp.MIN)
                num_eval_batches = int(num_eval_batches_tensor.item())

            if num_eval_batches > 0:
                eval_batches = islice(eval_batches, num_eval_batches)

            # Run model over batches.
            for eval_step, eval_batch in enumerate(eval_batches):
                self.eval_step(eval_batch, evaluator)

                # Log to console.
                if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.cfg.console_log_interval == 0:
                    log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")

            # Get final metrics.
            metrics = evaluator.compute_metrics()
            eval_metrics.update(metrics)

            # Only log metrics on rank 0 to avoid duplicate console output
            if get_global_rank() == 0:
                self.log_metrics_to_console(f"{evaluator.label}", metrics)

            del eval_batches

        # Eval compiles a bunch more versions, and the result is terrible. This way we get back to zero.
        if self.cfg.compile is not None:
            torch.compiler.reset()

        return eval_metrics

    def check_if_cancelled(self) -> Tuple[bool, int]:
        should_cancel = False
        cancel_reason: Optional[str] = None
        extra_steps = 0
        if get_global_rank() == 0:
            if self.cfg.time_limit is not None and time.time() - self._start_time >= self.cfg.time_limit:
                # First check if we've reached the training time limit.
                should_cancel = True
                cancel_reason = "time limit reached"
                extra_steps = self.cfg.extra_steps_after_cancel
            elif (
                self.cfg.early_stopping_factor is not None
                and self.global_step > self.cfg.scheduler.t_warmup
                and self.cur_train_loss > self.cfg.early_stopping_factor * self.min_train_loss
            ):
                # Next check if early stopping loss criteria is met.
                should_cancel = True
                cancel_reason = "early stopping from loss increase"
            elif wandb.run is not None and (api_key := os.environ.get("WANDB_API_KEY")) is not None:
                # Finally, check if someone canceled the run from W&B by adding the 'cancel' / 'canceled' tag..
                # We won't see it in the run object. So we have to use the import/export API to check.
                from requests.exceptions import RequestException
                from wandb.errors import CommError

                try:
                    api = wandb.Api(api_key=api_key)
                    run = api.run(wandb.run.path)
                    for tag in run.tags or []:
                        if tag.lower() in {
                            "cancel",
                            "canceled",
                            "cancelled",
                        }:
                            should_cancel = True
                            cancel_reason = "Weights & Biases tag"
                            extra_steps = self.cfg.extra_steps_after_cancel
                            break
                except (RequestException, CommError):
                    log.info("Failed to check if W&B run is cancelled, continuing run.")

        run_canceled = synchronize_flag(should_cancel, self.device)
        if run_canceled:
            extra_steps = synchronize_value(extra_steps, self.device)
            if cancel_reason is None:
                if extra_steps > 0:
                    log.warning(f"Run canceled, stopping in {extra_steps} more steps...")
                else:
                    log.warning("Run canceled")
            else:
                if extra_steps > 0:
                    log.warning(f"Run canceled due to {cancel_reason}, stopping in {extra_steps} more steps...")
                else:
                    log.warning(f"Run canceled due to {cancel_reason}")

        return run_canceled, extra_steps

    def fit(self):
        if self.cfg.stop_after is not None:
            if self.cfg.stop_at is None:
                self.cfg.stop_at = self.global_step + self.cfg.stop_after
            else:
                self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after)
        if self.cfg.stop_at is None:
            self.cfg.stop_at = self.max_steps + 10

        self._start_time = time.time()
        self._gc_init_state = gc.isenabled()  # cache if garbage collection is enabled, reset on close.

        # Disable automatic garbage collection, FSDP doesn't work well with it.
        if self.cfg.gen1_gc_interval is not None:
            gc.disable()

        if self.cfg.load_path is not None and self.global_step > 0 and self.cfg.eval_on_load:
            eval_metrics = self.eval()
            if wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)

            if self.cfg.swanlab is not None:
                swanlab.log(eval_metrics, step=self.global_step)

        # Set model to 'train' mode.
        self.dist_model.train()

        # Initialize monitors.
        assert self.cfg.device_train_batch_size is not None
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if wandb.run is not None:
                wandb.log(sys_metrics, step=0)

            if self.cfg.swanlab is not None:
                swanlab.log(sys_metrics, step=0)

        # Python Profiler stuff
        if self.cfg.python_profiling:
            python_profiler = cProfile.Profile()
        else:
            python_profiler = None

        # PyTorch Profiler stuff
        if self.cfg.torch_profiling and get_global_rank() == 0:
            from torch.profiler import schedule

            profiling_schedule = schedule(wait=1, warmup=5, active=3, repeat=1)

            def on_trace_ready(p):
                profiler_output_dir = Path(self.cfg.save_folder) / "profiler"
                profiler_output_dir.mkdir(exist_ok=True)

                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
                log.info(f"Profile by total GPU time at step {p.step_num}:\n{output}")
                output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
                log.info(f"Profile by total CPU time at step {p.step_num}:\n{output}")

                p.export_chrome_trace(str(trace_path := (profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz")))
                if self.cfg.remote_save_folder is not None:
                    upload_folder = f"{self.cfg.remote_save_folder.rstrip('/')}/profiler"
                    log.info(f"Tracing complete, uploading results to '{upload_folder}'...")
                    upload(trace_path, f"{upload_folder}/{trace_path.name}")

            from torch.profiler import ProfilerActivity

            torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=on_trace_ready,
            )
            del profiling_schedule
        else:
            import contextlib

            torch_profiler = contextlib.nullcontext()

        # Train.
        first_batch: bool = True
        cancel_initiated: bool = False
        stop_at: int = self.cfg.stop_at
        save_checkpoints: bool = True

        with torch_profiler as p:
            for epoch in range(self.epoch or 0, self.max_epochs):
                for batch in self.train_loader:
                    # Bookkeeping.
                    # NOTE: To track the global batch size / number of tokens per batch we make the assumption that all
                    # batches see the same number of tokens, which should be the case for language model pre-training
                    # (at least when drop_last=True).
                    # Alternatively we'd have to use a distributed all reduce over seq_len here, but I don't want that
                    # overhead. So for now I'm putting these assertions here so if the assumption is violated it will
                    # fail loudly.
                    batch_size, seq_len = batch["input_ids"].shape
                    assert seq_len == self.cfg.model.max_sequence_length
                    assert batch_size == self.cfg.device_train_batch_size
                    global_batch_size = batch_size * get_world_size()  # assumes batch size equal across ranks
                    self.global_step += 1
                    self.global_train_examples_seen_this_epoch += global_batch_size
                    self.global_train_tokens_seen += global_batch_size * seq_len
                    speed_monitor.batch_start(
                        global_total_tokens=self.global_train_tokens_seen,
                        device_batch_num_tokens=batch_size * seq_len,  # num tokens in batch for this device
                        # We start monitoring speed after the first batch since the first
                        # batch might be an outlier due to compiling and other initialization overhead.
                        num_fwd_flops=self.model.num_fwd_flops,  # this is per token
                        num_bck_flops=self.model.num_bck_flops,  # this is per token
                        record=not first_batch,
                    )

                    should_log_this_step = self.should_log_this_step()

                    # Run train step on batch.
                    metrics = self.train_step(batch, reduce_global_loss=should_log_this_step)

                    # Maybe collect other metrics.
                    if should_log_this_step:
                        # Speed metrics.
                        metrics.update(speed_monitor.check())
                        # System metrics.
                        metrics.update(self.system_metrics())
                        # Learning rate metrics.
                        metrics.update(lr_monitor.check())

                    # Log metrics to console.
                    if self.global_step % self.cfg.console_log_interval == 0:
                        if get_global_rank() == 0:
                            self.log_metrics_to_console(
                                f"[step={self.global_step}/{self.max_steps},epoch={epoch}]",
                                metrics,
                            )
                            # Log spike metrics to console if available
                            if self.spike_detection_enabled and self.should_log_spike_metrics_this_step:
                                self.log_spike_metrics_to_console(metrics)
                        else:
                            log.info(f"[step={self.global_step}/{self.max_steps},epoch={epoch}]")

                    # Log metrics to W&B.
                    if (
                        wandb.run is not None
                        and self.cfg.wandb is not None
                        and self.global_step % self.cfg.wandb.log_interval == 0
                    ):
                        wandb.log(metrics, step=self.global_step)

                        if (
                            self.layerwise_statis_collect_interval is not None
                            and self.global_step % self.layerwise_statis_collect_interval == 0
                        ):
                            layer_wise_metrics = {}
                            for (
                                layer_name,
                                layer_metrics,
                            ) in self.layerwise_statis.items():
                                layer_wise_metrics[f"layer_grad_norm/{layer_name}"] = (
                                    layer_metrics["grad_norm"][-1] if layer_metrics["grad_norm"] else 0.0
                                )
                                layer_wise_metrics[f"layer_param_norm/{layer_name}"] = (
                                    layer_metrics["param_norm"][-1] if layer_metrics["param_norm"] else 0.0
                                )
                                layer_wise_metrics[f"layer_var/{layer_name}"] = (
                                    layer_metrics["out_var"][-1] if layer_metrics["out_var"] else 0.0
                                )
                                layer_wise_metrics[f"layer_attn_var/{layer_name}"] = (
                                    layer_metrics["attn_out_var"][-1] if layer_metrics["attn_out_var"] else 0.0
                                )
                                layer_wise_metrics[f"layer_ffn_var/{layer_name}"] = (
                                    layer_metrics["ffn_out_var"][-1] if layer_metrics["ffn_out_var"] else 0.0
                                )
                                layer_wise_metrics[f"layer_attn_out_norm/{layer_name}"] = (
                                    layer_metrics["attn_out_norm"][-1] if layer_metrics["attn_out_norm"] else 0.0
                                )
                                layer_wise_metrics[f"layer_ffn_out_norm/{layer_name}"] = (
                                    layer_metrics["ffn_out_norm"][-1] if layer_metrics["ffn_out_norm"] else 0.0
                                )
                            wandb.log(layer_wise_metrics, step=self.global_step)
                            # Save layer-wise statistics to local file
                            self.save_layerwise_stats_to_file(layer_wise_metrics)
                            # Save layer-wise moments to local file
                            self.save_layerwise_moments()

                    if self.cfg.swanlab is not None and self.global_step % self.cfg.swanlab.log_interval == 0:
                        layer_wise_metrics = {}
                        for (
                            layer_name,
                            layer_metrics,
                        ) in self.layerwise_statis.items():
                            layer_wise_metrics[f"layer_grad_norm/{layer_name}"] = (
                                layer_metrics["grad_norm"][-1] if layer_metrics["grad_norm"] else 0.0
                            )
                            layer_wise_metrics[f"layer_param_norm/{layer_name}"] = (
                                layer_metrics["param_norm"][-1] if layer_metrics["param_norm"] else 0.0
                            )
                            layer_wise_metrics[f"layer_var/{layer_name}"] = (
                                layer_metrics["out_var"][-1] if layer_metrics["out_var"] else 0.0
                            )
                            layer_wise_metrics[f"layer_attn_var/{layer_name}"] = (
                                layer_metrics["attn_out_var"][-1] if layer_metrics["attn_out_var"] else 0.0
                            )
                            layer_wise_metrics[f"layer_ffn_var/{layer_name}"] = (
                                layer_metrics["ffn_out_var"][-1] if layer_metrics["ffn_out_var"] else 0.0
                            )
                            layer_wise_metrics[f"layer_attn_out_norm/{layer_name}"] = (
                                layer_metrics["attn_out_norm"][-1] if layer_metrics["attn_out_norm"] else 0.0
                            )
                            layer_wise_metrics[f"layer_ffn_out_norm/{layer_name}"] = (
                                layer_metrics["ffn_out_norm"][-1] if layer_metrics["ffn_out_norm"] else 0.0
                            )
                        swanlab.log(layer_wise_metrics, step=self.global_step)
                        # Save layer-wise statistics to local file
                        self.save_layerwise_stats_to_file(layer_wise_metrics)
                        # Save layer-wise moments to local file
                        self.save_layerwise_moments()

                    # Save layer-wise statistics to local file even if wandb/swanlab are not activated
                    if (
                        self.layerwise_statis_collect_interval is not None
                        and self.global_step % self.layerwise_statis_collect_interval == 0
                    ):
                        layer_wise_metrics = {}
                        for layer_name, layer_metrics in self.layerwise_statis.items():
                            layer_wise_metrics[f"layer_grad_norm/{layer_name}"] = (
                                layer_metrics["grad_norm"][-1] if layer_metrics["grad_norm"] else 0.0
                            )
                            layer_wise_metrics[f"layer_param_norm/{layer_name}"] = (
                                layer_metrics["param_norm"][-1] if layer_metrics["param_norm"] else 0.0
                            )
                            layer_wise_metrics[f"layer_var/{layer_name}"] = (
                                layer_metrics["out_var"][-1] if layer_metrics["out_var"] else 0.0
                            )
                            layer_wise_metrics[f"layer_attn_var/{layer_name}"] = (
                                layer_metrics["attn_out_var"][-1] if layer_metrics["attn_out_var"] else 0.0
                            )
                            layer_wise_metrics[f"layer_ffn_var/{layer_name}"] = (
                                layer_metrics["ffn_out_var"][-1] if layer_metrics["ffn_out_var"] else 0.0
                            )
                            layer_wise_metrics[f"layer_attn_out_norm/{layer_name}"] = (
                                layer_metrics["attn_out_norm"][-1] if layer_metrics["attn_out_norm"] else 0.0
                            )
                            layer_wise_metrics[f"layer_ffn_out_norm/{layer_name}"] = (
                                layer_metrics["ffn_out_norm"][-1] if layer_metrics["ffn_out_norm"] else 0.0
                            )
                        self.save_layerwise_stats_to_file(layer_wise_metrics)
                        self.save_layerwise_moments()

                    # Generate and log spike plot if needed
                    if self.should_generate_spike_plot_this_step():
                        self.generate_and_log_spike_plot()

                    # Maybe save sharded checkpoint.
                    if self.cfg.distributed_strategy == DistributedStrategy.fsdp:
                        if save_checkpoints and (
                            cancel_initiated
                            or (
                                self.cfg.save_interval is not None
                                and self.global_step % self.cfg.save_interval == 0
                                and self.cfg.save_num_checkpoints_to_keep != 0
                            )
                        ):
                            log.info("Saving checkpoint...")
                            checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                            log.info(f"Checkpoint saved to {checkpoint_path}")

                            # Remove any ephemeral checkpoints.
                            while self.ephemeral_checkpoints:
                                self.remove_ephemeral_checkpoint()

                            # Reset speed monitor so that we don't count the time taken to save checkpoints.
                            speed_monitor.reset()

                            # If the run was just canceled this will be the final checkpoint.
                            if cancel_initiated:
                                save_checkpoints = False
                        elif (
                            self.cfg.save_interval_ephemeral is not None
                            and self.global_step % self.cfg.save_interval_ephemeral == 0
                        ):
                            log.info("Saving ephemeral checkpoint...")
                            checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded_ephemeral)
                            log.info(f"Checkpoint saved to {checkpoint_path}")

                            # Reset speed monitor so that we don't count the time taken to save checkpoints.
                            speed_monitor.reset()

                    # Maybe save unsharded checkpoint.
                    # This code snippet should always execute when running DDP.
                    if (
                        save_checkpoints
                        and self.cfg.save_interval_unsharded is not None
                        and self.global_step % self.cfg.save_interval_unsharded == 0
                        and self.cfg.save_num_unsharded_checkpoints_to_keep != 0
                    ):
                        log.info("Saving unsharded checkpoint...")
                        checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                        log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                    # Maybe run evaluations.
                    if (
                        not cancel_initiated
                        and self.cfg.eval_interval is not None
                        and (self.global_step % self.cfg.eval_interval == 0 or self.global_step >= stop_at)
                    ):
                        eval_metrics = self.eval()

                        # Log metrics to W&B.
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        if self.cfg.swanlab is not None:
                            swanlab.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.dist_model.train()

                    # End of batch.
                    first_batch = False
                    if p is not None:
                        p.step()

                    if self.global_step >= stop_at:
                        break

                    # Run generation 1 garbage collection.
                    if self.cfg.gen1_gc_interval is not None and self.global_step % self.cfg.gen1_gc_interval == 0:
                        gc.collect(1)

                    # Python Profiler stuff
                    # We do this now, at the bottom of this loop, so we capture the work of getting the next batch.
                    if python_profiler is not None:
                        if self.global_step == 5:
                            python_profiler.enable()
                        elif self.global_step == 8:
                            python_profiler.disable()
                            python_profiler.print_stats(sort=SortKey.CUMULATIVE)
                            python_profiler = None
                else:
                    log.info("Training epoch complete")
                    self.epoch = epoch + 1
                    self.global_train_examples_seen_this_epoch = 0
                    self.dataset.start_index = 0
                    if self.epoch < self.max_epochs:
                        log.info(f"Reshuffling data loader for epoch {self.epoch}...")
                        self.dataset.reshuffle(self.epoch)
                    continue

                break

        # Save final checkpoint.
        if save_checkpoints:
            if (
                self.cfg.save_interval_unsharded is not None
                and self.last_unsharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final unsharded model checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")
            elif (
                self.cfg.save_num_checkpoints_to_keep != 0
                and self.last_sharded_checkpoint_step != self.global_step
                and self.cfg.distributed_strategy == DistributedStrategy.fsdp
            ):
                log.info("Saving final checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                log.info(f"Checkpoint saved to {checkpoint_path}")

    def close(self, exit_code: int = 0) -> None:
        gc_cuda()

        if self.indices_file is not None:
            self.indices_file.flush()
            self.indices_file.close()
        if self._gc_init_state:
            gc.enable()
        else:
            gc.disable()
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code, quiet=True)

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_val, exc_tb
        self.close(0 if exc_type is None else 1)
