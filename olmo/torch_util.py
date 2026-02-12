import gc
import logging
import math
import os
from functools import lru_cache
from typing import Dict, Optional, TypeVar, Union

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp.wrap import always_wrap_policy
from torch.distributed.tensor import DTensor, distribute_tensor

T = TypeVar("T")


@lru_cache(maxsize=128)
def log_once(
    logger: logging.Logger, msg: str, *args, level: int = logging.INFO, **kwargs
):
    logger.log(level, msg, *args, **kwargs)


def seed_all(seed: int):
    """Seed all rng objects."""
    import random

    import numpy as np

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_node_rank() -> int:
    return int(
        os.environ.get("NODE_RANK")
        or (get_global_rank() - get_local_rank()) // get_local_world_size()
    )


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    if is_distributed():
        return dist.get_world_size(group)
    else:
        return 1


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE") or 1)


def get_global_rank() -> int:
    if is_distributed():
        return int(os.environ.get("RANK") or dist.get_rank())
    else:
        return 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or 0)


def get_fs_local_rank() -> int:
    """Get the local rank per filesystem, meaning that, regardless of the number of nodes,
    if all ranks share the same filesystem then `get_fs_local_rank()` will be equivalent to `get_global_rank()`,
    but if nodes do not share the same filesystem then `get_fs_local_rank()` will be equivalent to `get_local_rank()`.
    """
    if os.environ.get("OLMO_SHARED_FS"):
        return int(os.environ.get("FS_LOCAL_RANK") or get_global_rank())
    else:
        return int(os.environ.get("FS_LOCAL_RANK") or get_local_rank())


def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o  # type: ignore


def ensure_finite_(
    x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False
):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def ensure_multiple_of(x: int, of: int) -> int:
    return of * math.ceil(x / of)


def get_default_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def peak_gpu_memory(reset: bool = False) -> Optional[float]:
    """
    Get the peak GPU memory usage in MB across all ranks.
    Only rank 0 will get the final result.
    """
    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    peak_mb = torch.cuda.max_memory_allocated(device) / 1000000
    if is_distributed():
        peak_mb_tensor = torch.tensor(peak_mb, device=device)
        dist.reduce(peak_mb_tensor, 0, dist.ReduceOp.MAX)
        peak_mb = peak_mb_tensor.item()

    if reset:
        # Reset peak stats.
        torch.cuda.reset_max_memory_allocated(device)

    return peak_mb


V = TypeVar("V", bool, int, float)


def synchronize_value(value: V, device: torch.device) -> V:
    if dist.is_available() and dist.is_initialized():
        value_tensor = torch.tensor(value, device=device)
        dist.broadcast(value_tensor, 0)
        return value_tensor.item()  # type: ignore
    else:
        return value


def synchronize_flag(flag: bool, device: torch.device) -> bool:
    return synchronize_value(flag, device)


def gc_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_cumulative_document_lengths(doc_lens: torch.Tensor) -> torch.Tensor:
    """
    Transform a batched tensor of document lengths into a 1D tensor of cumulative document
    lengths for the whole batch.
    """
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=doc_lens.device),
            torch.cumsum(doc_lens.masked_select(doc_lens != 0), 0, dtype=torch.int32),
        ]
    )


class _HiddenTensor:
    def __init__(self, x: torch.Tensor):
        self.x = x

    @property
    def device(self) -> torch.device:
        return self.x.device

    def to(self, *args, **kwargs) -> "_HiddenTensor":
        return _HiddenTensor(self.x.to(*args, **kwargs))


def hide_from_torch(x: torch.Tensor) -> _HiddenTensor:
    return _HiddenTensor(x)


def unhide_from_torch(x: Union[torch.Tensor, _HiddenTensor]) -> torch.Tensor:
    if isinstance(x, _HiddenTensor):
        return x.x
    else:
        return x


class SingleAccelerator(torch.nn.Module):
    process_group = None

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def get_local_tensor(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, DTensor):
        x = x.to_local()
        # An `AsyncCollectiveTensor` might be returned, which means the local tensor is not ready
        # yet (i.e. communication is not finished). In this case we need to call `.wait()`
        # to wait the local tensor to be ready.
        if hasattr(x, "wait"):
            return x.wait()  # type: ignore
        else:
            return x
    else:
        return x


def get_full_tensor(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, DTensor):
        return x.full_tensor()
    else:
        return x


def distribute_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not isinstance(source, DTensor):
        return get_full_tensor(target)

    if isinstance(target, DTensor):
        if (
            target.device_mesh == source.device_mesh
            and target.placements == source.placements
        ):
            return target
        else:
            return target.redistribute(
                device_mesh=source.device_mesh, placements=source.placements
            )

    return distribute_tensor(
        target, device_mesh=source.device_mesh, placements=source.placements
    )


def get_layer_wise_gradient_norm(model: FSDP, fsdp_model: FSDP) -> Dict[str, float]:
    """
    Compute the layer-wise gradient norm for an FSDP model.

    This method computes the gradient norm for each parameter, aggregates them by layer,
    and returns a dictionary mapping layer names to their gradient norms.
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            layer_name = name.rsplit(".", 1)[0]
            if layer_name not in norms:
                norms[layer_name] = 0.0
            norms[layer_name] += grad_norm**2

    # Each rank has a subset of the gradients, so we need to aggregate them.
    for layer_name in norms:
        norms[layer_name] = norms[layer_name] ** 0.5
        # Convert to a tensor for all_reduce.
        norm_tensor = torch.tensor(norms[layer_name], device=fsdp_model.device)
        dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM)
        norms[layer_name] = norm_tensor.item()

    return norms


def is_local_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
