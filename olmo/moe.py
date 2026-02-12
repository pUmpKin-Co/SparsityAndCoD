import logging
import math
import warnings
from abc import abstractmethod
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union, cast

import olmo_core.ops.moe as ops
import torch
import torch.distributed as dist
import torch.nn as nn
from olmo_core.distributed.parallel import (
    flatten_mesh,
    get_device_mesh_info,
    get_pp_stage_mesh,
    get_world_mesh,
)
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.moe.loss import load_balancing_loss, router_z_loss
from olmo_core.nn.utils import get_tp_wrappers
from olmo_core.ops import attach_auxiliary_loss
from olmo_core.train.common import ReduceType
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Placement, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    PrepareModuleOutput,
    parallelize_module,
)
from torch.nn import functional as F

from .config import (
    MoELoadBalancingLossGranularity,
    MoERouterGatingFunction,
    MoERouterType,
)
from .exceptions import OLMoConfigurationError
from .torch_util import (
    _HiddenTensor,
    distribute_like,
    ensure_multiple_of,
    get_default_device,
    get_local_tensor,
    get_world_size,
    hide_from_torch,
    is_distributed,
    log_once,
    move_to_device,
    unhide_from_torch,
)

try:
    import grouped_gemm  # type: ignore

    gmm = grouped_gemm.ops.gmm
except ImportError:
    gmm = None

log = logging.getLogger(__name__)


class PermutedAllToAllOutput(NamedTuple):
    parallel_x: torch.Tensor
    parallel_indices: torch.Tensor
    parallel_bin_ids: Optional[torch.Tensor]
    parallel_bins: torch.Tensor
    parallel_batch_size_per_expert: torch.Tensor
    recv_counts: Optional[List[int]]
    send_counts: Optional[List[int]]
    expert_capacity: int
    handle: dist.Work


class FeedForward(nn.Module):
    """
    Basic feed-forward module with SwiGLU activation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(
            d_model, hidden_size, bias=bias, dtype=dtype, device=init_device
        )
        self.w2 = nn.Linear(
            hidden_size, d_model, bias=bias, dtype=dtype, device=init_device
        )
        self.w3 = nn.Linear(
            d_model, hidden_size, bias=bias, dtype=dtype, device=init_device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the feed-forward on the input ``x``.

        :param x: The input of shape ``(*, d_model)``.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(
            float8_enabled=float8_enabled
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=prepare_module_input(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Replicate(),),
            ),
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan={
                "w1": colwise_parallel(),
                "w2": rowwise_parallel(
                    output_layouts=output_layout, use_local_output=use_local_output
                ),
                "w3": colwise_parallel(),
            },
        )


class MoEMLPBase(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        num_experts: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        self.num_local_experts = num_experts
        self.hidden_sharding_degree = 1
        self.ep_mesh: Optional[DeviceMesh] = None
        self.ep_pg: Optional[dist.ProcessGroup] = None

    def apply_ep(self, ep_mesh: DeviceMesh):
        """
        Apply expert parallelism.

        :param ep_mesh: A 1D device mesh to shard experts over.
        """
        if ep_mesh.ndim != 1:
            raise RuntimeError("expert parallel mesh must be 1 dimensional")
        self._shard_experts(ep_mesh)

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        """
        Apply expert parallelism.

        :param tp_mesh: A 1D device mesh to shard experts over.
        """
        del float8_enabled  # TODO
        if tp_mesh.ndim != 1:
            raise RuntimeError("tensor parallel mesh must be 1 dimensional")
        self._shard_experts(tp_mesh)

    def _shard_experts(self, mesh: DeviceMesh):
        num_shards = mesh.size()
        if self.num_experts % num_shards != 0:
            raise OLMoConfigurationError(
                f"'num_experts' ({self.num_experts}) must be divisible by the expert parallel shard degree ({num_shards})."
            )

        self.ep_mesh = mesh
        self.ep_pg = mesh.get_group()
        self.num_local_experts = self.num_experts // num_shards

        placements: List[Placement] = [Shard(0)]
        self.register_parameter("w1", nn.Parameter(distribute_tensor(self.w1, mesh, placements)))  # type: ignore
        self.register_parameter("w2", nn.Parameter(distribute_tensor(self.w2, mesh, placements)))  # type: ignore
        self.register_parameter("w3", nn.Parameter(distribute_tensor(self.w3, mesh, placements)))  # type: ignore

    def prepare_experts_for_fsdp(self, *, world_mesh: DeviceMesh, **kwargs):
        """
        Should be called before wrapping this module, or a parent module, with FSDP2.
        """
        # If expert/tensor parallel is not enabled then we don't need to do anything special here.
        if self.ep_mesh is None:
            return

        if self.ep_mesh.mesh_dim_names is None:
            raise RuntimeError("mesh must have named dimensions!")

        if (dim_names := world_mesh.mesh_dim_names) is None:
            raise RuntimeError("mesh must have named dimensions!")

        # If the experts are already sharded over a data parallel dimension, we need to shard them
        # over the other data parallel dimension, otherwise `fully_shard` called with the full DP
        # mesh won't handle this module correctly.
        if (ep_mesh_dim_name := self.ep_mesh.mesh_dim_names[0]).startswith("dp"):
            # Shard local experts over the adjacent DP dimension.
            dp_replicate_dim_name = dim_names[dim_names.index(ep_mesh_dim_name) - 1]
            dp_replicate_mesh = world_mesh[dp_replicate_dim_name]

            log_once(
                log,
                f"Sharding local experts over {get_device_mesh_info(dp_replicate_mesh)}...",
            )
            fully_shard(self, mesh=dp_replicate_mesh, **kwargs)

    def prepare_experts_for_ddp(self, *, world_mesh: DeviceMesh):
        """
        Should be called before wrapping this module, or a parent module, with FSDP2.
        """
        # TODO: do we need to do anything special here like with FSDP?
        del world_mesh
        pass


class MoEMLP(MoEMLPBase):
    """
    A basic expert MLP module with SwiGLU activation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__(
            d_model=d_model, hidden_size=hidden_size, num_experts=num_experts
        )
        # NOTE: these parameters need to have a large enough first dimension (which would be num experts)
        # in order to be sharded over big world sizes with FSDP, so we flatten the first 2 dimensions.
        self.w1 = nn.Parameter(
            torch.empty(
                num_experts * d_model,
                hidden_size,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.w2 = nn.Parameter(
            torch.empty(
                num_experts * hidden_size,
                d_model,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.w3 = nn.Parameter(
            torch.empty(
                num_experts * d_model,
                hidden_size,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for w in (self.w1, self.w2, self.w3):
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

    def extra_repr(self):
        return f"num_experts={self.num_experts}, in_features={self.d_model}, hidden_size={self.hidden_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the expert outputs.

        :param x: The input of shape ``(num_local_experts, N, d_model)``.
        """
        og_dtype = x.dtype

        # Scale gradients and get local tensors (in case of expert parallelism).
        # shape (all): (num_local_experts, hidden_size, d_model)
        w1, w2, w3 = (
            get_local_tensor(
                self.w1.view(self.num_experts, self.d_model, self.hidden_size)
            ),
            get_local_tensor(
                self.w2.view(self.num_experts, self.hidden_size, self.d_model)
            ),
            get_local_tensor(
                self.w3.view(self.num_experts, self.d_model, self.hidden_size)
            ),
        )

        x = x.type_as(w1)

        # Compute the MLP.
        return torch.bmm(F.silu(torch.bmm(x, w1)) * torch.bmm(x, w3), w2).to(
            dtype=og_dtype
        )


class ParallelMLPBase(nn.Module):
    """
    Wraps an MoE MLP layer to coordinate the routing and expert parallelism.
    """

    def __init__(
        self, *, mlp: MoEMLPBase, top_k: int, cache: Optional[BufferCache] = None
    ):
        super().__init__()
        self.mlp = mlp
        self.top_k = top_k
        self._cache = cache or BufferCache()
        self._expert_parallel_enabled: bool = False

    def warmup_cache(self, max_local_microbatch_size: int):
        del max_local_microbatch_size

    @property
    def d_model(self) -> int:
        return self.mlp.d_model

    @property
    def num_experts(self) -> int:
        return self.mlp.num_experts

    @property
    def num_local_experts(self) -> int:
        return self.mlp.num_local_experts

    @property
    def hidden_sharding_degree(self) -> int:
        return self.mlp.hidden_sharding_degree

    @property
    def ep_world_size(self) -> int:
        if self.ep_pg is not None:
            return get_world_size(self.ep_pg)
        else:
            return 1

    @property
    def ep_pg(self) -> Optional[dist.ProcessGroup]:
        return self.mlp.ep_pg

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        """
        Apply expert parallelism.
        """
        self.mlp.apply_ep(ep_mesh, **kwargs)
        self._expert_parallel_enabled = True

    def apply_tp(self, tp_mesh: DeviceMesh, **kwargs):
        """
        Apply tensor parallelism.
        """
        self.mlp.apply_tp(tp_mesh, **kwargs)
        self._expert_parallel_enabled = True

    def prepare_experts_for_fsdp(self, **kwargs):
        """
        Should be called before wrapping this module with FSDP2.
        """
        self.mlp.prepare_experts_for_fsdp(**kwargs)

    def prepare_experts_for_ddp(self, **kwargs):
        """
        Should be called before wrapping this module with DDP2.
        """
        self.mlp.prepare_experts_for_ddp(**kwargs)

    def indices_and_bins(
        self,
        expert_indices: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param expert_indices: A 1D tensor.
        :param batch_size_per_expert: A 1D tensor.
        """
        expert_indices = expert_indices.int()

        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        # shape: (batch_size,), (batch_size,)
        # TODO: for non-dropless MoE, should do secondary sort by expert weight so we drop tokens
        # with the lowest expert weight.
        bin_ids, indices = torch.sort(expert_indices)

        # Calculate the bin bounds for the sorted items/tokens.
        # shape: (num_experts,)
        bins = torch.empty_like(batch_size_per_expert, dtype=torch.int32)
        torch.cumsum(batch_size_per_expert, 0, out=bins)

        return indices.int(), bin_ids, bins

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: The input of shape ``(N, d_model)``.
        :param expert_weights: Expert weights of shape ``(N, top_k)``.
        :param expert_indices: The indices of the top-k experts, shape ``(N, top_k)``.
        :param batch_size_per_expert: The number of items routed to each expert, shape ``(num_experts,)``.

        :returns: The output with the same shape as ``x``.
        """
        x, expert_weights, expert_indices, batch_size_per_expert = (
            get_local_tensor(x),
            get_local_tensor(expert_weights),
            get_local_tensor(expert_indices),
            get_local_tensor(batch_size_per_expert),
        )

        in_shape = x.size()

        # shape: (N, d_model)
        x = x.view(-1, x.shape[-1])
        # shape: (batch_size * top_k,)
        expert_weights = expert_weights.flatten()
        # shape: (batch_size * top_k,)
        expert_indices = expert_indices.flatten()

        with torch.no_grad():
            indices, bin_ids, bins = self.indices_and_bins(
                expert_indices, batch_size_per_expert
            )

        # Compute the experts.
        if not self._expert_parallel_enabled:
            x = self.forward_once(
                x,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
                indices=indices,
                bin_ids=bin_ids,
                bins=bins,
                batch_size_per_expert=batch_size_per_expert,
            )
        else:
            x = self.parallel_forward_once(
                x,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
                indices=indices,
                bin_ids=bin_ids,
                bins=bins,
                batch_size_per_expert=batch_size_per_expert,
            )

        return x.view(in_shape)

    @abstractmethod
    def forward_once(
        self,
        x: torch.Tensor,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: The input of shape ``(*, d_model)``, typically ``(num_docs, seq_len, d_model)``
            such that ``num_docs x seq_len = batch_size``.
        :param expert_weights: Expert weights of shape ``(batch_size, top_k)``, where ``batch_size``
            typically equals ``num_docs x seq_len``.
        :param expert_indices: The indices of the top-k experts, shape ``(batch_size, top_k)``.
        """
        raise NotImplementedError

    def parallel_forward_once(
        self,
        x: torch.Tensor,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: The input of shape ``(*, d_model)``.
        :param expert_weights: Expert weights of shape ``(N, top_k)``, where ``N``
            typically equals ``batch_size x seq_len``.
        :param expert_indices: The indices of the top-k experts, shape ``(N, top_k)``.
        """
        # NOTE: This function implements the same computation as forward_once
        # but with expert model parallelism.
        #
        # 1. Permute the tokens locally so that they are grouped by their
        # expert assignments. This allows us to transfer all of the tokens
        # for a remote device in one communication primitive.
        #
        # 2. Permute the tokens across the expert parallel devices. After
        # this is completed each device has all of the tokens assigned to
        # its set of experts in its local HBM.
        #
        # 3. Permute the tokens locally so that they are grouped by their
        # expert assignment. After the distributed permutation the tokens
        # are grouped by which device they came from. We re-order them
        # locally to allow for efficient computation.
        #
        # After this series of permutations we compute the linear layers
        # and then repeat these three steps in reverse to produce the final
        # output.

        (
            parallel_x,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            parallel_batch_size_per_expert,
            recv_counts,
            send_counts,
            expert_capacity,
            parallel_x_handle,
        ) = self.permute_and_all_to_all(
            x,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
            batch_size_per_expert=batch_size_per_expert,
        )

        parallel_x_handle.wait()
        parallel_x = self.compute_local_experts(
            parallel_x,
            parallel_indices=parallel_indices,
            parallel_bin_ids=parallel_bin_ids,
            parallel_bins=parallel_bins,
            parallel_batch_size_per_expert=parallel_batch_size_per_expert,
            expert_capacity=expert_capacity,
        )

        x, x_handle = self.reverse_all_to_all(
            parallel_x, send_counts=send_counts, recv_counts=recv_counts
        )

        x_handle.wait()

        x = self.unpermute(
            x,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
        )
        return x

    @abstractmethod
    def permute_and_all_to_all(
        self,
        x: torch.Tensor,
        *,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> PermutedAllToAllOutput:
        raise NotImplementedError

    @abstractmethod
    def compute_local_experts(
        self,
        parallel_x,
        *,
        parallel_indices: torch.Tensor,
        parallel_bin_ids: Optional[torch.Tensor],
        parallel_bins: torch.Tensor,
        parallel_batch_size_per_expert: torch.Tensor,
        expert_capacity: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def reverse_all_to_all(
        self,
        parallel_x: torch.Tensor,
        *,
        send_counts: Optional[List[int]],
        recv_counts: Optional[List[int]],
    ) -> Tuple[torch.Tensor, dist.Work]:
        raise NotImplementedError

    @abstractmethod
    def unpermute(
        self,
        x,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
    ):
        raise NotImplementedError


class ParallelMLP(ParallelMLPBase):
    def __init__(
        self,
        *,
        mlp: MoEMLP,
        top_k: int,
        capacity_factor: float,
        cache: Optional[BufferCache] = None,
        max_local_microbatch_size: Optional[int] = None,
    ):
        super().__init__(mlp=mlp, top_k=top_k, cache=cache)
        self.capacity_factor = capacity_factor
        self.tp_degree: int = 1
        self.max_local_microbatch_size = max_local_microbatch_size
        if self.max_local_microbatch_size is not None:
            self.warmup_cache(self.max_local_microbatch_size)

    def warmup_cache(self, max_local_microbatch_size: int):
        self.max_local_microbatch_size = max_local_microbatch_size
        expert_capacity = self.expert_capacity(
            self.max_local_microbatch_size // self.tp_degree
        )
        local_expert_capacity = expert_capacity // self.ep_world_size
        self._get_parallel_indices_and_bins(
            expert_capacity=expert_capacity,
            local_expert_capacity=local_expert_capacity,
            device=get_default_device(),
        )

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        super().apply_ep(ep_mesh, **kwargs)
        if self.max_local_microbatch_size is not None:
            self.warmup_cache(self.max_local_microbatch_size)

    def apply_tp(self, tp_mesh: DeviceMesh, **kwargs):
        super().apply_tp(tp_mesh, **kwargs)
        self.tp_degree = tp_mesh.size()
        if self.max_local_microbatch_size is not None:
            self.warmup_cache(self.max_local_microbatch_size)

    def expert_capacity(self, local_batch_size: int) -> int:
        # NOTE: need to ensure this is the same across the process group.
        # If local batch sizes are different then these will be different, and `parallel_forward_once`
        # will break. This shouldn't be a problem with our trainer, but would be an issue for inference.
        # To avoid that you could set `self.max_local_microbatch_size` up-front.
        if self.max_local_microbatch_size is not None:
            max_local_microbatch_size = self.max_local_microbatch_size // self.tp_degree
            if local_batch_size > max_local_microbatch_size:
                raise RuntimeError(
                    f"Local batch size ({local_batch_size:d}) bigger than "
                    f"configured max local batch size ({max_local_microbatch_size:d})"
                )
            else:
                local_batch_size = max_local_microbatch_size

        ideal_local_inputs_per_expert = self.top_k * local_batch_size / self.num_experts
        allowed_local_inputs_per_expert = ensure_multiple_of(
            int(self.capacity_factor * ideal_local_inputs_per_expert), 8
        )
        return self.ep_world_size * allowed_local_inputs_per_expert

    @torch.no_grad()
    def _get_parallel_indices_and_bins(
        self, *, expert_capacity: int, local_expert_capacity: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices_cache_key = (
            f"moe_par_expert_indices_{expert_capacity}_{local_expert_capacity}"
        )
        bins_cache_key = (
            f"moe_par_expert_bins_{expert_capacity}_{local_expert_capacity}"
        )

        if (
            parallel_indices := self._cache.get_for_device(indices_cache_key, device)
        ) is not None and (
            parallel_bins := self._cache.get_for_device(bins_cache_key, device)
        ) is not None:
            return parallel_indices, parallel_bins

        # Construct the expert indices for the permuted tokens.
        # shape: (num_experts,) = (num_local_experts * ep_world_size,)
        parallel_top_expert = torch.remainder(
            torch.arange(
                self.num_experts * self.hidden_sharding_degree,
                dtype=torch.int32,
                device=device,
            ),
            self.num_local_experts,
        )

        # shape: (num_local_experts * ep_world_size * local_expert_capacity,)
        #      = (num_local_experts * expert_capacity,)
        parallel_top_expert = torch.repeat_interleave(
            parallel_top_expert,
            local_expert_capacity,
            output_size=parallel_top_expert.numel() * local_expert_capacity,
        )

        # shape: (num_local_experts * expert_capacity,)
        _, parallel_indices = torch.sort(parallel_top_expert)
        parallel_indices = parallel_indices.int()

        # Calculate the bins boundaries from the token counts.
        # shape: (num_local_experts,)
        parallel_batch_size_per_expert = move_to_device(
            torch.tensor([expert_capacity] * self.num_local_experts),
            parallel_indices.device,
        )
        # shape: (num_local_experts,)
        parallel_bins = torch.empty_like(
            parallel_batch_size_per_expert, dtype=torch.int32
        )
        torch.cumsum(parallel_batch_size_per_expert, 0, out=parallel_bins)

        self._cache[indices_cache_key] = parallel_indices
        self._cache[bins_cache_key] = parallel_bins

        return parallel_indices, parallel_bins

    def forward_once(
        self,
        x: torch.Tensor,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        del bin_ids, batch_size_per_expert, expert_indices

        batch_size = expert_weights.numel() // self.top_k
        expert_capacity = self.expert_capacity(batch_size)

        x = self.permute_and_compute(
            x,
            indices=indices,
            expert_weights=expert_weights,
            bins=bins,
            expert_capacity=expert_capacity,
            top_k=self.top_k,
        )
        return x

    def permute_and_all_to_all(
        self,
        x: torch.Tensor,
        *,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> PermutedAllToAllOutput:
        del bin_ids

        expert_capacity = self.expert_capacity(x.shape[0])
        local_expert_capacity = expert_capacity // self.ep_world_size

        # Permute locally so that the tokens for each device are stored contiguously.
        # shape: (num_experts, local_expert_capacity, d_model)
        x = ops.binned_gather(x, indices, bins, local_expert_capacity, self.top_k)

        # If we're sharding the experts along the hidden dimension
        # multiple devices own parts of the same sets of experts.
        # Replicate the token counts so devices that share experts
        # get all of the tokens assigned to them.
        if self.hidden_sharding_degree > 1:
            # shape: (num_local_experts, ep_world_size // hidden_sharding_degree, local_expert_capacity, d_model)
            x = x.view(self.num_local_experts, -1, local_expert_capacity, self.d_model)
            # shape: (num_experts * hidden_sharding_degree, local_expert_capacity, d_model)
            x = x.repeat(1, self.hidden_sharding_degree, 1, 1).view(
                -1, local_expert_capacity, self.d_model
            )

        # After we do the cross-device permutation we have the tokens on the
        # correct device but not yet grouped by expert because we received
        # tokens from each device as contiguous chunks. To group the tokens
        # for expert computation we'll do one more local permutation.
        # shape (both): (num_local_experts,)
        parallel_indices, parallel_bins = self._get_parallel_indices_and_bins(
            expert_capacity=expert_capacity,
            local_expert_capacity=local_expert_capacity,
            device=x.device,
        )

        # Start the cross-device permutation asynchronously so we can
        # overlap communication with computation.
        # shape: (num_local_experts * ep_world_size, local_expert_capacity, d_model)
        #     ~= (num_local_experts, expert_capacity, d_model)
        parallel_x, handle = ops.all_to_all(x, group=self.ep_pg, async_op=True)

        return PermutedAllToAllOutput(
            parallel_x,
            parallel_indices,
            None,
            parallel_bins,
            batch_size_per_expert,
            None,
            None,
            expert_capacity,
            handle,
        )

    def compute_local_experts(
        self,
        parallel_x,
        *,
        parallel_indices: torch.Tensor,
        parallel_bin_ids: Optional[torch.Tensor],
        parallel_bins: torch.Tensor,
        parallel_batch_size_per_expert: torch.Tensor,
        expert_capacity: int,
    ) -> torch.Tensor:
        assert parallel_bin_ids is None
        del parallel_batch_size_per_expert

        # Locally permute the tokens and perform the expert computation.
        parallel_x = self.permute_and_compute(
            parallel_x,
            indices=parallel_indices,
            expert_weights=None,
            bins=parallel_bins,
            expert_capacity=expert_capacity,
            top_k=1,
        )

        return parallel_x

    def reverse_all_to_all(
        self,
        parallel_x: torch.Tensor,
        *,
        send_counts: Optional[List[int]],
        recv_counts: Optional[List[int]],
    ) -> Tuple[torch.Tensor, dist.Work]:
        assert send_counts is None
        assert recv_counts is None

        # Un-permute the tokens across the devices.
        x, handle = ops.all_to_all(parallel_x, group=self.ep_pg, async_op=True)
        return x, handle

    def unpermute(
        self,
        x,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
    ):
        del expert_indices, bin_ids

        # Reduce along the hidden sharding to get the final outputs.
        if self.hidden_sharding_degree > 1:
            x = ops.sum_tensor(
                x.view(self.hidden_sharding_degree, -1, self.d_model), dim=0
            )

        # Un-permute locally to setup for the next series of operations.
        x = ops.binned_scatter(
            x.view(self.num_experts, -1, self.d_model),
            indices,
            expert_weights,
            bins,
            self.top_k,
        )

        return x

    def permute_and_compute(
        self,
        x: torch.Tensor,
        *,
        indices: torch.Tensor,
        expert_weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        expert_capacity: int,
        top_k: int,
    ) -> torch.Tensor:
        x = x.view(-1, x.shape[-1])

        # Route the tokens for MoE computation.
        # shape: (num_experts, expert_capacity, d_model)
        x = ops.binned_gather(x, indices, bins, expert_capacity, top_k)

        # Perform the expert computation.
        # shape: (num_experts, expert_capacity, d_model)
        x = self.mlp(x)

        # Un-route the data for the MoE output. Items that were dropped will be zeroed out.
        # shape: (N, d_model)
        x = ops.binned_scatter(x, indices, expert_weights, bins, top_k)
        return x


class ParallelDroplessMLP(ParallelMLPBase):
    """
    A dropless implementation of a :class:`ParallelMLP`.

    .. warning::
        When expert parallelism is enabled the forward pass involves a host-device sync.
    """

    def __init__(self, *, mlp, top_k: int, cache: Optional[BufferCache] = None):
        super().__init__(mlp=mlp, top_k=top_k, cache=cache)

    def forward_once(
        self,
        x: torch.Tensor,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        del expert_indices
        return self.permute_and_compute(
            x,
            batch_size_per_expert=batch_size_per_expert,
            indices=indices,
            bin_ids=bin_ids,
            expert_weights=expert_weights,
            bins=bins,
            top_k=self.top_k,
        )

    @torch._dynamo.disable()  # TODO: might be able to relax this, or be more fine-grained
    def permute_and_all_to_all(
        self,
        x: torch.Tensor,
        *,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> PermutedAllToAllOutput:
        with torch.no_grad():
            # If we're sharding the experts along the hidden dimension
            # multiple devices own parts of the same sets of experts.
            # Replicate the token counts so every device gets the counts.
            repeated_batch_size_per_expert = ops.repeat(
                batch_size_per_expert,
                (self.hidden_sharding_degree,),
            )

            # Pass token count information to the device on which the
            # target expert resides.
            parallel_batch_size_per_expert = torch.empty_like(
                repeated_batch_size_per_expert,
            )
            tpe_handle = dist.all_to_all_single(
                parallel_batch_size_per_expert,
                repeated_batch_size_per_expert,
                group=self.ep_pg,
                async_op=True,
            )
            assert tpe_handle is not None

        # Permute locally and without any padding so that tokens for each
        # parallel device are stored contiguously.
        x = ops.gather(x.view(-1, x.shape[-1]), indices, bin_ids, bins, self.top_k)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()

            # Reshape to (ep_world_size, num_local_experts).
            repeated_batch_size_per_expert = repeated_batch_size_per_expert.view(
                self.ep_world_size, self.num_local_experts
            )
            parallel_batch_size_per_expert = parallel_batch_size_per_expert.view(
                self.ep_world_size, self.num_local_experts
            )

            # NOTE: host-device sync here.
            send_counts = repeated_batch_size_per_expert.sum(dim=-1).cpu().tolist()
            recv_counts = parallel_batch_size_per_expert.sum(dim=-1).cpu().tolist()
            tokens_received = sum(recv_counts)

        # If we're sharding the experts along the hidden dimension
        # multiple devices own parts of the same sets of experts.
        # Replicate the token counts so devices that share experts
        # get all of the tokens assigned to them.
        x = ops.repeat(x, (self.hidden_sharding_degree, 1))

        with torch.no_grad():
            # After we do the cross-device permutation we have the tokens on the
            # correct device but not yet grouped by expert because we received
            # tokens from each device as contiguous chunks. To group the tokens
            # for expert computation we'll do one more local permutation. The
            # rest of this torch.no_grad() scope sets up the indices and bins
            # for this permutation.

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(
                    self.num_experts * self.hidden_sharding_degree,
                    dtype=torch.int32,
                    device=indices.device,
                ),
                self.num_local_experts,
            )

            parallel_top_expert = torch.repeat_interleave(
                parallel_top_expert,
                parallel_batch_size_per_expert.flatten(),
                output_size=tokens_received,
            )

            parallel_bin_ids, parallel_indices = torch.sort(parallel_top_expert)

            # Calculate the bins boundaries from the token counts.
            parallel_batch_size_per_expert = parallel_batch_size_per_expert.sum(
                dim=0,
                dtype=torch.long,
            )
            parallel_bins = torch.empty_like(
                parallel_batch_size_per_expert, dtype=torch.int32
            )
            torch.cumsum(parallel_batch_size_per_expert, 0, out=parallel_bins)

        # Start the cross-device permutation asynchronously so we can
        # overlap communication with computation.
        parallel_x, parallel_x_handle = ops.all_to_all(
            x,
            recv_counts,
            send_counts,
            group=self.ep_pg,
            async_op=True,
        )

        return PermutedAllToAllOutput(
            parallel_x,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            parallel_batch_size_per_expert,
            recv_counts,
            send_counts,
            -1,
            parallel_x_handle,
        )

    @torch._dynamo.disable()  # TODO: might be able to relax this, or be more fine-grained
    def compute_local_experts(
        self,
        parallel_x,
        *,
        parallel_indices: torch.Tensor,
        parallel_bin_ids: Optional[torch.Tensor],
        parallel_bins: torch.Tensor,
        parallel_batch_size_per_expert: torch.Tensor,
        expert_capacity: int,
    ) -> torch.Tensor:
        assert parallel_bin_ids is not None
        del expert_capacity

        parallel_x = self.permute_and_compute(
            parallel_x,
            batch_size_per_expert=parallel_batch_size_per_expert,
            indices=parallel_indices.int(),
            bin_ids=parallel_bin_ids,
            expert_weights=None,
            bins=parallel_bins,
            top_k=1,
        )

        return parallel_x

    @torch._dynamo.disable()  # TODO: might be able to relax this, or be more fine-grained
    def reverse_all_to_all(
        self,
        parallel_x: torch.Tensor,
        *,
        send_counts: Optional[List[int]],
        recv_counts: Optional[List[int]],
    ) -> Tuple[torch.Tensor, dist.Work]:
        assert send_counts is not None
        assert recv_counts is not None

        # Un-permute the tokens across the devices.
        x, handle = ops.all_to_all(
            parallel_x,
            send_counts,
            recv_counts,
            group=self.ep_pg,
            async_op=True,
        )
        return x, handle

    @torch._dynamo.disable()  # TODO: might be able to relax this, or be more fine-grained
    def unpermute(
        self,
        x,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
    ):
        del expert_indices

        # Reduce along the hidden sharding to get the final outputs.
        x = ops.sum_tensor(x.view(self.hidden_sharding_degree, -1, self.d_model), dim=0)

        # Un-permute locally to setup for the next series of operations.
        x = ops.scatter(x, indices, bin_ids, expert_weights, bins, self.top_k)

        return x

    def permute_and_compute(
        self,
        x: torch.Tensor,
        *,
        batch_size_per_expert: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        expert_weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        x = x.view(-1, x.shape[-1])

        # Route the tokens for MoE computation.
        x = ops.gather(x, indices, bin_ids, bins, top_k)

        # Perform the expert computation.
        x = self.mlp(x, batch_size_per_expert)

        # Un-route the data for the MoE output.
        return ops.scatter(x, indices, bin_ids, expert_weights, bins, top_k)


class _UniformExpertAssignment(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, num_experts: int):
        del ctx
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)


_uniform_expert_assignment: Callable[
    [torch.Tensor, int], torch.Tensor
] = _UniformExpertAssignment.apply  # type: ignore


class MoERouter(nn.Module):
    """
    A base class for MoE router modules.

    :param d_model: The model dimensionality (hidden size).
    :param num_experts: The total number of experts.
    :param top_k: The number of experts to assign to each item/token.
    :param jitter_eps: Controls the amount of noise added to the input during training.
    :param normalize_expert_weights: The type of norm (e.g. ``2.0`` for L2 norm) to use to normalize
        the expert weights.
    :param uniform_expert_assignment: Force uniform assignment. Useful for benchmarking.
    :param bias_gamma: If set to a positive float, experts scores for top-k routing will be adjusted
        by a bias following the "auxiliary-loss-free load balancing" strategy from DeepSeek-v3.
        A reasonable value is on the order of 0.0001.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        top_k: int = 1,
        jitter_eps: Optional[float] = None,
        normalize_expert_weights: Optional[float] = None,
        uniform_expert_assignment: bool = False,
        bias_gamma: Optional[float] = None,
        gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax,
        lb_loss_weight: Optional[float] = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_eps = jitter_eps
        self.normalize_expert_weights = normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment
        self.bias_gamma = bias_gamma
        self.gating_function = gating_function
        self.lb_loss_weight = lb_loss_weight
        self.lb_loss_granularity = lb_loss_granularity
        self.z_loss_weight = z_loss_weight
        self.group: Optional[dist.ProcessGroup] = None
        self.cp_mesh: Optional[dist.DeviceMesh] = None
        self.tp_mesh: Optional[dist.DeviceMesh] = None

        if self.bias_gamma is not None:
            assert self.bias_gamma > 0
            self.register_buffer(
                "score_bias", torch.zeros(self.num_experts, device=init_device)
            )
        else:
            self.register_buffer("score_bias", None)

        # NOTE: we don't use buffers for t hese because we don't want FSDP to manage them, and we
        # don't use a BufferCache because `torch.compile()` doesn't handle that well when we're modifying
        # values in the cache.
        self._batch_size_per_expert = hide_from_torch(
            torch.zeros(self.num_experts, device=init_device)
        )
        self._score_bias_batch_size_per_expert: Optional[_HiddenTensor] = None
        self._load_balancing_loss: Optional[_HiddenTensor] = None
        self._z_loss: Optional[_HiddenTensor] = None

    def reset_parameters(self):
        self._batch_size_per_expert = hide_from_torch(
            torch.zeros(self.num_experts, device=self.device)
        )

        if self.bias_gamma is not None:
            assert self.score_bias is not None
            score_bias = cast(torch.Tensor, self.score_bias)
            score_bias.zero_()
            self._score_bias_batch_size_per_expert = hide_from_torch(
                torch.zeros(self.num_experts, device=self.device)
            )

        if self.lb_loss_weight is not None:
            self._load_balancing_loss = hide_from_torch(
                torch.zeros([], device=self.device)
            )

        if self.z_loss_weight is not None:
            self._z_loss = hide_from_torch(torch.zeros([], device=self.device))

    @property
    def device(self) -> torch.device:
        return get_default_device()

    @property
    def score_bias_batch_size_per_expert(self) -> Optional[torch.Tensor]:
        if self.bias_gamma is not None:
            if self._score_bias_batch_size_per_expert is None:
                self._score_bias_batch_size_per_expert = hide_from_torch(
                    torch.zeros(self.num_experts, device=self.device)
                )
            elif self._score_bias_batch_size_per_expert.device != self.device:
                self._score_bias_batch_size_per_expert = (
                    self._score_bias_batch_size_per_expert.to(self.device)
                )
        return (
            None
            if self._score_bias_batch_size_per_expert is None
            else unhide_from_torch(self._score_bias_batch_size_per_expert)
        )

    @score_bias_batch_size_per_expert.setter
    def score_bias_batch_size_per_expert(self, value: torch.Tensor):
        self._score_bias_batch_size_per_expert = hide_from_torch(value)

    @property
    def batch_size_per_expert(self) -> torch.Tensor:
        if self._batch_size_per_expert.device != self.device:
            self._batch_size_per_expert = self._batch_size_per_expert.to(self.device)
        return unhide_from_torch(self._batch_size_per_expert)

    @batch_size_per_expert.setter
    def batch_size_per_expert(self, value: torch.Tensor):
        self._batch_size_per_expert = hide_from_torch(value)

    @property
    def load_balancing_loss(self) -> Optional[torch.Tensor]:
        if self.lb_loss_weight is not None:
            if self._load_balancing_loss is None:
                self._load_balancing_loss = hide_from_torch(
                    torch.zeros([], device=self.device)
                )
            elif self._load_balancing_loss.device != self.device:
                self._load_balancing_loss = self._load_balancing_loss.to(self.device)
        return (
            None
            if self._load_balancing_loss is None
            else unhide_from_torch(self._load_balancing_loss)
        )

    @load_balancing_loss.setter
    def load_balancing_loss(self, value: torch.Tensor):
        self._load_balancing_loss = hide_from_torch(value)

    @property
    def z_loss(self) -> Optional[torch.Tensor]:
        if self.z_loss_weight is not None:
            if self._z_loss is None:
                self._z_loss = hide_from_torch(torch.zeros([], device=self.device))
            elif self._z_loss.device != self.device:
                self._z_loss = self._z_loss.to(self.device)
        return None if self._z_loss is None else unhide_from_torch(self._z_loss)

    @z_loss.setter
    def z_loss(self, value: torch.Tensor):
        self._z_loss = hide_from_torch(value)

    @torch.no_grad()
    def post_batch(self, dry_run: bool = False):
        if self.bias_gamma is None or not self.training:
            return

        assert self.score_bias is not None
        assert self.score_bias_batch_size_per_expert is not None
        score_bias = cast(torch.Tensor, self.score_bias)
        batch_size_per_expert = self.score_bias_batch_size_per_expert

        # Maybe reduce across the process group.
        if is_distributed():
            dist.all_reduce(batch_size_per_expert, group=self.group)

        ideal_batch_size_per_expert = batch_size_per_expert.mean(
            dim=0, keepdim=True, dtype=torch.float32
        )
        bias_delta = (
            self.bias_gamma
            * (ideal_batch_size_per_expert - batch_size_per_expert).sign()
        )
        # NOTE: have to be careful here to manage the case where `score_bias` is a DTensor.
        bias_delta = distribute_like(score_bias, bias_delta)

        if not dry_run:
            get_local_tensor(score_bias).add_(get_local_tensor(bias_delta))

        # Reset the accumulator.
        batch_size_per_expert.zero_()

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        if self.jitter_eps is None or not self.training:
            return x
        else:
            low = 1.0 - self.jitter_eps
            high = 1.0 + self.jitter_eps
            noise = torch.rand_like(x)
            return x * (low + noise * (high - low))

    def get_top_k(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        expert_weights: torch.Tensor
        expert_indices: torch.Tensor
        if self.bias_gamma is None:
            if self.top_k == 1:
                expert_weights, expert_indices = scores.max(dim=-1, keepdim=True)
            else:
                expert_weights, expert_indices = torch.topk(scores, self.top_k, dim=-1)
        else:
            assert self.score_bias is not None
            with torch.no_grad():
                _, expert_indices = torch.topk(
                    scores + self.score_bias.unsqueeze(0), self.top_k, dim=-1  # type: ignore
                )
            expert_weights = scores.gather(-1, expert_indices)

        if self.uniform_expert_assignment:
            expert_indices = _uniform_expert_assignment(
                expert_indices, self.num_experts
            )
            expert_weights = scores.gather(-1, expert_indices)

        return expert_weights, expert_indices

    @abstractmethod
    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the input ``x`` of shape ``(*, d_model)``, compute the un-normalized expert scores.

        :returns: The expert logits, shape ``(*, num_experts)``.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}

        # Load imbalance.
        batch_size_per_expert = self.batch_size_per_expert
        mean_batch_size = batch_size_per_expert.mean(dtype=torch.float)
        # Avoid division by zero when no data has been accumulated yet
        if mean_batch_size > 0:
            out["load imbalance"] = (
                batch_size_per_expert.max() / mean_batch_size,
                ReduceType.max,
            )
        else:
            out["load imbalance"] = (
                torch.tensor(
                    0.0, device=batch_size_per_expert.device, dtype=torch.float
                ),
                ReduceType.max,
            )

        # Load balancing loss.
        if self.lb_loss_weight is not None:
            assert self.load_balancing_loss is not None
            out["load balancing loss"] = (
                self.lb_loss_weight * self.load_balancing_loss,
                ReduceType.mean,
            )
            out["load balancing loss unscaled"] = (
                self.load_balancing_loss.clone(),
                ReduceType.mean,
            )

        # Router Z loss.
        if self.z_loss_weight is not None:
            assert self.z_loss is not None
            out["router Z loss"] = (self.z_loss_weight * self.z_loss, ReduceType.mean)
            out["router Z loss unscaled"] = (self.z_loss.clone(), ReduceType.mean)

        if reset:
            self.reset_metrics()

        return out

    def reset_metrics(self):
        if (bz_per_expert := self.batch_size_per_expert) is not None:
            bz_per_expert.zero_()
        if (lb_loss := self.load_balancing_loss) is not None:
            lb_loss.zero_()
        if (z_loss := self.z_loss) is not None:
            z_loss.zero_()

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Given the input ``x`` of shape ``(B, S, d_model)``, compute the experts assignment.

        :returns: The expert weights of shape ``(B, S, top_k)``,
            the expert indices of shape ``(B, S, top_k)``,
            the total number of items routed to each expert, with shape ``(num_experts,)``,
            and optionally the auxiliary losses.
        """
        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x).float()

        # shape: (batch_size, seq_len, num_experts)
        if self.gating_function == MoERouterGatingFunction.softmax:
            scores = logits.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = F.sigmoid(logits)
        else:
            raise NotImplementedError(self.gating_function)

        # shape: (batch_size, seq_len, top_k)
        expert_weights, expert_indices = self.get_top_k(scores)

        if self.normalize_expert_weights is not None:
            expert_weights = expert_weights.div(
                torch.norm(
                    expert_weights,
                    p=self.normalize_expert_weights,
                    dim=-1,
                    keepdim=True,
                )
            )

        with torch.no_grad():
            # Histogram the expert ids to identify the number of items/tokens routed to each expert.
            # shape: (batch_size, seq_len, num_experts)
            batched_batch_size_per_expert = ops.batched_histc(
                expert_indices, self.num_experts
            )
            # shape: (batch_size, num_experts)
            batched_batch_size_per_expert = batched_batch_size_per_expert.sum(dim=1)
            # shape: (num_experts,)
            batch_size_per_expert = batched_batch_size_per_expert.sum(dim=0)

        # Maybe compute auxiliary losses and accumulate metrics.
        aux_loss: Optional[torch.Tensor] = None
        if self.training and torch.is_grad_enabled():
            with torch.autocast(enabled=False, device_type=x.device.type):
                if self.lb_loss_weight is not None:
                    assert self.load_balancing_loss is not None

                    # Make sure scores are normalized, otherwise load balancing loss doesn't work well.
                    if self.gating_function == MoERouterGatingFunction.sigmoid:
                        scores = scores / scores.sum(dim=-1, keepdim=True)

                    lb_loss = load_balancing_loss(
                        num_experts=self.num_experts,
                        top_k=self.top_k,
                        expert_scores=scores,
                        batch_size_per_expert=batch_size_per_expert,
                        batched_batch_size_per_expert=batched_batch_size_per_expert,
                        granularity=self.lb_loss_granularity,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.load_balancing_loss += lb_loss.detach()

                    scaled_lb_loss = self.lb_loss_weight * lb_loss
                    aux_loss = scaled_lb_loss

                if self.z_loss_weight is not None:
                    assert self.z_loss is not None

                    z_loss = router_z_loss(
                        expert_logits=logits,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.z_loss += z_loss.detach()

                    scaled_z_loss = self.z_loss_weight * z_loss
                    aux_loss = (
                        scaled_z_loss if aux_loss is None else aux_loss + scaled_z_loss
                    )

            self.batch_size_per_expert += batch_size_per_expert
            if self.bias_gamma is not None:
                assert self.score_bias_batch_size_per_expert is not None
                self.score_bias_batch_size_per_expert += batch_size_per_expert

        return expert_weights, expert_indices, batch_size_per_expert, aux_loss

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        del float8_enabled
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Shard(1),),
                use_local_output=True,
            ),
        )
        self.tp_mesh = tp_mesh

    def apply_cp(self, cp_mesh: DeviceMesh):
        self.cp_mesh = cp_mesh


class MoELinearRouter(MoERouter):
    """
    A simple, learned, linear router.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(init_device=init_device, **kwargs)
        # NOTE: this parameter needs to have a large enough first dimension (which would be num experts)
        # in order to be sharded over big world sizes with FSDP. So we flatten it to a single dimension tensor.
        # And for that reason we don't support a 'bias' option.
        self.weight = nn.Parameter(
            torch.empty(
                self.num_experts * self.d_model, device=init_device, dtype=dtype
            )
        )
        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return (
            self.weight.device
            if self.weight.device.type != "meta"
            else torch.device("cpu")
        )

    def reset_parameters(self) -> None:
        super().reset_parameters()
        nn.init.trunc_normal_(self.weight, std=0.02, a=-3 * 0.02, b=3 * 0.02)

    def extra_repr(self):
        return f"in_features={self.d_model}, num_experts={self.num_experts}"

    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        weight = get_local_tensor(self.weight).view(self.num_experts, self.d_model)
        # Ensure dtype consistency between input and weight tensors
        x = x.type_as(weight)
        return F.linear(x, weight)

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        super().apply_tp(tp_mesh, float8_enabled=float8_enabled)
        self.register_parameter(
            "weight",
            nn.Parameter(distribute_tensor(self.weight, tp_mesh, [Replicate()])),
        )


class MoEBase(nn.Module):
    """
    Base class for MoE implementations.
    """

    def __init__(
        self,
        *,
        model_config,
        moe_config,
        dtype: torch.dtype = torch.float32,
        cache: Optional[BufferCache] = None,
        **kwargs,
    ):
        super().__init__()
        if moe_config.scale_loss_by_num_layers:
            if moe_config.lb_loss_weight is not None:
                moe_config.lb_loss_weight = (
                    moe_config.lb_loss_weight / model_config.n_layers
                )
            if moe_config.z_loss_weight is not None:
                moe_config.z_loss_weight = (
                    moe_config.z_loss_weight / model_config.n_layers
                )

        if moe_config.router_type == MoERouterType.linear:
            self.router = MoELinearRouter(
                d_model=model_config.d_model,
                num_experts=moe_config.num_experts,
                top_k=moe_config.top_k,
                jitter_eps=moe_config.jitter_eps,
                normalize_expert_weights=moe_config.normalize_expert_weights,
                uniform_expert_assignment=moe_config.uniform_expert_assignment,
                bias_gamma=moe_config.bias_gamma,
                gating_function=moe_config.gating_function,
                lb_loss_weight=moe_config.lb_loss_weight,
                lb_loss_granularity=moe_config.lb_loss_granularity,
                z_loss_weight=moe_config.z_loss_weight,
                init_device=model_config.init_device,
                dtype=dtype,
            )

        self.capacity_factor = moe_config.capacity_factor
        self.experts = self._init_parallel_mlp(
            d_model=model_config.d_model,
            num_experts=moe_config.num_experts,
            hidden_size=moe_config.hidden_size,
            dtype=dtype,
            init_device=model_config.init_device,
            cache=cache,
            **kwargs,
        )

        if moe_config.shared_mlp is not None:
            self.shared_mlp = FeedForward(
                d_model=model_config.d_model,
                hidden_size=moe_config.shared_mlp.hidden_size,
                bias=model_config.include_bias,
                dtype=dtype,
                init_device=model_config.init_device,
            )
        else:
            self.shared_mlp = None
        self._ep_enabled = False

    @property
    def num_experts(self) -> int:
        return self.router.num_experts

    @property
    def top_k(self) -> int:
        return self.router.top_k

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    def warmup_cache(self, max_local_microbatch_size: int):
        self.experts.warmup_cache(max_local_microbatch_size)

    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        return self.router.compute_metrics(reset=reset)

    def reset_metrics(self):
        self.router.reset_metrics()

    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        self.router.post_batch(dry_run=dry_run)

    @abstractmethod
    def _init_parallel_mlp(
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        **kwargs,
    ) -> ParallelMLPBase:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        """
        Run the MoE on the input ``x`` of shape ``(*, d_model)``.

        :param x: The input of shape ``(*, d_model)``.

        :returns: The output of the MoE layer, the optional load-balancing loss, and the optional
            router Z-loss.
        """
        expert_weights, expert_indices, batch_size_per_expert, router_aux_loss = (
            self.router(x, loss_div_factor=loss_div_factor)
        )

        if router_aux_loss is not None:
            x = attach_auxiliary_loss(x, router_aux_loss)

        shared_out: Optional[torch.Tensor] = None
        if self.shared_mlp is not None:
            shared_out = self.shared_mlp(x)

        out = self.experts(x, expert_weights, expert_indices, batch_size_per_expert)

        if shared_out is not None:
            shared_out = shared_out / (self.top_k + 1)
            out = shared_out.add(out, alpha=self.top_k / (self.top_k + 1))

        return out

    def apply_pp(self, pp_mesh: DeviceMesh):
        world_mesh = get_world_mesh()
        assert world_mesh is not None
        stage_mesh = get_pp_stage_mesh(world_mesh, pp_mesh)
        group = flatten_mesh(stage_mesh).get_group()
        self.router.group = group

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        """
        Apply expert parallelism.
        """
        self.experts.apply_ep(ep_mesh, **kwargs)
        self._ep_enabled = True

    def prepare_experts_for_fsdp(self, **kwargs):
        """
        Should be called before wrapping this module with FSDP2.
        """
        self.experts.prepare_experts_for_fsdp(**kwargs)

    def prepare_experts_for_ddp(self, **kwargs):
        """
        Should be called before wrapping this module with DDP2.
        """
        self.experts.prepare_experts_for_ddp(**kwargs)

    def apply_cp(self, cp_mesh: DeviceMesh):
        self.router.apply_cp(cp_mesh)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        # Sequence parallel for the most part.
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Shard(1),),
                use_local_output=False,
            ),
        )

        # Sequence parallel.
        self.router.apply_tp(tp_mesh, float8_enabled=float8_enabled)

        # Expert parallel.
        self.experts.apply_tp(tp_mesh, float8_enabled=float8_enabled)

        # Model parallel.
        if self.shared_mlp is not None:
            self.shared_mlp.apply_tp(
                tp_mesh,
                input_layout=Shard(1),
                output_layout=Shard(1),
                use_local_output=True,
                float8_enabled=float8_enabled,
            )

        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleOutput(
                output_layouts=(Shard(1),),
                desired_output_layouts=(output_layout or Replicate(),),
                use_local_output=use_local_output,
            ),
        )


class MoE(MoEBase):
    """
    A basic MoE implementation.
    """

    def __init__(
        self,
        *,
        model_config,
        moe_config,
        dtype: torch.dtype = torch.float32,
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            model_config=model_config,
            moe_config=moe_config,
            dtype=dtype,
            cache=cache,
        )

    def _init_parallel_mlp(  # type: ignore[override]
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> ParallelMLP:
        return ParallelMLP(
            mlp=MoEMLP(
                d_model=d_model,
                hidden_size=hidden_size,
                num_experts=num_experts,
                dtype=dtype,
                init_device=init_device,
            ),
            top_k=self.router.top_k,
            capacity_factor=self.capacity_factor,
            cache=cache,
        )


class DroplessMoEMLP(MoEMLPBase):
    """
    A dropless expert MLP module with SwiGLU activation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__(
            d_model=d_model, hidden_size=hidden_size, num_experts=num_experts
        )
        # NOTE: these parameters need to have a large enough first dimension (which would be num experts)
        # in order to be sharded over big world sizes with FSDP, so we flatten the first 2 dimensions.
        self.w1 = nn.Parameter(
            torch.empty(
                num_experts * hidden_size,
                d_model,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.w2 = nn.Parameter(
            torch.empty(
                num_experts * hidden_size,
                d_model,
                device=init_device,
                dtype=dtype,
            ),
        )
        self.w3 = nn.Parameter(
            torch.empty(
                num_experts * hidden_size,
                d_model,
                device=init_device,
                dtype=dtype,
            ),
        )

        self._gmm = gmm
        if self._gmm is None:
            warnings.warn(
                "Grouped GEMM not available, so the MoE will be substantially slower. "
                "Please install the 'grouped_gemm' package if possible.\n"
                "https://github.com/tgale96/grouped_gemm"
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for w in (self.w1, self.w2, self.w3):
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

    @torch._dynamo.disable()
    def gmm(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        batch_sizes: torch.Tensor,
        trans_b: bool = False,
    ) -> torch.Tensor:
        if self._gmm is not None:
            # grouped-gemm only accepts BF16
            return self._gmm(x.to(torch.bfloat16), w.to(torch.bfloat16), batch_sizes, trans_b=trans_b)  # type: ignore
        else:
            out = []
            start = 0
            for i, size in enumerate(batch_sizes.cpu().numpy()):
                rhs = w[i, :, :].t() if trans_b else w[i, :, :]
                out.append(x[start : start + size, :] @ rhs)
                start += size
            return torch.cat(out)

    def forward(
        self, x: torch.Tensor, batch_size_per_expert: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the expert outputs.

        :param x: The input of shape ``(*, d_model)``.
        :param batch_size_per_expert: Specifies how many items/tokens go to each expert. Should be a
            1-D ``LongTensor``.
        """
        # Scale gradients and get local tensors (in case of expert parallelism).
        # shape (all): (num_local_experts, hidden_size, d_model)
        w1, w2, w3 = (
            get_local_tensor(
                self.w1.view(self.num_experts, self.hidden_size, self.d_model)
            ),
            get_local_tensor(
                self.w2.view(self.num_experts, self.hidden_size, self.d_model)
            ),
            get_local_tensor(
                self.w3.view(self.num_experts, self.hidden_size, self.d_model)
            ),
        )

        # Compute the MLP.
        x1 = self.gmm(x, w1, batch_size_per_expert.cpu(), trans_b=True)
        x2 = self.gmm(x, w3, batch_size_per_expert.cpu(), trans_b=True)
        x1 = F.silu(x1) * x2
        return self.gmm(x1, w2, batch_size_per_expert.cpu())


class DroplessMoE(MoEBase):
    """
    A dropless MoE implementation.
    """

    def _init_parallel_mlp(  # type: ignore[override]
        self,
        *,
        d_model: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> ParallelDroplessMLP:
        return ParallelDroplessMLP(
            mlp=DroplessMoEMLP(
                d_model=d_model,
                num_experts=num_experts,
                hidden_size=hidden_size,
                dtype=dtype,
                init_device=init_device,
            ),
            top_k=self.router.top_k,
            cache=cache,
        )
