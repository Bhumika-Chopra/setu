# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SetuConnector: KV connector bridging vLLM's KV cache with Setu's
distributed tensor shard system for disaggregated prefill/decode.

On the prefill side, the KV cache buffer is registered as a Setu tensor
shard.  Setu's select().where() + copy() then handles non-contiguous
page access and direct NCCL transfer to the decode buffer.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Metadata dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SetuReqMeta:
    """Per-request metadata communicated from scheduler to worker."""

    request_id: str
    is_load: bool
    block_ids: list[int]
    remote_engine_id: str | None = None
    remote_block_ids: list[int] | None = None


@dataclass
class SetuConnectorMetadata(KVConnectorMetadata):
    """Scheduler-to-worker metadata for one scheduler step."""

    requests: list[SetuReqMeta] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Worker implementation
# ---------------------------------------------------------------------------


class SetuConnectorWorker:
    """Worker-side logic: shard registration, save (no-op), load (copy)."""

    def __init__(self, engine_id: str, setu_endpoint: str,
                 block_size: int) -> None:
        self._engine_id = engine_id
        self._setu_endpoint = setu_endpoint
        self._block_size = block_size

        # Initialised in connect()
        self._client: Any = None
        self._shard_refs: dict[str, Any] = {}
        self._is_mla: bool = False

        # Per-step state set by bind / cleared by clear
        self._metadata: SetuConnectorMetadata | None = None

        # Pending copy operations: {request_id: {layer_name: copy_op_id}}
        self._pending_loads: dict[str, dict[str, int]] = {}

        # Requests whose saves have been marked complete
        self._finished_sending: set[str] = set()

        # Requests whose loads have been fully waited on
        self._finished_recving: set[str] = set()

        # Requests that had load errors (block ids)
        self._block_ids_with_load_errors: set[int] = set()

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Establish connection to the Setu NodeAgent early, before KV
        cache allocation."""
        from setu.client import Client

        self._client = Client(self._setu_endpoint)
        assert self._client.is_connected, (
            f"Failed to connect to Setu at {self._setu_endpoint}"
        )
        logger.info("Connected to Setu at %s", self._setu_endpoint)

    def allocate_kv_caches(
        self,
        kv_cache_config: "KVCacheConfig",
        attn_backends: dict[str, type[AttentionBackend]],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Have Setu allocate KV cache tensors with their final
        reshaped dimensions, then return them as PyTorch tensors via IPC.

        This replicates the shape computation from
        ``_allocate_kv_cache`` + ``_reshape_kv_cache`` in attn_utils.py
        but delegates the actual memory allocation to Setu.
        """
        from setu._commons.datatypes import Device, TensorShardSpec
        from vllm.v1.kv_cache_interface import AttentionSpec

        assert self._client is not None, (
            "connect() must be called before allocate_kv_caches()"
        )

        setu_device = Device(torch_device=device)
        kv_caches: dict[str, torch.Tensor] = {}

        # Allocate one Setu shard per KVCacheTensor (may be shared across
        # layers) and keep a mapping from layer_name → (shard_ref, tensor).
        allocated: dict[int, tuple[Any, torch.Tensor]] = {}  # id(kv_cache_tensor) → (ref, tensor)

        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            assert isinstance(kv_cache_spec, AttentionSpec)

            for layer_name in kv_cache_group_spec.layer_names:
                # Find the raw tensor size for this layer from kv_cache_tensors
                kv_cache_tensor_obj = None
                for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
                    if layer_name in kv_cache_tensor.shared_by:
                        kv_cache_tensor_obj = kv_cache_tensor
                        break
                assert kv_cache_tensor_obj is not None, (
                    f"No KV cache tensor found for layer {layer_name}"
                )

                raw_size = kv_cache_tensor_obj.size
                assert raw_size % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_size // kv_cache_spec.page_size_bytes

                attn_backend = attn_backends[layer_name]
                kv_cache_shape = attn_backend.get_kv_cache_shape(
                    num_blocks,
                    kv_cache_spec.block_size,
                    kv_cache_spec.num_kv_heads,
                    kv_cache_spec.head_size,
                )

                # Apply stride ordering (same logic as _reshape_kv_cache)
                try:
                    kv_cache_stride_order = (
                        attn_backend.get_kv_cache_stride_order())
                    assert len(kv_cache_stride_order) == len(kv_cache_shape)
                except (AttributeError, NotImplementedError):
                    kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

                alloc_shape = tuple(
                    kv_cache_shape[i] for i in kv_cache_stride_order)
                inv_order = [
                    kv_cache_stride_order.index(i)
                    for i in range(len(kv_cache_stride_order))
                ]

                dtype = kv_cache_spec.dtype
                tensor_obj_id = id(kv_cache_tensor_obj)

                if tensor_obj_id not in allocated:
                    # First layer using this KVCacheTensor — allocate via Setu
                    shard_name = f"{self._engine_id}/kv_buffer/{layer_name}"
                    dims = self._build_dim_specs_from_shape(alloc_shape)

                    spec = TensorShardSpec(
                        name=shard_name,
                        dims=dims,
                        dtype=dtype,
                        device=setu_device,
                    )
                    shard_ref = self._client.register_tensor_shard(spec)
                    assert shard_ref is not None, (
                        f"Failed to register shard for layer {layer_name}"
                    )

                    # Wait for Setu to finish allocating GPU memory
                    self._client._client.wait_for_shard_allocation(
                        shard_ref.shard_id)

                    # Get the tensor back from Setu via IPC
                    shard = self._client._get_tensor_shard(shard_ref)
                    raw_tensor = shard.tensor
                    assert raw_tensor.shape == alloc_shape, (
                        f"Setu tensor shape {raw_tensor.shape} != "
                        f"expected {alloc_shape}"
                    )

                    allocated[tensor_obj_id] = (shard_ref, raw_tensor)
                    self._shard_refs[layer_name] = shard_ref

                    logger.info(
                        "Setu-allocated KV shard %s  shape=%s  dtype=%s",
                        shard_name, list(kv_cache_shape), dtype,
                    )
                else:
                    # Shared tensor — reuse the same allocation
                    shard_ref, raw_tensor = allocated[tensor_obj_id]
                    self._shard_refs[layer_name] = shard_ref
                    logger.info(
                        "Reusing Setu shard for layer %s (shared tensor)",
                        layer_name,
                    )

                # Permute to the logical shape expected by vLLM
                kv_caches[layer_name] = raw_tensor.permute(*inv_order)
                self._is_mla = len(kv_cache_shape) == 3

        return kv_caches

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register every vLLM KV cache buffer as a Setu tensor shard.

        If shards were already registered via ``allocate_kv_caches()``,
        this is a no-op.
        """
        if self._shard_refs:
            logger.info(
                "Setu shards already registered via allocate_kv_caches(), "
                "skipping register_kv_caches()"
            )
            return

        from setu._commons.datatypes import Device, TensorShardSpec

        assert self._client is not None, (
            "connect() must be called before register_kv_caches()"
        )

        for layer_name, kv_tensor in kv_caches.items():
            shard_name = f"{self._engine_id}/kv_buffer/{layer_name}"
            device = Device(torch_device=kv_tensor.device)
            dims, is_mla = self._build_dim_specs(kv_tensor)
            self._is_mla = is_mla

            spec = TensorShardSpec(
                name=shard_name,
                dims=dims,
                dtype=kv_tensor.dtype,
                device=device,
            )
            shard_ref = self._client.register_tensor_shard(spec)
            assert shard_ref is not None, (
                f"Failed to register shard for layer {layer_name}"
            )
            self._shard_refs[layer_name] = shard_ref
            logger.info(
                "Registered Setu shard %s  shape=%s  dtype=%s",
                shard_name, list(kv_tensor.shape), kv_tensor.dtype,
            )

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.disconnect()
            self._client = None

    # -- metadata binding ----------------------------------------------------

    def bind_connector_metadata(
            self, metadata: SetuConnectorMetadata) -> None:
        self._metadata = metadata

    def clear_connector_metadata(self) -> None:
        self._metadata = None

    # -- save (prefill side) -------------------------------------------------

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """No-op: the data already lives in the registered paged buffer."""
        pass

    def wait_for_save(self) -> None:
        """No-op: data is in-place.  Mark producer requests as sent."""
        assert self._metadata is not None
        for req_meta in self._metadata.requests:
            if not req_meta.is_load:
                self._finished_sending.add(req_meta.request_id)

    # -- load (decode side) --------------------------------------------------

    def start_load_kv(
            self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Issue Setu copy ops to pull remote KV into local paged buffer."""
        assert self._metadata is not None
        assert self._client is not None

        for req_meta in self._metadata.requests:
            if not req_meta.is_load:
                continue

            assert req_meta.remote_engine_id is not None
            assert req_meta.remote_block_ids is not None

            local_slots = self._block_ids_to_slots(req_meta.block_ids)
            remote_slots = self._block_ids_to_slots(req_meta.remote_block_ids)

            copy_ops: dict[str, int] = {}

            for layer_name in self._shard_refs:
                remote_shard_name = (
                    f"{req_meta.remote_engine_id}/kv_buffer/{layer_name}"
                )
                local_shard_name = (
                    f"{self._engine_id}/kv_buffer/{layer_name}"
                )

                src = self._client.select(remote_shard_name).where(
                    "slot", remote_slots)
                dst = self._client.select(local_shard_name).where(
                    "slot", local_slots)

                try:
                    copy_op_id = self._client.pull(src, dst)
                except RuntimeError:
                    logger.error(
                        "Setu pull failed for request %s layer %s",
                        req_meta.request_id, layer_name,
                    )
                    self._block_ids_with_load_errors.update(
                        req_meta.block_ids)
                    break

                assert copy_op_id is not None
                copy_ops[layer_name] = copy_op_id

            if copy_ops:
                self._pending_loads[req_meta.request_id] = copy_ops

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until the copy for *layer_name* is done for all requests."""
        assert self._client is not None
        for req_id, ops in list(self._pending_loads.items()):
            copy_op_id = ops.get(layer_name)
            if copy_op_id is not None:
                self._client.wait(copy_op_id)
                del ops[layer_name]

    # -- completion tracking -------------------------------------------------

    def get_finished(
        self, finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        # Drain finished sending
        sending = self._finished_sending.copy()
        self._finished_sending.clear()

        # Check which loads are fully done (all layers waited)
        for req_id in list(self._pending_loads):
            if not self._pending_loads[req_id]:
                # All layer ops waited → done
                self._finished_recving.add(req_id)
                del self._pending_loads[req_id]

        recving = self._finished_recving.copy()
        self._finished_recving.clear()

        return (sending or None, recving or None)

    def get_block_ids_with_load_errors(self) -> set[int]:
        errors = self._block_ids_with_load_errors.copy()
        self._block_ids_with_load_errors.clear()
        return errors

    # -- helpers -------------------------------------------------------------

    def _block_ids_to_slots(self, block_ids: list[int]) -> list[int]:
        """Expand block IDs to flat slot indices."""
        slots: list[int] = []
        for bid in block_ids:
            base = bid * self._block_size
            slots.extend(range(base, base + self._block_size))
        return slots

    @staticmethod
    def _build_dim_specs_from_shape(
        shape: tuple[int, ...],
    ) -> list[Any]:
        """Build TensorDimSpec list from an arbitrary shape tuple.

        Each dimension is named ``dim_0``, ``dim_1``, etc.  This is used
        when Setu allocates the tensor (``allocate_kv_caches``), where we
        know the exact shape but not the semantic meaning of each dim.
        """
        from setu._commons.datatypes import TensorDimSpec

        return [
            TensorDimSpec(f"dim_{i}", size, 0, size)
            for i, size in enumerate(shape)
        ]

    @staticmethod
    def _build_dim_specs(
        kv_tensor: torch.Tensor,
    ) -> tuple[list[Any], bool]:
        """Build TensorDimSpec list from a paged KV cache tensor.

        Non-MLA shape: [2, num_pages, page_size, head_dim]
        MLA shape:     [num_pages, page_size, kv_dim]
        """
        from setu._commons.datatypes import TensorDimSpec

        shape = kv_tensor.shape
        is_mla = len(shape) == 3

        if is_mla:
            num_pages, page_size, kv_dim = shape
            total_slots = num_pages * page_size
            dims = [
                TensorDimSpec("slot", total_slots, 0, total_slots),
                TensorDimSpec("kv_dim", kv_dim, 0, kv_dim),
            ]
        else:
            kv_heads, num_pages, page_size, head_dim = shape[0], shape[1], shape[2], shape[3]
            total_slots = num_pages * page_size
            dims = [
                TensorDimSpec("kv", kv_heads, 0, kv_heads),
                TensorDimSpec("slot", total_slots, 0, total_slots),
                TensorDimSpec("hidden", head_dim, 0, head_dim),
            ]

        return dims, is_mla


# ---------------------------------------------------------------------------
# Scheduler implementation
# ---------------------------------------------------------------------------


class SetuConnectorScheduler:
    """Scheduler-side logic: token matching, block tracking, metadata build."""

    def __init__(self, engine_id: str, setu_endpoint: str,
                 block_size: int) -> None:
        self._engine_id = engine_id
        self._setu_endpoint = setu_endpoint
        self._block_size = block_size

        # Pending requests to include in next build_connector_meta
        # {req_id: SetuReqMeta}
        self._pending_saves: dict[str, SetuReqMeta] = {}
        self._pending_loads: dict[str, SetuReqMeta] = {}

    # -- scheduler callbacks -------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        kv_params = request.kv_transfer_params
        if kv_params is not None and "setu_remote_engine_id" in kv_params:
            # All prompt tokens (except the last) can be loaded from remote.
            # Align to block boundary.
            num_prompt_tokens = len(request.prompt_token_ids or [])
            loadable = self._align_to_block(
                num_prompt_tokens - 1) - num_computed_tokens
            return max(loadable, 0), False
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        block_ids = list(blocks.get_block_ids()[0])

        kv_params = request.kv_transfer_params
        if num_external_tokens > 0 and kv_params is not None:
            # Decode side: will load from remote
            self._pending_loads[req_id] = SetuReqMeta(
                request_id=req_id,
                is_load=True,
                block_ids=block_ids,
                remote_engine_id=kv_params["setu_remote_engine_id"],
                remote_block_ids=kv_params["setu_remote_block_ids"],
            )
        elif num_external_tokens == 0 and kv_params is None:
            # Prefill side: newly computed, will be saved (in-place)
            self._pending_saves[req_id] = SetuReqMeta(
                request_id=req_id,
                is_load=False,
                block_ids=block_ids,
            )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput,
    ) -> SetuConnectorMetadata:
        meta = SetuConnectorMetadata()

        for req_meta in self._pending_loads.values():
            meta.requests.append(req_meta)
        for req_meta in self._pending_saves.values():
            meta.requests.append(req_meta)

        # Reset per-step state
        self._pending_loads.clear()
        self._pending_saves.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        kv_params = request.kv_transfer_params
        if kv_params is not None and "setu_remote_engine_id" in kv_params:
            # Decode (consumer) side: blocks freed normally
            return False, None

        # Prefill (producer) side: hold blocks so decode can copy,
        # and return transfer params for the decode engine.
        transfer_params = {
            "setu_remote_engine_id": self._engine_id,
            "setu_remote_block_ids": block_ids,
            "setu_endpoint": self._setu_endpoint,
        }
        return True, transfer_params

    # -- helpers -------------------------------------------------------------

    def _align_to_block(self, num_tokens: int) -> int:
        """Align down to block boundary."""
        return (num_tokens // self._block_size) * self._block_size


# ---------------------------------------------------------------------------
# Main connector class
# ---------------------------------------------------------------------------


class SetuConnector(KVConnectorBase_V1):
    """KV connector that bridges vLLM with Setu's tensor shard system."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        engine_id = self._kv_transfer_config.engine_id
        setu_endpoint = self._kv_transfer_config.get_from_extra_config(
            "setu_endpoint", "tcp://localhost:5555",
        )
        block_size = vllm_config.cache_config.block_size

        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = SetuConnectorScheduler(
                engine_id, setu_endpoint, block_size)
            self._worker: SetuConnectorWorker | None = None
        else:
            self._scheduler: SetuConnectorScheduler | None = None  # type: ignore[no-redef]
            self._worker = SetuConnectorWorker(
                engine_id, setu_endpoint, block_size)

    # ======================================================================
    # Worker-side methods
    # ======================================================================

    def connect(self) -> None:
        if self._worker is not None:
            self._worker.connect()

    def allocate_kv_caches(
        self,
        kv_cache_config: "KVCacheConfig",
        attn_backends: dict[str, type[AttentionBackend]],
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
        if self._worker is None:
            return None
        return self._worker.allocate_kv_caches(
            kv_cache_config, attn_backends, device)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self._worker is not None
        self._worker.register_kv_caches(kv_caches)

    def bind_connector_metadata(
            self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self._worker is not None:
            assert isinstance(connector_metadata, SetuConnectorMetadata)
            self._worker.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self._worker is not None:
            self._worker.clear_connector_metadata()

    def start_load_kv(
            self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        assert self._worker is not None
        self._worker.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert self._worker is not None
        self._worker.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        assert self._worker is not None
        self._worker.save_kv_layer(layer_name, kv_layer, attn_metadata,
                                   **kwargs)

    def wait_for_save(self) -> None:
        assert self._worker is not None
        self._worker.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        assert self._worker is not None
        return self._worker.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self._worker is not None
        return self._worker.get_block_ids_with_load_errors()

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.shutdown()

    # ======================================================================
    # Scheduler-side methods
    # ======================================================================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self._scheduler is not None
        return self._scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        assert self._scheduler is not None
        self._scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self._scheduler is not None
        return self._scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request, block_ids)
