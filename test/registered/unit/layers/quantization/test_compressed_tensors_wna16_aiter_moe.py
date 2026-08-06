"""CPU-only contracts for compressed-tensors AITER W4A16 MoE.

The FlyDSL kernel consumes a physical layout which is different from the
checkpoint and Triton layouts.  These tests deliberately replace AITER's
shuffle helpers with small CPU implementations so that nibble order, signed
conversion, scale layout, and backend selection can be validated without a
ROCm device.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationType

import sglang.srt.layers.quantization.compressed_tensors.compressed_tensors as ct_config
import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe as wna16
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16AiterMoE,
    CompressedTensorsWNA16TritonMoE,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pack_gptq_word(signed_values):
    """Pack signed values using compressed-tensors' uint4b8 representation."""
    assert len(signed_values) == 8
    word = 0
    for index, value in enumerate(signed_values):
        assert -8 <= value <= 7
        word |= (value + 8) << (4 * index)
    # torch.tensor(..., dtype=int32) rejects positive Python ints above INT_MAX.
    return word if word < 2**31 else word - 2**32


@pytest.fixture
def fake_aiter(monkeypatch):
    calls = {"shuffle_weight": [], "pack": [], "shuffle_scale": []}

    fake_root = ModuleType("aiter")
    fake_root.__path__ = []
    # int8 is an 8-bit storage stand-in for the int4 shell dtype.  The physical
    # tensor still contains two nibbles per element, exactly as the real helper.
    fake_root.dtypes = SimpleNamespace(i4x2=torch.int8)

    fake_ops = ModuleType("aiter.ops")
    fake_ops.__path__ = []
    fake_shuffle = ModuleType("aiter.ops.shuffle")

    def shuffle_weight(value, layout):
        calls["shuffle_weight"].append((value.clone(), layout))
        # Identity makes the byte-level expected values independently legible.
        return value.contiguous()

    def pack_int8_to_packed_int4(value):
        calls["pack"].append(value.clone())
        values = value.contiguous().view(-1, 8).to(torch.int16)
        nibbles = (values & 0xF).to(torch.uint8)
        # AITER's special ordering is v0/v4, v1/v5, v2/v6, v3/v7.
        packed = nibbles[:, :4] | (nibbles[:, 4:] << 4)
        return packed.reshape(-1).view(torch.int8)

    def shuffle_scale_for_int4(value, group_size):
        calls["shuffle_scale"].append((value.clone(), group_size))
        experts, groups, out_features = value.shape
        return (
            value.view(experts, groups // 2, 2, out_features)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

    fake_shuffle.shuffle_weight = shuffle_weight
    fake_shuffle.pack_int8_to_packed_int4 = pack_int8_to_packed_int4
    fake_shuffle.shuffle_scale_for_int4 = shuffle_scale_for_int4

    fake_flydsl = ModuleType("aiter.ops.flydsl")
    fake_flydsl.__path__ = []
    fake_flydsl_utils = ModuleType("aiter.ops.flydsl.utils")
    fake_flydsl_utils.is_flydsl_available = lambda: True

    monkeypatch.setitem(sys.modules, "aiter", fake_root)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.shuffle", fake_shuffle)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", fake_flydsl)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.utils", fake_flydsl_utils)
    monkeypatch.setattr(wna16, "_use_aiter", True)
    return calls


def _weight_quant(**overrides):
    values = {
        "num_bits": 4,
        "type": QuantizationType.INT,
        "strategy": "group",
        "group_size": 32,
        "actorder": None,
        "symmetric": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _quant_config():
    return SimpleNamespace(quant_format=CompressionFormat.pack_quantized.value)


def _new_scheme(**weight_overrides):
    return CompressedTensorsWNA16AiterMoE(
        _quant_config(), _weight_quant(**weight_overrides)
    )


def _mock_exec_moe_config(monkeypatch, **overrides):
    config = {
        "deepep_dispatcher_output_dtype": "auto",
        "enable_eplb": False,
        "enable_elastic_expert_backup": False,
        "kt_weight_path": None,
    }
    config.update(overrides)
    monkeypatch.setattr(
        ct_config,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(**config)),
    )


def _moe_config(group_size=32, num_bits=4, symmetric=True, actorder=None):
    weight_config = {
        "num_bits": num_bits,
        "type": "int",
        "symmetric": symmetric,
        "strategy": "group",
        "group_size": group_size,
    }
    if actorder is not None:
        weight_config["actorder"] = actorder
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "experts": {
                "targets": ["re:.*mlp.experts.*"],
                "weights": weight_config,
                "input_activations": None,
            }
        },
    }


def test_gptq_uint4b8_to_aiter_signed_values_and_special_nibble_order(fake_aiter):
    first = [-8, -7, -1, 0, 1, 6, 7, -6]
    second = [7, 6, 5, 4, 3, 2, 1, 0]
    # Checkpoint layout is [E, K/8, N].
    checkpoint = torch.tensor(
        [[[_pack_gptq_word(first), _pack_gptq_word(second)]]],
        dtype=torch.int32,
    )

    converted = CompressedTensorsWNA16AiterMoE._gptq_int32_chunk_to_aiter_i4x2(
        checkpoint
    )

    unpacked, layout = fake_aiter["shuffle_weight"][0]
    assert layout == (16, 16)
    assert unpacked.shape == (1, 2, 8)
    torch.testing.assert_close(
        unpacked,
        torch.tensor([[first, second]], dtype=torch.int8),
        rtol=0,
        atol=0,
    )
    # first -> [v0|v4<<4, v1|v5<<4, v2|v6<<4, v3|v7<<4]
    expected_bytes = torch.tensor(
        [[[0x18, 0x69, 0x7F, 0xA0], [0x37, 0x26, 0x15, 0x04]]],
        dtype=torch.uint8,
    )
    assert converted.shape == (1, 2, 4)
    torch.testing.assert_close(converted.view(torch.uint8), expected_bytes)


def test_full_repack_chunks_experts_without_changing_output(fake_aiter, monkeypatch):
    logical_rows = [
        [[-8, -4, -1, 0, 1, 3, 6, 7], [7, 5, 3, 1, -1, -3, -5, -7]],
        [[0, 1, 2, 3, 4, 5, 6, 7], [-8, -7, -6, -5, -4, -3, -2, -1]],
        [[-2, 4, -6, 7, 1, -3, 5, -8], [6, -1, 2, -7, 3, -5, 0, 4]],
    ]
    checkpoint = torch.tensor(
        [
            [[_pack_gptq_word(expert[0]), _pack_gptq_word(expert[1])]]
            for expert in logical_rows
        ],
        dtype=torch.int32,
    )

    # Each expert has 2 * 8 logical elements, so this forces one expert/chunk.
    monkeypatch.setattr(
        CompressedTensorsWNA16AiterMoE,
        "_REPACK_CHUNK_LOGICAL_ELEMENTS",
        16,
    )
    converted = CompressedTensorsWNA16AiterMoE._gptq_int32_to_aiter_i4x2(checkpoint)

    assert [call[0].shape[0] for call in fake_aiter["shuffle_weight"]] == [1, 1, 1]
    assert converted.shape == (3, 2, 4)
    for expert, rows in enumerate(logical_rows):
        for out_feature, values in enumerate(rows):
            n = [value & 0xF for value in values]
            expected = torch.tensor(
                [
                    n[0] | n[4] << 4,
                    n[1] | n[5] << 4,
                    n[2] | n[6] << 4,
                    n[3] | n[7] << 4,
                ],
                dtype=torch.uint8,
            )
            torch.testing.assert_close(
                converted[expert, out_feature].view(torch.uint8), expected
            )


@pytest.mark.parametrize(
    "bad_weight",
    [
        torch.zeros((1, 1, 1), dtype=torch.int16),
        torch.zeros((1, 1), dtype=torch.int32),
    ],
)
def test_full_repack_rejects_wrong_dtype_or_rank(fake_aiter, bad_weight):
    with pytest.raises(TypeError, match="three-dimensional torch.int32"):
        CompressedTensorsWNA16AiterMoE._gptq_int32_to_aiter_i4x2(bad_weight)


def test_prepare_scale_uses_group32_bf16_paired_group_layout(fake_aiter):
    scale = torch.arange(12, dtype=torch.bfloat16).view(1, 4, 3)

    converted = CompressedTensorsWNA16AiterMoE._prepare_aiter_scale(scale)

    assert fake_aiter["shuffle_scale"][0][1] == 32
    expected = torch.tensor(
        [0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(converted, expected)
    assert converted.ndim == 1
    assert converted.is_contiguous()


@pytest.mark.parametrize(
    "bad_scale",
    [
        torch.zeros((1, 2, 3), dtype=torch.float32),
        torch.zeros((2, 3), dtype=torch.bfloat16),
    ],
)
def test_prepare_scale_rejects_non_bf16_or_non_3d(fake_aiter, bad_scale):
    with pytest.raises(TypeError, match="BF16 scale tensor"):
        CompressedTensorsWNA16AiterMoE._prepare_aiter_scale(bad_scale)


def test_process_weights_marks_parameters_shuffled_and_is_idempotent(fake_aiter):
    layer = torch.nn.Module()
    # K=64, N=16: enough groups to exercise the paired BF16 scale layout.
    layer.w13_weight_packed = torch.nn.Parameter(
        torch.zeros((1, 8, 16), dtype=torch.int32), requires_grad=False
    )
    layer.w2_weight_packed = torch.nn.Parameter(
        torch.zeros((1, 8, 16), dtype=torch.int32), requires_grad=False
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones((1, 2, 16), dtype=torch.bfloat16), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.ones((1, 2, 16), dtype=torch.bfloat16), requires_grad=False
    )
    dispatcher_configs = []
    layer.dispatcher = SimpleNamespace(
        set_quant_config=lambda config: dispatcher_configs.append(config)
    )
    scheme = object.__new__(CompressedTensorsWNA16AiterMoE)

    scheme.process_weights_after_loading(layer)
    calls_after_first_conversion = len(fake_aiter["shuffle_weight"])
    scheme.process_weights_after_loading(layer)

    assert getattr(layer.w13_weight_packed, "is_shuffled", False) is True
    assert getattr(layer.w2_weight_packed, "is_shuffled", False) is True
    assert layer.w13_weight_packed.shape == (1, 16, 32)
    assert layer.w2_weight_packed.shape == (1, 16, 32)
    assert layer.w13_weight_scale.shape == (32,)
    assert layer.w2_weight_scale.shape == (32,)
    assert layer.is_aiter_w4a16_converted is True
    assert "restart the model worker" in (
        layer._online_weight_update_unsupported_reason
    )
    assert dispatcher_configs == [{"dispatcher_output_dtype": "bf16"}]
    assert len(fake_aiter["shuffle_weight"]) == calls_after_first_conversion


@pytest.mark.parametrize(
    "update_call",
    [
        lambda updater: updater.update_weights_from_disk("unused", "auto"),
        lambda updater: updater.update_weights_from_distributed(
            [], [], [], "uninitialized"
        ),
        lambda updater: updater.update_weights_from_tensor([]),
        lambda updater: updater.update_weights_from_ipc(
            SimpleNamespace(zmq_handles={})
        ),
    ],
    ids=["disk", "distributed", "tensor", "ipc"],
)
def test_all_online_weight_update_transports_reject_before_writing(update_call):
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        WeightUpdater,
    )

    model = torch.nn.Module()
    layer = torch.nn.Module()
    experts = torch.nn.Module()
    experts._online_weight_update_unsupported_reason = "runtime FlyDSL layout"
    layer.add_module("experts", experts)
    model.add_module("layer0", layer)

    def unexpected_write(*args, **kwargs):
        pytest.fail("online-update preflight allowed a write path to run")

    updater = WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=SimpleNamespace(),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=unexpected_write,
        recapture_cuda_graph=unexpected_write,
        get_model_runner=unexpected_write,
    )

    success, message = update_call(updater)

    assert success is False
    assert "Online weight updates are not supported" in message
    assert "layer0.experts" in message
    assert "runtime FlyDSL layout" in message


def test_online_weight_update_preflight_ignores_canonical_models_and_names_root():
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        _unsupported_online_weight_update_error,
    )

    model = torch.nn.Module()
    assert _unsupported_online_weight_update_error(model) is None

    model._online_weight_update_unsupported_reason = "runtime-only packing"
    message = _unsupported_online_weight_update_error(model)
    assert message is not None
    assert "'<root>'" in message
    assert "runtime-only packing" in message


def test_restore_rejects_reload_after_aiter_layout_conversion():
    layer = torch.nn.Module()
    scheme = object.__new__(CompressedTensorsWNA16AiterMoE)

    # Before post-load conversion the checkpoint buffers are still canonical,
    # so there is nothing to restore.
    scheme.restore_weights_before_loading(layer)

    layer.is_aiter_w4a16_converted = True
    with pytest.raises(NotImplementedError, match="Recreate the model worker"):
        scheme.restore_weights_before_loading(layer)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_bits": 8}, "num_bits=8"),
        ({"type": QuantizationType.FLOAT}, "type='float'"),
        ({"strategy": "channel"}, "strategy='channel'"),
        ({"group_size": 128}, "group_size=128"),
        ({"symmetric": False}, "symmetric=False"),
        ({"actorder": "group"}, "actorder='group'"),
    ],
)
def test_scheme_rejects_unsupported_quantization(fake_aiter, overrides, message):
    with pytest.raises(ValueError, match=message):
        _new_scheme(**overrides)


def test_scheme_requires_aiter_enablement(fake_aiter, monkeypatch):
    monkeypatch.setattr(wna16, "_use_aiter", False)
    with pytest.raises(ValueError, match="SGLANG_USE_AITER=1"):
        _new_scheme()


def test_scheme_rejects_unavailable_flydsl(fake_aiter, monkeypatch):
    flydsl_utils = sys.modules["aiter.ops.flydsl.utils"]
    monkeypatch.setattr(flydsl_utils, "is_flydsl_available", lambda: False)

    with pytest.raises(RuntimeError, match="is_flydsl_available.*false"):
        _new_scheme()


def test_scheme_reports_missing_aiter_shuffle_api(fake_aiter, monkeypatch):
    shuffle_module = sys.modules["aiter.ops.shuffle"]
    monkeypatch.delattr(shuffle_module, "pack_int8_to_packed_int4")

    with pytest.raises(RuntimeError, match="does not provide.*conversion APIs"):
        _new_scheme()


def test_create_weights_rejects_non_bf16(fake_aiter):
    scheme = _new_scheme()

    with pytest.raises(ValueError, match="requires BF16"):
        scheme.create_weights(
            torch.nn.Module(),
            num_experts=1,
            hidden_size=128,
            intermediate_size_per_partition=128,
            params_dtype=torch.float16,
        )


def test_create_weights_rejects_expert_bias(fake_aiter):
    scheme = _new_scheme()

    with pytest.raises(ValueError, match="does not support expert bias"):
        scheme.create_weights(
            torch.nn.Module(),
            num_experts=1,
            hidden_size=128,
            intermediate_size_per_partition=128,
            params_dtype=torch.bfloat16,
            with_bias=True,
        )


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size"),
    [(130, 128), (128, 130)],
)
def test_create_weights_rejects_non_128_aligned_dimensions(
    fake_aiter, hidden_size, intermediate_size
):
    scheme = _new_scheme()

    with pytest.raises(ValueError, match="multiples of 128"):
        scheme.create_weights(
            torch.nn.Module(),
            num_experts=1,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=torch.bfloat16,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"activation": "gelu"}, "supports only SiLU"),
        ({"is_gated": False}, "requires a gated MLP"),
        ({"gemm1_alpha": 1.0}, "without alpha or clamp"),
        ({"gemm1_clamp_limit": 7.0}, "without alpha or clamp"),
        ({"swiglu_limit": 7.0}, "without alpha or clamp"),
    ],
)
def test_create_moe_runner_rejects_unsupported_mlp_semantics(
    fake_aiter, overrides, message
):
    scheme = _new_scheme()
    config = MoeRunnerConfig(**overrides)

    with pytest.raises(ValueError, match=message):
        scheme.create_moe_runner(torch.nn.Module(), config)


def test_create_moe_runner_rejects_programmatic_ktransformers_wrapper(fake_aiter):
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod

    scheme = _new_scheme()
    layer = torch.nn.Module()
    layer.quant_method = object.__new__(KTEPWrapperMethod)

    with pytest.raises(ValueError, match="KTransformers expert wrapper"):
        scheme.create_moe_runner(layer, MoeRunnerConfig())


def test_get_aiter_quant_info_marks_runtime_views_and_forwards_route_metadata(
    fake_aiter,
):
    from sglang.srt.layers.moe.moe_runner.aiter import AiterQuantType

    scheme = _new_scheme()
    scheme.moe_runner_config = SimpleNamespace(apply_router_weight_on_input=True)
    layer = torch.nn.Module()
    layer.w13_weight_packed = torch.nn.Parameter(
        torch.arange(8, dtype=torch.int8).view(1, 2, 4), requires_grad=False
    )
    layer.w2_weight_packed = torch.nn.Parameter(
        torch.arange(8, 16, dtype=torch.int8).view(1, 2, 4),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones(4, dtype=torch.bfloat16), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.full((4,), 2, dtype=torch.bfloat16), requires_grad=False
    )
    expert_mask = torch.tensor([True, False, True])
    layer.dispatcher = SimpleNamespace(expert_mask_gpu=expert_mask)

    quant_info = scheme.get_aiter_quant_info(layer)

    assert quant_info.w13_weight.dtype == torch.int8  # fake i4x2 marker
    assert quant_info.w2_weight.dtype == torch.int8
    assert getattr(quant_info.w13_weight, "is_shuffled", False) is True
    assert getattr(quant_info.w2_weight, "is_shuffled", False) is True
    assert quant_info.quant_type is AiterQuantType.PER_1X32
    assert quant_info.w13_scale is layer.w13_weight_scale
    assert quant_info.w2_scale is layer.w2_weight_scale
    assert quant_info.expert_mask is expert_mask
    assert quant_info.doweight_stage1 is True


@pytest.mark.parametrize("use_fused_loader", [False, True])
def test_fused_moe_loader_transposes_aiter_checkpoint_before_repack(
    use_fused_loader, monkeypatch
):
    """Exercise both real FusedMoE loader allowlists used by packed weights."""
    fake_sgl_kernel = ModuleType("sgl_kernel")
    fake_sgl_kernel.gelu_and_mul = lambda *args, **kwargs: None
    fake_sgl_kernel.moe_sum_reduce = lambda *args, **kwargs: None
    fake_sgl_kernel.silu_and_mul = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "sgl_kernel", fake_sgl_kernel)

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    layer = object.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.moe_tp_rank = 0
    layer.quant_method = object.__new__(CompressedTensorsWNA16AiterMoE)
    layer.quant_config = None
    layer.use_flashinfer_trtllm_moe = False
    layer.use_triton_kernels = False
    layer._maybe_load_fp8_shared_expert_as_fp4 = lambda **kwargs: False

    captured = {}

    def capture_load(**kwargs):
        captured.update(kwargs)

    layer._load_model_weight_or_group_weight_scale = capture_load

    loaded = torch.arange(6, dtype=torch.int32).view(2, 3)
    param = torch.nn.Parameter(
        torch.empty((1, 3, 2), dtype=torch.int32), requires_grad=False
    )
    param.is_transposed = True

    if use_fused_loader:
        layer.weight_loader_fused(
            param,
            loaded,
            weight_name="w13_weight_packed",
            shard_id="w13",
        )
    else:
        layer._weight_loader_impl(
            param,
            loaded,
            weight_name="w13_weight_packed",
            shard_id="w1",
            expert_id=0,
        )

    torch.testing.assert_close(
        captured["loaded_weight"], loaded.t().contiguous(), rtol=0, atol=0
    )


def test_rocm_explicit_aiter_selects_aiter_but_auto_stays_triton(
    fake_aiter, monkeypatch
):
    quant_config = CompressedTensorsConfig.from_config(_moe_config())
    layer = torch.nn.Module()
    layer_name = "model.layers.0.mlp.experts"
    monkeypatch.setattr(ct_config, "_is_hip", True)
    monkeypatch.setattr(ct_config, "get_moe_a2a_backend", lambda: MoeA2ABackend.NONE)
    _mock_exec_moe_config(monkeypatch)

    monkeypatch.setattr(
        ct_config, "get_moe_runner_backend", lambda: MoeRunnerBackend.AITER
    )
    assert isinstance(
        quant_config.get_moe_scheme(layer, layer_name),
        CompressedTensorsWNA16AiterMoE,
    )

    monkeypatch.setattr(
        ct_config, "get_moe_runner_backend", lambda: MoeRunnerBackend.AUTO
    )
    assert isinstance(
        quant_config.get_moe_scheme(layer, layer_name),
        CompressedTensorsWNA16TritonMoE,
    )


@pytest.mark.parametrize(
    ("resolved_backend", "uses_global_expert_ids"),
    [
        (MoeRunnerBackend.TRITON, False),
        (MoeRunnerBackend.AITER, True),
    ],
)
def test_resolved_runner_overrides_auto_dispatcher_assumption(
    monkeypatch, resolved_backend, uses_global_expert_ids
):
    """Make standard-EP expert-ID handling follow the concrete runner."""
    import sglang.srt.layers.moe.token_dispatcher.standard as standard_dispatcher

    config = MoeRunnerConfig(
        num_experts=4,
        num_local_experts=2,
        num_fused_shared_experts=0,
    )
    MoeRunner(resolved_backend, config)

    monkeypatch.setattr(
        standard_dispatcher,
        "get_parallel",
        lambda: SimpleNamespace(moe_ep_size=2, moe_ep_rank=0),
    )
    monkeypatch.setattr(
        standard_dispatcher,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.AUTO,
    )
    monkeypatch.setattr(
        standard_dispatcher,
        "get_moe_a2a_backend",
        lambda: MoeA2ABackend.NONE,
    )
    monkeypatch.setattr(standard_dispatcher, "_use_aiter", True)

    dispatcher = standard_dispatcher.StandardDispatcher(config)

    assert config.resolved_runner_backend is resolved_backend
    assert dispatcher.use_aiter_moe_runner is uses_global_expert_ids


@pytest.mark.parametrize(
    "a2a_backend",
    [
        MoeA2ABackend.MOONCAKE,
        MoeA2ABackend.NIXL,
        MoeA2ABackend.PPLX,
        MoeA2ABackend.FLASHINFER,
    ],
)
def test_explicit_aiter_rejects_incompatible_a2a_backend(
    fake_aiter, monkeypatch, a2a_backend
):
    quant_config = CompressedTensorsConfig.from_config(_moe_config())
    monkeypatch.setattr(ct_config, "_is_hip", True)
    monkeypatch.setattr(
        ct_config, "get_moe_runner_backend", lambda: MoeRunnerBackend.AITER
    )
    monkeypatch.setattr(ct_config, "get_moe_a2a_backend", lambda: a2a_backend)

    with pytest.raises(ValueError, match="supports only.*none, deepep, or mori"):
        quant_config.get_moe_scheme(torch.nn.Module(), "model.layers.0.mlp.experts")


@pytest.mark.parametrize("dispatch_dtype", ["fp8", "int8", "nvfp4"])
def test_aiter_w4a16_rejects_non_bf16_deepep_override(monkeypatch, dispatch_dtype):
    _mock_exec_moe_config(monkeypatch, deepep_dispatcher_output_dtype=dispatch_dtype)

    with pytest.raises(
        ValueError,
        match="requires BF16 DeepEP dispatch.*Use 'auto' or 'bf16'",
    ):
        ct_config._validate_aiter_w4a16_runtime(MoeA2ABackend.DEEPEP)


@pytest.mark.parametrize("dispatch_dtype", ["auto", "bf16"])
def test_aiter_w4a16_accepts_bf16_deepep_configuration(monkeypatch, dispatch_dtype):
    _mock_exec_moe_config(monkeypatch, deepep_dispatcher_output_dtype=dispatch_dtype)

    ct_config._validate_aiter_w4a16_runtime(MoeA2ABackend.DEEPEP)


@pytest.mark.parametrize("dispatch_dtype", ["fp8", "fp4"])
def test_aiter_w4a16_rejects_non_bf16_mori_override(monkeypatch, dispatch_dtype):
    _mock_exec_moe_config(monkeypatch)
    monkeypatch.setenv("SGLANG_MORI_DISPATCH_DTYPE", dispatch_dtype)

    with pytest.raises(
        ValueError,
        match="requires BF16 Mori dispatch.*SGLANG_MORI_DISPATCH_DTYPE",
    ):
        ct_config._validate_aiter_w4a16_runtime(MoeA2ABackend.MORI)


@pytest.mark.parametrize("legacy_env", ["SGLANG_MORI_FP8_DISP", "SGLANG_MORI_FP4_DISP"])
def test_aiter_w4a16_rejects_legacy_quantized_mori_override(monkeypatch, legacy_env):
    _mock_exec_moe_config(monkeypatch)
    monkeypatch.delenv("SGLANG_MORI_DISPATCH_DTYPE", raising=False)
    monkeypatch.setenv(legacy_env, "1")

    with pytest.raises(ValueError, match=f"deprecated {legacy_env}=1"):
        ct_config._validate_aiter_w4a16_runtime(MoeA2ABackend.MORI)


@pytest.mark.parametrize("dispatch_dtype", ["auto", "bf16"])
def test_aiter_w4a16_accepts_bf16_mori_configuration(monkeypatch, dispatch_dtype):
    _mock_exec_moe_config(monkeypatch)
    monkeypatch.setenv("SGLANG_MORI_DISPATCH_DTYPE", dispatch_dtype)
    # The new variable has precedence over deprecated flags, matching Mori.
    monkeypatch.setenv("SGLANG_MORI_FP8_DISP", "1")
    monkeypatch.setenv("SGLANG_MORI_FP4_DISP", "1")

    ct_config._validate_aiter_w4a16_runtime(MoeA2ABackend.MORI)


@pytest.mark.parametrize(
    ("config_field", "cli_flag"),
    [
        ("enable_eplb", "--enable-eplb"),
        ("enable_elastic_expert_backup", "--enable-elastic-expert-backup"),
        ("kt_weight_path", "--kt-weight-path"),
    ],
)
def test_aiter_w4a16_rejects_incompatible_expert_management_paths(
    monkeypatch, config_field, cli_flag
):
    _mock_exec_moe_config(monkeypatch, **{config_field: True})

    with pytest.raises(ValueError, match=cli_flag):
        ct_config._validate_aiter_w4a16_runtime(MoeA2ABackend.NONE)
