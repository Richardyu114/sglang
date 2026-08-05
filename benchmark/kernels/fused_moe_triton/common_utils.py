import itertools
import json
from types import SimpleNamespace
from typing import Dict, List, TypedDict

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import get_config_dtype_str
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_file_name,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.srt.utils.hf_transformers_utils import get_config


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def calculate_shard_intermediate_size(
    intermediate_size: int, tp_size: int, ep_size: int = 1
) -> int:
    assert tp_size % ep_size == 0
    moe_tp_size = tp_size // ep_size
    assert intermediate_size % moe_tp_size == 0
    return 2 * intermediate_size // moe_tp_size


def get_model_config(
    model_name: str,
    tp_size: int,
    ep_size: int = 1,
    disable_shared_experts_fusion: bool = False,
    topk_ids_dir: str = None,
) -> Dict:
    config = get_config(model_name, trust_remote_code=True)
    architecture = config.architectures[0]
    block_shape = None
    if (
        hasattr(config, "quantization_config")
        and "weight_block_size" in config.quantization_config
    ):
        block_shape = config.quantization_config["weight_block_size"]
        assert len(block_shape) == 2

    if (
        hasattr(config, "quantization_config")
        and "config_groups" in config.quantization_config
    ):
        config_groups = config.quantization_config["config_groups"]
        # Get group_size from the first group's weights config
        first_group = next(iter(config_groups.values()), {})
        weights_config = first_group.get("weights", {})
        group_size = weights_config.get("group_size")
        block_shape = [0, group_size]
        assert len(block_shape) == 2
    # Replace config with text_config for encoder-decoder models after getting block_shape and architecture
    if hasattr(config, "text_config"):
        text_config = config.get_text_config()
        # Some models (e.g. MiniMax-M3) carry text_config as a plain dict; wrap
        # it so downstream attribute access works uniformly.
        if isinstance(text_config, dict):
            config = SimpleNamespace(**text_config)
        else:
            config = text_config

    hidden_size = config.hidden_size
    if architecture == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts // ep_size
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
    elif architecture == "JambaForCausalLM":
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
    elif architecture in [
        "Qwen2MoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "MellumForCausalLM",
    ]:
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture in [
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV4ForCausalLM",
        "Glm4MoeForCausalLM",
        "GlmMoeDsaForCausalLM",
        "KimiK25ForConditionalGeneration",
        "KimiVLForConditionalGeneration",
        "MistralLarge3ForCausalLM",
    ]:
        E = (config.n_routed_experts // ep_size) + (
            0
            if disable_shared_experts_fusion
            or architecture
            not in [
                "DeepseekV3ForCausalLM",
                "DeepseekV32ForCausalLM",
                "Glm4MoeForCausalLM",
                "GlmMoeDsaForCausalLM",
                "MistralLarge3ForCausalLM",
            ]
            else 1
        )
        topk = config.num_experts_per_tok + (
            0 if disable_shared_experts_fusion or topk_ids_dir is None else 1
        )
        intermediate_size = config.moe_intermediate_size
    elif architecture == "Llama4ForConditionalGeneration":
        E = config.num_local_experts // ep_size + (
            0 if disable_shared_experts_fusion else 1
        )
        topk = config.num_experts_per_tok + (
            0 if disable_shared_experts_fusion or topk_ids_dir is None else 1
        )
        intermediate_size = config.intermediate_size
    elif architecture in [
        "Grok1ForCausalLM",
        "Grok1ImgGen",
        "Grok1AForCausalLM",
    ]:
        E = config.num_local_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture in [
        "BailingMoEForCausalLM",
        "BailingMoeForCausalLM",
        "BailingMoeV2ForCausalLM",
    ]:
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture == "HYV3ForCausalLM":
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.expert_hidden_dim
    elif architecture == "NemotronHForCausalLM":
        E = config.n_routed_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        hidden_size = getattr(config, "moe_latent_size", None) or hidden_size
    elif architecture == "Gemma4ForConditionalGeneration":
        E = config.num_experts // ep_size
        topk = config.top_k_experts
        intermediate_size = config.moe_intermediate_size
    elif architecture == "Lfm2MoeForCausalLM":
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture == "MiniMaxM3SparseForConditionalGeneration":
        # Serving fuses the shared expert into the routed-expert tensor by
        # default (E = num_local_experts + 1), so tune for that shape unless
        # fusion is explicitly disabled.
        E = config.num_local_experts // ep_size + (
            0 if disable_shared_experts_fusion else 1
        )
        topk = config.num_experts_per_tok + (
            0 if disable_shared_experts_fusion or topk_ids_dir is None else 1
        )
        intermediate_size = config.intermediate_size
    elif architecture == "UnlimitedOCRForCausalLM":
        E = config.n_routed_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    else:
        # Default: Mixtral
        E = config.num_local_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size

    shard_intermediate_size = calculate_shard_intermediate_size(
        intermediate_size, tp_size, ep_size
    )

    # gfx942 (MI300X) remaps MXFP8 [1,32]->[128,128] at load; tune must match.
    # sm100/gfx95 run native [1,32] and must not remap (mirrors fp8.py).
    if (
        architecture == "MiniMaxM3SparseForConditionalGeneration"
        and is_hip()
        and not is_gfx95_supported()
        and block_shape == [1, 32]
    ):
        block_shape = [128, 128]

    # text_config may not carry torch_dtype; fall back to bf16.
    torch_dtype = getattr(config, "torch_dtype", None) or torch.bfloat16

    return {
        "num_experts": E,
        "topk": topk,
        "hidden_size": hidden_size,
        "shard_intermediate_size": shard_intermediate_size,
        "dtype": torch_dtype,
        "block_shape": block_shape,
        "architecture": architecture,
    }


# Search axes for the ROCm/HIP backend. Declared as data rather than nested
# loops so that adding or narrowing an axis is a one-line change.
#
# matrix_instr_nonkdim and kpack are HIPOptions fields (see
# triton/backends/amd/compiler.py); they reach the compiler because the
# fused_moe launch sites splat the whole config dict into the kernel call, and
# HIPBackend.parse_options() absorbs any kwarg matching a HIPOptions field.
# Both feed the same pass: add_accelerate_matmul(pm, arch, nonkdim, kpack).
#
# matrix_instr_nonkdim selects the MFMA instruction shape: 16 forces
# mfma_16x16, 0 lets the compiler choose. 32 is deliberately NOT in the space:
# on gfx942 the compiler already picks mfma_32x32 by default, so nonkdim=32
# measured identical to nonkdim=0 (within 0.6%, against a 0.2% noise floor) on
# an int4_w4a16 MoE shape -- it only doubles the search cost.
#
# These two axes interact non-additively and the interaction changes sign with
# the shape, so they must be searched jointly rather than one at a time. On an
# E=384/N=256 int4_w4a16 shape, relative to (nonkdim=0, kpack=1):
#   M=4096 : nonkdim=16 alone -4.1%, kpack=2 alone +0.9%, together +2.9%
#   M=16384: nonkdim=16 alone +3.6%, kpack=2 alone +4.3%, together +1.5%
# Note this contradicts the general "mfma_16x16 usually wins" guidance in AMD's
# MI300X tuning guide, which is stated for fp16/fp8 GEMM; on the int4
# dequantization path nonkdim=16 is a regression for half the shapes measured.
_ROCM_SEARCH_AXES: Dict[str, List[int]] = {
    "BLOCK_SIZE_M": [32, 64, 128, 256],
    "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
    "BLOCK_SIZE_K": [32, 64, 128, 256],
    "GROUP_SIZE_M": [1, 4, 8, 16, 32],
    "num_warps": [1, 2, 4, 8],
    "num_stages": [2],
    "waves_per_eu": [0],
    "matrix_instr_nonkdim": [0, 16],
    "kpack": [1, 2],
}


def get_rocm_configs_compute_bound(
    axis_overrides: Dict[str, List[int]] = None,
) -> List[BenchmarkConfig]:
    """Cartesian product of _ROCM_SEARCH_AXES, with any axis overridable.

    An empty override list is rejected rather than honored: itertools.product
    returns nothing if any axis is empty, so a typo'd or empty value list would
    hand back a zero-config search space and the tuner would fall through to
    the default config with no error.
    """
    axes = dict(_ROCM_SEARCH_AXES)
    for key, values in (axis_overrides or {}).items():
        if key not in axes:
            raise ValueError(f"unknown search axis {key!r}; known axes: {sorted(axes)}")
        if not values:
            raise ValueError(f"search axis {key!r} was overridden with no values")
        axes[key] = values
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*axes.values())]


def get_configs_compute_bound() -> List[Dict[str, int]]:
    configs: List[BenchmarkConfig] = []
    if is_hip():
        configs = get_rocm_configs_compute_bound()
    else:
        for num_stages in [2, 3, 4, 5]:
            for block_m in [16, 32, 64, 128, 256]:
                for block_k in [64, 128, 256]:
                    for block_n in [32, 64, 128, 256]:
                        for num_warps in [4, 8]:
                            for group_size in [1, 16, 32, 64]:
                                configs.append(
                                    {
                                        "BLOCK_SIZE_M": block_m,
                                        "BLOCK_SIZE_N": block_n,
                                        "BLOCK_SIZE_K": block_k,
                                        "GROUP_SIZE_M": group_size,
                                        "num_warps": num_warps,
                                        "num_stages": num_stages,
                                    }
                                )
    return configs


# Key order for the emitted JSON, with a flag for whether the key is required.
# Optional keys are emitted only when present, so a config tuned on a backend
# that does not use them is written exactly as before.
_CONFIG_KEY_ORDER = (
    # Portable tile / launch parameters.
    ("BLOCK_SIZE_M", True),
    ("BLOCK_SIZE_N", True),
    ("BLOCK_SIZE_K", True),
    ("GROUP_SIZE_M", True),
    ("num_warps", True),
    ("num_stages", True),
    # ROCm/HIP compiler options, consumed via HIPOptions.
    ("waves_per_eu", False),
    ("matrix_instr_nonkdim", False),
    ("kpack", False),
    # Kernel-level toggles.
    ("USE_TMA", False),
)

_KNOWN_CONFIG_KEYS = frozenset(key for key, _ in _CONFIG_KEY_ORDER)


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    """Normalize key order for the saved config.

    Any key not listed in _CONFIG_KEY_ORDER is dropped, which would silently
    discard a parameter the user put in the search space -- the tuning run
    would report a speedup that the saved config cannot reproduce. Raise
    instead, so an unrecognized key is a loud failure at save time.
    """
    unknown = sorted(set(config) - _KNOWN_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"config contains keys that sort_config() would drop: {unknown}. "
            f"Add them to _CONFIG_KEY_ORDER in {__name__} so they are written "
            f"to the tuned config file."
        )
    return {
        key: config[key]
        for key, required in _CONFIG_KEY_ORDER
        if required or key in config
    }


def save_configs(
    configs: Dict[int, BenchmarkConfig],
    filename: str,
) -> None:
    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def get_config_filename(
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: List[int],
) -> str:
    dtype_str = get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int4_w4a16=use_int4_w4a16,
    )

    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    N = shard_intermediate_size // 2
    if use_int4_w4a16:
        N = N // 2

    filename = get_config_file_name(
        num_experts,
        N,
        dtype_str,
        block_shape,
        per_channel_quant,
    )

    return filename


def get_default_batch_sizes() -> List[int]:
    return [
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        256,
        512,
        1024,
        1536,
        2048,
        3072,
        4096,
    ]
