import math
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    build_zipf_group_probabilities,
    get_available_tokens,
)


def validate_prefix_cache_config(
    *,
    num_requests: int,
    input_len: int,
    output_len: int,
    hit_rate: float,
    num_groups: int,
    zipf_alpha: Optional[float],
    range_ratio: float,
) -> None:
    if num_requests <= 0:
        raise ValueError(f"--num-prompts must be > 0, got {num_requests}")
    if input_len <= 0:
        raise ValueError(f"--random-input-len must be > 0, got {input_len}")
    if output_len < 0:
        raise ValueError(f"--random-output-len must be >= 0, got {output_len}")
    if not math.isfinite(hit_rate) or not 0.0 <= hit_rate <= 1.0:
        raise ValueError(
            f"--cache-hit-rate must be a finite float in [0, 1], got {hit_rate!r}"
        )
    if not 1 <= num_groups <= num_requests:
        raise ValueError(
            "--cache-num-groups must be between 1 and --num-prompts "
            f"({num_requests}), got {num_groups}"
        )
    if zipf_alpha is not None and (not math.isfinite(zipf_alpha) or zipf_alpha <= 0):
        raise ValueError(
            f"--cache-zipf-alpha must be a finite float > 0, got {zipf_alpha!r}"
        )
    if range_ratio != 1.0:
        raise ValueError(
            "--dataset-name=prefix-cache currently requires "
            f"--random-range-ratio=1.0, got {range_ratio}"
        )


def compute_prefix_len(total_input_len: int, hit_rate: float) -> int:
    """Compute the requested reusable prefix length.

    This first implementation intentionally leaves page alignment to the server
    and reports the observed cached-token count separately.
    """
    return math.floor(total_input_len * hit_rate)


def assign_prefix_groups(
    num_prompts: int,
    num_groups: int,
    zipf_alpha: Optional[float],
    seed: int,
) -> np.ndarray:
    """Assign exactly ``num_prompts`` requests to prefix groups."""
    rng = np.random.default_rng(seed)
    if zipf_alpha is None:
        assignments = np.arange(num_prompts, dtype=np.int64) % num_groups
        rng.shuffle(assignments)
        return assignments

    probabilities = build_zipf_group_probabilities(num_groups, zipf_alpha)
    return rng.choice(
        num_groups,
        size=num_prompts,
        replace=True,
        p=probabilities,
    )


def _token_pool(tokenizer: PreTrainedTokenizerBase) -> List[int]:
    available = list(dict.fromkeys(get_available_tokens(tokenizer)))
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        available = [token_id for token_id in available if token_id < vocab_size]
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    non_special = [token_id for token_id in available if token_id not in special_ids]
    token_ids = non_special or available
    if not token_ids:
        raise ValueError("Tokenizer vocabulary does not contain any integer token IDs")
    return token_ids


def _unique_marker_width(
    num_values: int,
    base: int,
    sequence_len: int,
) -> int:
    if num_values <= 1 or sequence_len == 0:
        return 0

    capacity = 1
    for width in range(1, sequence_len + 1):
        capacity *= base
        if capacity >= num_values:
            return width
    raise ValueError(
        f"Cannot generate {num_values} distinct token sequences of length "
        f"{sequence_len} from a vocabulary of {base} usable tokens"
    )


def _generate_unique_sequences(
    *,
    count: int,
    length: int,
    token_ids: Sequence[int],
    rng: np.random.Generator,
) -> List[List[int]]:
    if length == 0:
        return [[] for _ in range(count)]

    width = _unique_marker_width(count, len(token_ids), length)
    result = []
    for sequence_id in range(count):
        sequence = rng.choice(token_ids, size=length, replace=True).tolist()
        marker = sequence_id
        for position in range(width):
            sequence[position] = token_ids[marker % len(token_ids)]
            marker //= len(token_ids)
        result.append([int(token_id) for token_id in sequence])
    return result


def _generate_group_isolated_suffixes(
    *,
    assignments: Sequence[int],
    length: int,
    token_ids: Sequence[int],
    rng: np.random.Generator,
) -> List[List[int]]:
    """Generate suffixes that branch immediately within each prefix group."""
    if length == 0:
        return [[] for _ in assignments]

    suffixes = [
        [int(token_id) for token_id in rng.choice(token_ids, size=length)]
        for _ in assignments
    ]
    requests_by_group = {}
    for request_id, group_id in enumerate(assignments):
        requests_by_group.setdefault(int(group_id), []).append(request_id)

    vocab_size = len(token_ids)
    for group_id in sorted(requests_by_group):
        request_ids = requests_by_group[group_id]
        if len(request_ids) > vocab_size:
            raise ValueError(
                "Prefix-cache suffix isolation supports at most "
                f"{vocab_size} requests per active prefix group, but group "
                f"{group_id} has {len(request_ids)}. Increase "
                "--cache-num-groups or reduce --num-prompts."
            )

        offset = int(rng.integers(vocab_size))
        for local_index, request_id in enumerate(request_ids):
            suffixes[request_id][0] = token_ids[(offset + local_index) % vocab_size]

    return suffixes


def _decode(tokenizer: PreTrainedTokenizerBase, token_ids: List[int]) -> str:
    try:
        return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(token_ids)


def _encode(tokenizer: PreTrainedTokenizerBase, text: str) -> List[int]:
    encoded = tokenizer.encode(text)
    return encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)


def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def generate_prefix_cache_requests(
    *,
    num_prompts: int,
    input_len: int,
    output_len: int,
    hit_rate: float,
    num_groups: int,
    zipf_alpha: Optional[float],
    seed: int,
    tokenizer: PreTrainedTokenizerBase,
    return_text: bool,
) -> List[DatasetRow]:
    prefix_len = compute_prefix_len(input_len, hit_rate)
    suffix_len = input_len - prefix_len
    token_ids = _token_pool(tokenizer)

    # Keep assignment and token generation independent so changes to one do
    # not perturb the other for a fixed seed.
    assignments = assign_prefix_groups(
        num_prompts=num_prompts,
        num_groups=num_groups,
        zipf_alpha=zipf_alpha,
        seed=seed,
    )
    token_rng = np.random.default_rng(seed + 1)
    prefixes = _generate_unique_sequences(
        count=num_groups,
        length=prefix_len,
        token_ids=token_ids,
        rng=token_rng,
    )
    suffixes = _generate_group_isolated_suffixes(
        assignments=assignments,
        length=suffix_len,
        token_ids=token_ids,
        rng=token_rng,
    )

    rows = []
    for request_id, group_id_value in enumerate(assignments):
        group_id = int(group_id_value)
        prefix_ids = prefixes[group_id]
        prompt_ids = prefix_ids + suffixes[request_id]

        if return_text:
            prefix = _decode(tokenizer, prefix_ids) if prefix_ids else ""
            suffix = _decode(tokenizer, suffixes[request_id])
            prompt = prefix + suffix
            prompt_token_ids = _encode(tokenizer, prompt)
            if prefix_ids:
                primed_token_ids = _encode(tokenizer, prefix)
                effective_prefix_len = _common_prefix_len(
                    primed_token_ids, prompt_token_ids
                )
            else:
                effective_prefix_len = 0
            effective_prompt_len = len(prompt_token_ids)
        else:
            prefix = list(prefix_ids)
            prompt = prompt_ids
            effective_prefix_len = prefix_len
            effective_prompt_len = input_len

        rows.append(
            DatasetRow(
                prompt=prompt,
                prompt_len=effective_prompt_len,
                output_len=output_len,
                routing_key=f"prefix-cache-{seed}-{group_id}",
                cache_prefix=prefix,
                cache_prefix_len=effective_prefix_len,
                cache_group_id=group_id,
            )
        )

    active_groups = len(set(int(group_id) for group_id in assignments))
    mode = "text" if return_text else "token IDs"
    print("\nGenerated prefix-cache dataset statistics:")
    print(f"Input mode: {mode}")
    print(f"Total prompts: {len(rows)}")
    print(f"Requested input tokens per prompt: {input_len}")
    print(f"Requested prefix tokens: {prefix_len}")
    print(f"Requested cache hit rate: {hit_rate:.4f}")
    print(f"Active/requested prefix groups: {active_groups}/{num_groups}")
    if zipf_alpha is not None:
        print(f"Zipf alpha: {zipf_alpha}")

    return rows


@dataclass
class PrefixCacheDataset(BaseDataset):
    num_requests: int
    input_len: int
    output_len: int
    hit_rate: float
    num_groups: int
    zipf_alpha: Optional[float]
    seed: int
    return_text: bool
    range_ratio: float

    @classmethod
    def from_args(cls, args: Namespace) -> "PrefixCacheDataset":
        config = cls(
            num_requests=args.num_prompts,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            hit_rate=getattr(args, "cache_hit_rate", 0.0),
            num_groups=getattr(args, "cache_num_groups", 1),
            zipf_alpha=getattr(args, "cache_zipf_alpha", None),
            seed=args.seed,
            return_text=not getattr(args, "tokenize_prompt", False),
            range_ratio=getattr(args, "random_range_ratio", 1.0),
        )
        validate_prefix_cache_config(
            num_requests=config.num_requests,
            input_len=config.input_len,
            output_len=config.output_len,
            hit_rate=config.hit_rate,
            num_groups=config.num_groups,
            zipf_alpha=config.zipf_alpha,
            range_ratio=config.range_ratio,
        )
        return config

    def load(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_id: Optional[str] = None,
    ) -> List[DatasetRow]:
        return generate_prefix_cache_requests(
            num_prompts=self.num_requests,
            input_len=self.input_len,
            output_len=self.output_len,
            hit_rate=self.hit_rate,
            num_groups=self.num_groups,
            zipf_alpha=self.zipf_alpha,
            seed=self.seed,
            tokenizer=tokenizer,
            return_text=self.return_text,
        )
