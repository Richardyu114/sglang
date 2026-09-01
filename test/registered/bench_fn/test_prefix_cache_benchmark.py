import asyncio
import sys
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sglang.benchmark import serving
from sglang.benchmark.datasets import DATASET_MAPPING
from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.datasets.prefix_cache import (
    _token_pool,
    assign_prefix_groups,
    compute_prefix_len,
    generate_prefix_cache_requests,
    validate_prefix_cache_config,
)
from sglang.benchmark.serving import (
    BenchmarkMetrics,
    RequestFuncOutput,
    build_prefix_cache_metadata,
    get_active_cache_prefixes,
    prime_prefix_cache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def create_lightweight_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
    vocab.update({f"tok_{i}": i + 4 for i in range(2048)})
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )


def make_metrics(completed: int, total_input: int) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_input_text=total_input,
        total_input_vision=0,
        total_output=completed,
        total_output_retokenized=completed,
        request_throughput=1.0,
        input_throughput=1.0,
        output_throughput=1.0,
        output_throughput_retokenized=1.0,
        total_throughput=1.0,
        total_throughput_retokenized=1.0,
        mean_ttft_ms=1.0,
        median_ttft_ms=1.0,
        std_ttft_ms=0.0,
        p90_ttft_ms=1.0,
        p95_ttft_ms=1.0,
        p99_ttft_ms=1.0,
        mean_tpot_ms=1.0,
        median_tpot_ms=1.0,
        std_tpot_ms=0.0,
        p90_tpot_ms=1.0,
        p95_tpot_ms=1.0,
        p99_tpot_ms=1.0,
        mean_itl_ms=1.0,
        median_itl_ms=1.0,
        std_itl_ms=0.0,
        p90_itl_ms=1.0,
        p95_itl_ms=1.0,
        p99_itl_ms=1.0,
        max_itl_ms=1.0,
        mean_e2e_latency_ms=1.0,
        median_e2e_latency_ms=1.0,
        std_e2e_latency_ms=0.0,
        p90_e2e_latency_ms=1.0,
        p95_e2e_latency_ms=1.0,
        p99_e2e_latency_ms=1.0,
        concurrency=1.0,
        max_output_tokens_per_s=1.0,
        max_concurrent_requests=1,
    )


class TestPrefixCacheDataset(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = create_lightweight_tokenizer()

    def generate(self, **overrides):
        config = {
            "num_prompts": 12,
            "input_len": 100,
            "output_len": 8,
            "hit_rate": 0.3,
            "num_groups": 3,
            "zipf_alpha": None,
            "seed": 42,
            "tokenizer": self.tokenizer,
            "return_text": False,
        }
        config.update(overrides)
        return generate_prefix_cache_requests(**config)

    def test_dataset_is_registered(self):
        self.assertIn("prefix-cache", DATASET_MAPPING)

    def test_raw_ids_keep_exact_lengths_and_boundaries(self):
        for hit_rate, expected_prefix_len in ((0.0, 0), (0.3, 30), (1.0, 100)):
            with self.subTest(hit_rate=hit_rate):
                rows = self.generate(hit_rate=hit_rate)
                self.assertEqual(len(rows), 12)
                for row in rows:
                    self.assertEqual(len(row.prompt), 100)
                    self.assertEqual(row.prompt_len, 100)
                    self.assertEqual(row.cache_prefix_len, expected_prefix_len)
                    self.assertEqual(row.prompt[:expected_prefix_len], row.cache_prefix)

    def test_uniform_assignment_is_balanced_and_deterministic(self):
        first = assign_prefix_groups(11, 4, None, seed=17)
        second = assign_prefix_groups(11, 4, None, seed=17)
        np.testing.assert_array_equal(first, second)
        counts = Counter(first.tolist())
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_single_and_per_request_group_extremes(self):
        single_group_rows = self.generate(num_groups=1)
        self.assertEqual({row.cache_group_id for row in single_group_rows}, {0})

        per_request_rows = self.generate(num_groups=12)
        self.assertEqual(
            Counter(row.cache_group_id for row in per_request_rows),
            Counter({group_id: 1 for group_id in range(12)}),
        )
        self.assertEqual(len({tuple(row.cache_prefix) for row in per_request_rows}), 12)

    def test_prefixes_use_seeded_cyclic_token_sequences(self):
        rows = self.generate(num_groups=4, input_len=20, hit_rate=0.5)
        token_pool = _token_pool(self.tokenizer, for_text=False)
        token_positions = {
            token_id: position for position, token_id in enumerate(token_pool)
        }
        prefixes = {}
        for row in rows:
            prefixes.setdefault(row.cache_group_id, row.cache_prefix)

        first_tokens = [prefixes[group_id][0] for group_id in sorted(prefixes)]
        self.assertEqual(len(first_tokens), len(set(first_tokens)))
        for prefix in prefixes.values():
            positions = [token_positions[token_id] for token_id in prefix]
            for left, right in zip(positions, positions[1:]):
                self.assertEqual(right, (left + 1) % len(token_pool))

    def test_text_token_pool_excludes_byte_fallback_tokens(self):
        vocab = {
            "[UNK]": 0,
            "[PAD]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "<0x41>": 4,
            "safe_token": 5,
        }
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
        )

        self.assertEqual(_token_pool(hf_tokenizer, for_text=True), [5])
        self.assertEqual(_token_pool(hf_tokenizer, for_text=False), [4, 5])

    def test_prefix_generation_rejects_more_groups_than_tokens(self):
        with self.assertRaisesRegex(
            ValueError,
            "at most 2048 prefix groups, but got 2049",
        ):
            self.generate(
                num_prompts=2049,
                input_len=2,
                hit_rate=0.5,
                num_groups=2049,
            )

    def test_first_suffix_token_is_unique_within_each_group(self):
        # The total request count exceeds the usable vocabulary, but each group
        # still fits. Group-local assignment should therefore isolate every
        # suffix branch without imposing an unnecessary global limit.
        rows = self.generate(
            num_prompts=3000,
            input_len=2,
            hit_rate=0.5,
            num_groups=2,
        )
        first_suffix_tokens = {}
        for row in rows:
            token = row.prompt[row.cache_prefix_len]
            first_suffix_tokens.setdefault(row.cache_group_id, []).append(token)

        for tokens in first_suffix_tokens.values():
            self.assertEqual(len(tokens), len(set(tokens)))

    def test_suffix_isolation_rejects_oversized_group(self):
        with self.assertRaisesRegex(
            ValueError,
            "at most 2048 requests per active prefix group.*group 0 has 2049",
        ):
            self.generate(
                num_prompts=2049,
                input_len=2,
                hit_rate=0.5,
                num_groups=1,
            )

    def test_zipf_assignment_is_deterministic_and_skewed(self):
        first = assign_prefix_groups(10_000, 4, 1.5, seed=7)
        second = assign_prefix_groups(10_000, 4, 1.5, seed=7)
        np.testing.assert_array_equal(first, second)
        counts = Counter(first.tolist())
        self.assertGreater(counts[0], counts[3])

    def test_text_mode_reports_retokenized_lengths(self):
        rows = self.generate(return_text=True)
        for row in rows:
            prompt_ids = self.tokenizer.encode(row.prompt)
            prefix_ids = self.tokenizer.encode(row.cache_prefix)
            expected_lcp = 0
            for prefix_token, prompt_token in zip(prefix_ids, prompt_ids):
                if prefix_token != prompt_token:
                    break
                expected_lcp += 1
            self.assertEqual(row.prompt_len, len(prompt_ids))
            self.assertEqual(row.cache_prefix_len, expected_lcp)

    def test_validation_rejects_invalid_configurations(self):
        base = {
            "num_requests": 10,
            "input_len": 100,
            "output_len": 10,
            "hit_rate": 0.5,
            "num_groups": 2,
            "zipf_alpha": None,
            "range_ratio": 1.0,
        }
        invalid_overrides = [
            {"hit_rate": -0.1},
            {"hit_rate": 1.1},
            {"hit_rate": float("nan")},
            {"num_groups": 0},
            {"num_groups": 11},
            {"zipf_alpha": 0.0},
            {"range_ratio": 0.5},
        ]
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_prefix_cache_config(**(base | overrides))

    def test_cli_validation_runs_before_benchmark_setup(self):
        argv = [
            "serving.py",
            "--dataset-name",
            "prefix-cache",
            "--random-range-ratio",
            "1.0",
            "--cache-hit-rate",
            "1.1",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(serving, "run_benchmark") as run_benchmark,
            self.assertRaises(SystemExit),
        ):
            serving.cli_main()
        run_benchmark.assert_not_called()


class TestPrefixCachePriming(CustomTestCase):
    def setUp(self):
        self.rows = [
            DatasetRow(
                prompt=[10, 11],
                prompt_len=2,
                output_len=1,
                routing_key="prefix-cache-42-0",
                cache_prefix=[10],
                cache_prefix_len=1,
                cache_group_id=0,
            ),
            DatasetRow(
                prompt=[10, 12],
                prompt_len=2,
                output_len=1,
                routing_key="prefix-cache-42-0",
                cache_prefix=[10],
                cache_prefix_len=1,
                cache_group_id=0,
            ),
            DatasetRow(
                prompt=[30, 31],
                prompt_len=2,
                output_len=1,
                routing_key="prefix-cache-42-2",
                cache_prefix=[30],
                cache_prefix_len=1,
                cache_group_id=2,
            ),
        ]

    def test_only_active_groups_are_primed_with_matching_routing_keys(self):
        calls = []

        async def request_func(request_func_input, pbar=None):
            calls.append(request_func_input)
            return RequestFuncOutput(success=True)

        count = asyncio.run(
            prime_prefix_cache(
                backend="sglang",
                request_func=request_func,
                api_url="http://server/generate",
                model_id="model",
                input_requests=self.rows,
                lora_names=[],
                extra_request_body={},
            )
        )

        self.assertEqual(count, 2)
        self.assertEqual([call.prompt for call in calls], [[10], [30]])
        self.assertEqual(
            [call.routing_key for call in calls],
            ["prefix-cache-42-0", "prefix-cache-42-2"],
        )
        self.assertTrue(all(call.output_len == 1 for call in calls))
        self.assertTrue(
            all(
                call.extra_request_body["sampling_params"]
                == {"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}
                for call in calls
            )
        )

    def test_zero_length_prefix_does_not_prime(self):
        rows = [
            DatasetRow(
                prompt=[1, 2],
                prompt_len=2,
                output_len=1,
                cache_prefix=[],
                cache_prefix_len=0,
                cache_group_id=0,
            )
        ]
        request_func = MagicMock()
        count = asyncio.run(
            prime_prefix_cache(
                backend="sglang",
                request_func=request_func,
                api_url="http://server/generate",
                model_id="model",
                input_requests=rows,
                lora_names=[],
                extra_request_body={},
            )
        )
        self.assertEqual(count, 0)
        request_func.assert_not_called()

    def test_prime_failure_aborts_with_group_context(self):
        async def request_func(request_func_input, pbar=None):
            return RequestFuncOutput(success=False, error="server rejected request")

        with self.assertRaisesRegex(ValueError, "group 0.*server rejected request"):
            asyncio.run(
                prime_prefix_cache(
                    backend="sglang",
                    request_func=request_func,
                    api_url="http://server/generate",
                    model_id="model",
                    input_requests=self.rows,
                    lora_names=[],
                    extra_request_body={},
                )
            )

    def test_inconsistent_group_metadata_is_rejected(self):
        rows = self.rows[:2]
        rows[1].cache_prefix = [99]
        with self.assertRaisesRegex(ValueError, "same group"):
            get_active_cache_prefixes(rows)

    def test_metadata_reports_effective_and_requested_values(self):
        metadata = build_prefix_cache_metadata(
            self.rows,
            requested_input_len=10,
            requested_hit_rate=0.3,
            requested_num_groups=4,
            zipf_alpha=1.2,
            prime_request_count=2,
        )
        self.assertEqual(
            metadata["requested_prefix_tokens"], compute_prefix_len(10, 0.3)
        )
        self.assertEqual(metadata["effective_prefix_tokens"], 1)
        self.assertEqual(metadata["cache_num_groups_active"], 2)
        self.assertEqual(metadata["cache_prime_footprint_tokens"], 2)
        self.assertFalse(metadata["cache_page_alignment_applied"])


class TestPrefixCacheLifecycle(CustomTestCase):
    def test_warmup_flush_prime_then_timer(self):
        events = []
        rows = [
            DatasetRow(
                prompt=[10, 11],
                prompt_len=2,
                output_len=1,
                routing_key="prefix-cache-42-0",
                cache_prefix=[10],
                cache_prefix_len=1,
                cache_group_id=0,
            ),
            DatasetRow(
                prompt=[20, 21],
                prompt_len=2,
                output_len=1,
                routing_key="prefix-cache-42-1",
                cache_prefix=[20],
                cache_prefix_len=1,
                cache_group_id=1,
            ),
        ]

        async def request_func(request_func_input, pbar=None):
            events.append(("request", tuple(request_func_input.prompt)))
            return RequestFuncOutput(
                success=True,
                prompt_len=request_func_input.prompt_len,
                output_len=1,
                latency=0.001,
                ttft=0.001,
            )

        async def profile_request(api_url):
            events.append(("profile", api_url.rsplit("/", 1)[-1]))
            return RequestFuncOutput(success=True)

        benchmark_args = SimpleNamespace(
            dataset_name="prefix-cache",
            warmup_requests=1,
            plot_throughput=False,
            profile_steps=None,
            profile_num_steps=None,
            cache_report=False,
            backend="test-prefix-cache",
            tag=None,
            sharegpt_output_len=None,
            random_input_len=2,
            random_output_len=1,
            random_range_ratio=1.0,
            cache_hit_rate=0.5,
            cache_num_groups=2,
            cache_zipf_alpha=None,
            output_details=False,
        )

        response = MagicMock(status_code=404)
        response.json.return_value = {}
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as output_file:
            benchmark_args.output_file = output_file.name
            serving.set_global_args(benchmark_args)

            def flush(*_args, **_kwargs):
                events.append(("flush", None))

            def perf_counter():
                events.append(("timer", None))
                return float(len(events))

            with (
                patch.dict(
                    serving.ASYNC_REQUEST_FUNCS,
                    {"test-prefix-cache": request_func},
                ),
                patch.object(serving, "flush_server_cache", side_effect=flush),
                patch.object(serving.time, "sleep"),
                patch.object(serving.time, "perf_counter", side_effect=perf_counter),
                patch.object(serving.requests, "get", return_value=response),
                patch.object(
                    serving,
                    "async_request_profile",
                    side_effect=profile_request,
                ),
                patch.object(
                    serving,
                    "calculate_metrics",
                    return_value=(make_metrics(2, 4), [1, 1]),
                ),
            ):
                result = asyncio.run(
                    serving.benchmark(
                        backend="test-prefix-cache",
                        api_url="http://server/generate",
                        base_url="http://server",
                        model_id="model",
                        tokenizer=MagicMock(),
                        input_requests=rows,
                        request_rate=float("inf"),
                        max_concurrency=2,
                        disable_tqdm=True,
                        lora_names=[],
                        lora_request_distribution=None,
                        lora_zipf_alpha=None,
                        extra_request_body={},
                        profile=True,
                        flush_cache=False,
                        warmup_requests=1,
                    )
                )

        self.assertEqual(
            events[:6],
            [
                ("request", (10, 11)),
                ("flush", None),
                ("request", (10,)),
                ("request", (20,)),
                ("profile", "start_profile"),
                ("timer", None),
            ],
        )
        self.assertEqual(result["completed"], 2)
        self.assertEqual(len(result["input_lens"]), 2)
        self.assertEqual(result["cache_prime_request_count"], 2)


if __name__ == "__main__":
    unittest.main()
