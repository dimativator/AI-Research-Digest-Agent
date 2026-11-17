import unittest
from types import SimpleNamespace

from research_digest.llm.usage import DEFAULT_MODEL, MeteredClient, summarize_usage


class UsageTests(unittest.TestCase):
    def test_cost_is_summed_per_call_with_long_context_rate(self):
        responses = iter(
            [
                SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=1000, output_tokens=100)
                ),
                SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=300000, output_tokens=100)
                ),
            ]
        )
        client = MeteredClient(
            SimpleNamespace(
                responses=SimpleNamespace(create=lambda **kw: next(responses))
            )
        )
        client.responses.create(model=DEFAULT_MODEL)
        client.responses.create(model=DEFAULT_MODEL)
        total = summarize_usage(client.records)
        self.assertEqual(total["input_tokens"], 301000)
        self.assertEqual(total["output_tokens"], 200)
        self.assertAlmostEqual(total["estimated_api_cost_usd"], 0.1205)

    def test_failed_call_keeps_timing_and_marks_total_unknown(self):
        def fail(**kwargs):
            raise RuntimeError("failed")

        client = MeteredClient(SimpleNamespace(responses=SimpleNamespace(create=fail)))
        with self.assertRaises(RuntimeError):
            client.responses.create(model=DEFAULT_MODEL)
        total = summarize_usage(client.records)
        self.assertEqual(total["calls_without_usage"], 1)
        self.assertIsNone(total["estimated_api_cost_usd"])
        self.assertGreaterEqual(total["llm_latency_seconds"], 0)

    def test_unknown_model_has_tokens_but_no_price(self):
        client = MeteredClient(
            SimpleNamespace(
                responses=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(
                        usage=SimpleNamespace(input_tokens=10, output_tokens=2)
                    )
                )
            )
        )
        client.responses.create(model="other")
        total = summarize_usage(client.records)
        self.assertEqual(total["input_tokens"], 10)
        self.assertIsNone(total["estimated_api_cost_usd"])
