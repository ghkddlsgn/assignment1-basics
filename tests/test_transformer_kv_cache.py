import unittest

import torch

from cs336_basics.transformer_LM import PreNormTransformerBlock, TransformerLM


class TransformerKVCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = TransformerLM(
            hidden_dim=16, vocab_size=32, num_layers=2, num_heads=2,
            max_kv_len=16, device="cpu", dtype=torch.float32,
        )
        self.tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    def test_cached_logits_match_full_sequence(self):
        self.model.eval()
        for chunk_sizes in [(1, 1, 1, 1), (2, 2), (3, 1), (4,)]:
            with self.subTest(chunk_sizes=chunk_sizes):
                self.model.reset_kv_cache()
                pieces = []
                seen = 0
                pointers = None
                with torch.no_grad():
                    expected = self.model(self.tokens)
                    for size in chunk_sizes:
                        pieces.append(self.model(self.tokens[:, seen : seen + size], use_cache=True))
                        seen += size
                        for cache in self.model.kv_cache:
                            self.assertEqual(cache["k"].len, seen)
                            self.assertEqual(cache["v"].len, seen)
                        current_pointers = [
                            (cache["k"].buffer.data_ptr(), cache["v"].buffer.data_ptr())
                            for cache in self.model.kv_cache
                        ]
                        if pointers is None:
                            pointers = current_pointers
                        else:
                            self.assertEqual(current_pointers, pointers)
                torch.testing.assert_close(
                    torch.cat(pieces, dim=1), expected, atol=1e-5, rtol=1e-5
                )

    def test_training_ignores_existing_cache_and_preserves_gradients(self):
        self.model.eval()
        with torch.no_grad():
            self.model(self.tokens[:, :2], use_cache=True)
            expected = self.model(self.tokens)
            snapshots = [
                (cache["k"].get_cache().clone(), cache["v"].get_cache().clone())
                for cache in self.model.kv_cache
            ]

        self.model.train()
        for _ in range(2):
            self.model.zero_grad(set_to_none=True)
            actual = self.model(self.tokens, use_cache=False)
            torch.testing.assert_close(actual, expected)
            actual.square().mean().backward()
            for parameter in self.model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())

        for cache, (keys, values) in zip(self.model.kv_cache, snapshots, strict=True):
            self.assertEqual(cache["k"].len, 2)
            self.assertEqual(cache["v"].len, 2)
            torch.testing.assert_close(cache["k"].get_cache(), keys)
            torch.testing.assert_close(cache["v"].get_cache(), values)
            self.assertFalse(cache["k"].buffer.requires_grad)
            self.assertFalse(cache["v"].buffer.requires_grad)

    def test_reset_clears_the_buffers_attention_uses(self):
        for block, cache in zip(self.model.transformer_blocks, self.model.kv_cache, strict=True):
            attention = block.multi_head_attention_rope
            self.assertIs(attention.k_cache, cache["k"])
            self.assertIs(attention.v_cache, cache["v"])

        with torch.no_grad():
            self.model(self.tokens, use_cache=True)
            self.model.reset_kv_cache()
            for cache in self.model.kv_cache:
                self.assertEqual(cache["k"].len, 0)
                self.assertEqual(cache["v"].len, 0)
            new_tokens = self.tokens[:1, 1:]
            expected = self.model(new_tokens)
            actual = self.model(new_tokens, use_cache=True)
        torch.testing.assert_close(actual, expected)
        for cache in self.model.kv_cache:
            self.assertEqual(cache["k"].len, 3)
            self.assertEqual(cache["v"].len, 3)

    def test_registered_caches_follow_model_dtype(self):
        self.model.to(device="cpu", dtype=torch.float64)
        for cache in self.model.kv_cache:
            self.assertEqual(cache["k"].buffer.dtype, torch.float64)
            self.assertEqual(cache["v"].buffer.dtype, torch.float64)
        self.assertFalse(any("cache" in name for name in self.model.state_dict()))

    def test_standalone_block_returns_tensor_and_keeps_training_cache_empty(self):
        block = PreNormTransformerBlock(
            hidden_dim=16, num_heads=2, context_length=16, device="cpu", dtype=torch.float32
        )
        inputs = torch.randn(2, 3, 16, requires_grad=True)
        output = block(inputs)
        self.assertEqual(output.shape, inputs.shape)
        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all().item())
        attention = block.multi_head_attention_rope
        self.assertEqual(attention.k_cache.len, 0)
        self.assertEqual(attention.v_cache.len, 0)


if __name__ == "__main__":
    unittest.main()
