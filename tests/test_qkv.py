"""The QKV cache must stay within its budget and never reuse changed input."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("lbh_qkv_test", Path(__file__).resolve().parents[1] / "qkv.py")
qkv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qkv)


class TensorView:
    def __init__(self, rows, width, pointer=1):
        self.shape = (rows, width)
        self.pointer = pointer
        self.dtype, self.device, self._version = "bfloat16", "cuda:0", 0

    def data_ptr(self):
        return self.pointer

    def stride(self):
        return (self.shape[1], 1)

    def is_inference(self):
        return False

    def numel(self):
        return self.shape[0] * self.shape[1]

    def element_size(self):
        return 2


def projection(x):
    return TensorView(x.shape[0], qkv.QKV_WIDTH)


class QKVTests(unittest.TestCase):
    def test_high_resolution_budget_and_release(self):
        wrapper = qkv.BoundedQKVProjection(projection, 8193)
        inputs = [TensorView(rows, 5376, i + 1) for i, rows in enumerate((4096, 4096, 1))]
        first = [wrapper(x) for x in inputs]
        second = [wrapper(x) for x in inputs]
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertIsNot(first[2], second[2])
        self.assertEqual(wrapper.peak_cache_bytes, 336 * 1024**2)
        self.assertEqual((wrapper.cache_hits, wrapper.projected_chunks), (2, 4))
        self.assertEqual(wrapper.live_cache_bytes, 0)
        self.assertTrue(wrapper.released)
        with self.assertRaises(RuntimeError):
            wrapper(inputs[0])

    def test_version_change_prevents_reuse(self):
        x = TensorView(9, 5376)
        wrapper = qkv.BoundedQKVProjection(projection, 9)
        first = wrapper(x)
        x._version += 1
        self.assertIsNot(first, wrapper(x))
        self.assertEqual((wrapper.key_misses, wrapper.cache_hits), (1, 0))

    def test_disabled_and_single_pass_store_nothing(self):
        x = TensorView(9, 5376)
        for options in ({"enabled": False}, {"cold": False}):
            with self.subTest(options=options):
                wrapper = qkv.BoundedQKVProjection(projection, 9, **options)
                for _ in range(wrapper.expected_calls):
                    wrapper(x)
                self.assertEqual((wrapper.cache_stores, wrapper.cache_hits, wrapper.live_cache_bytes), (0, 0, 0))

    def test_small_budget_skips_storage(self):
        x = TensorView(9, 5376)
        wrapper = qkv.BoundedQKVProjection(projection, 9, max_cache_bytes=1)
        wrapper(x)
        wrapper(x)
        self.assertEqual((wrapper.budget_skips, wrapper.cache_hits), (1, 0))

    def test_invalid_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            qkv.BoundedQKVProjection(projection, 1, max_cache_bytes=qkv.MAX_CACHE_BYTES + 1)
        wrapper = qkv.BoundedQKVProjection(projection, 9)
        with self.assertRaises(RuntimeError):
            wrapper(TensorView(8, 5376))
        wrapper.release()


if __name__ == "__main__":
    unittest.main()
