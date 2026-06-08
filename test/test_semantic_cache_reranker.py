import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import semantic_cache_system as scs


class _FakeTensor:
    def __init__(self, values):
        self.values = scs.np.array(values, dtype=float)

    def __getitem__(self, item):
        return _FakeTensor(self.values[item])

    def exp(self):
        return _FakeTensor(scs.np.exp(self.values))

    def tolist(self):
        return self.values.tolist()


class _NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeFunctional:
    @staticmethod
    def log_softmax(tensor, dim):
        values = tensor.values
        shifted = values - scs.np.max(values, axis=dim, keepdims=True)
        logged = shifted - scs.np.log(scs.np.sum(scs.np.exp(shifted), axis=dim, keepdims=True))
        return _FakeTensor(logged)


class _FakeNN:
    functional = _FakeFunctional()


class _FakeTorch:
    nn = _FakeNN()

    @staticmethod
    def inference_mode():
        return _NoopContext()

    @staticmethod
    def stack(tensors, dim=0):
        return _FakeTensor(scs.np.stack([tensor.values for tensor in tensors], axis=dim))


class _FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        pairs = kwargs["pairs"]
        self.calls.append(
            {
                "batch_size": len(pairs),
                "logits_to_keep": kwargs.get("logits_to_keep"),
            }
        )
        batch_logits = []
        for pair in pairs:
            doc_label = pair.split("<Document>: ", 1)[1]
            doc_idx = int(doc_label.replace("doc", ""))
            logits = [0.0] * 6
            logits[1] = 0.0
            logits[2] = float(doc_idx)
            batch_logits.append([logits])
        return type("ModelOutput", (), {"logits": _FakeTensor(batch_logits)})()


class RerankerTests(unittest.TestCase):
    def test_model_supports_logits_to_keep_detection(self):
        class SupportsNamedParameter:
            def forward(self, input_ids=None, logits_to_keep=0):
                return None

        class SupportsKwargs:
            def forward(self, **kwargs):
                return None

        class DoesNotSupport:
            def forward(self, input_ids=None):
                return None

        self.assertTrue(scs.Reranker._model_supports_logits_to_keep(SupportsNamedParameter()))
        self.assertTrue(scs.Reranker._model_supports_logits_to_keep(SupportsKwargs()))
        self.assertFalse(scs.Reranker._model_supports_logits_to_keep(DoesNotSupport()))

    def test_rerank_scores_in_bounded_batches_with_last_logits_only(self):
        reranker = scs.Reranker.__new__(scs.Reranker)
        reranker.model = _FakeModel()
        reranker.torch = _FakeTorch()
        reranker.token_false_id = 1
        reranker.token_true_id = 2
        reranker.batch_size = 2
        reranker._supports_logits_to_keep = True
        reranker._process_inputs = lambda pairs: {"pairs": pairs}

        results = reranker.rerank(
            "query",
            ["doc0", "doc1", "doc2", "doc3", "doc4"],
            top_k=3,
            relevance_threshold=0.0,
        )

        self.assertEqual([idx for idx, _, _ in results], [4, 3, 2])
        self.assertEqual([call["batch_size"] for call in reranker.model.calls], [2, 2, 1])
        self.assertEqual([call["logits_to_keep"] for call in reranker.model.calls], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
