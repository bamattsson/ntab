import math

import numpy as np
import pytest
import torch

from nfab_baseline.constants import FP_SIZE, N_MOL_PROP_FEATURES
from nfab_baseline.model import AffinityModel
from nfab_evaluate.metrics import pearson_r_per_assay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(
    n_targets: int = 10,
    n_standard_types: int = 3,
    hidden_dim: int = 64,
    embed_dim: int = 16,
) -> AffinityModel:
    return AffinityModel(
        n_targets=n_targets,
        n_standard_types=n_standard_types,
        hidden_dim=hidden_dim,
        target_embed_dim=embed_dim,
        min_assay_size=3,  # small for tests
        lr=1e-3,
    )


def _make_batch(
    batch_size: int = 8,
    n_targets: int = 10,
    n_standard_types: int = 3,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]
]:
    rng = torch.Generator().manual_seed(0)
    fps = torch.rand(batch_size, FP_SIZE, generator=rng)
    props = torch.randn(batch_size, N_MOL_PROP_FEATURES)
    target_idx = torch.randint(0, n_targets, (batch_size,))
    standard_type_idx = torch.randint(0, n_standard_types, (batch_size,))
    labels = torch.rand(batch_size) * 5 + 4
    assay_ids = [f"ASSAY_{i % 3}" for i in range(batch_size)]
    return fps, props, target_idx, standard_type_idx, labels, assay_ids


# ---------------------------------------------------------------------------
# pearson_r_per_assay
# ---------------------------------------------------------------------------


class TestPearsonRPerAssay:
    def test_perfect_correlation_returns_one(self) -> None:
        # Predictions == labels → Pearson r = 1.0
        vals = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
        assay_ids = ["A"] * 5
        r, _, _ = pearson_r_per_assay(
            preds=vals, labels=vals, assay_ids=assay_ids, min_assay_size=3
        )
        assert pytest.approx(r, abs=1e-5) == 1.0

    def test_perfect_anticorrelation_returns_minus_one(self) -> None:
        preds = np.array([8.0, 7.0, 6.0, 5.0, 4.0])
        labels = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
        assay_ids = ["A"] * 5
        r, _, _ = pearson_r_per_assay(
            preds=preds, labels=labels, assay_ids=assay_ids, min_assay_size=3
        )
        assert pytest.approx(r, abs=1e-5) == -1.0

    def test_averages_across_assays(self) -> None:
        # Assay A: perfect correlation (r=1), Assay B: perfect anticorrelation (r=-1)
        # Mean should be 0
        preds = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
        labels = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        assay_ids = ["A", "A", "A", "B", "B", "B"]
        r, _, _ = pearson_r_per_assay(
            preds=preds, labels=labels, assay_ids=assay_ids, min_assay_size=3
        )
        assert pytest.approx(r, abs=1e-5) == 0.0

    def test_assays_below_min_size_are_skipped(self) -> None:
        # Assay A has 5 samples (qualifies), Assay B has 2 (skipped)
        # Only Assay A contributes, which has perfect correlation
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 1.0])
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 9.0])
        assay_ids = ["A", "A", "A", "A", "A", "B", "B"]
        r, _, _ = pearson_r_per_assay(
            preds=preds, labels=labels, assay_ids=assay_ids, min_assay_size=3
        )
        assert pytest.approx(r, abs=1e-5) == 1.0

    def test_all_assays_below_min_size_returns_nan(self) -> None:
        preds = np.array([1.0, 2.0])
        labels = np.array([1.0, 2.0])
        assay_ids = ["A", "A"]
        r, _, _ = pearson_r_per_assay(
            preds=preds, labels=labels, assay_ids=assay_ids, min_assay_size=3
        )
        assert math.isnan(r)

    def test_default_min_assay_size_is_10(self) -> None:
        # 9 samples in one assay → should be skipped → NaN
        preds = np.arange(9, dtype=np.float32)
        labels = np.arange(9, dtype=np.float32)
        assay_ids = ["A"] * 9
        r, _, _ = pearson_r_per_assay(preds=preds, labels=labels, assay_ids=assay_ids)
        assert math.isnan(r)

    def test_size_weighted_differs_from_macro_for_unequal_assays(self) -> None:
        # Assay A (3 samples, r≈1) and Assay B (9 samples, r≈-1).
        # Macro avg = (1 + -1) / 2 = 0.
        # Size-weighted avg = (3*1 + 9*(-1)) / 12 = -0.5.
        preds_a = np.array([1.0, 2.0, 3.0])
        labels_a = np.array([1.0, 2.0, 3.0])
        preds_b = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        labels_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        preds = np.concatenate([preds_a, preds_b])
        labels = np.concatenate([labels_a, labels_b])
        assay_ids = ["A"] * 3 + ["B"] * 9
        r_macro, _, _ = pearson_r_per_assay(
            preds, labels, assay_ids, min_assay_size=3, weighted=False
        )
        r_weighted, _, _ = pearson_r_per_assay(
            preds, labels, assay_ids, min_assay_size=3, weighted=True
        )
        assert pytest.approx(r_macro, abs=1e-5) == 0.0
        assert pytest.approx(r_weighted, abs=1e-5) == -0.5


# ---------------------------------------------------------------------------
# AffinityModel — forward pass
# ---------------------------------------------------------------------------


class TestAffinityModelForward:
    def test_output_shape_is_batch_by_1(self) -> None:
        model = _make_model()
        fps, props, target_idx, standard_type_idx, _, _ = _make_batch(batch_size=8)
        out = model(fps, props, target_idx, standard_type_idx)
        assert out.shape == (8, 1)

    def test_output_shape_batch_size_1(self) -> None:
        model = _make_model().eval()
        fps, props, target_idx, standard_type_idx, _, _ = _make_batch(batch_size=1)
        out = model(fps, props, target_idx, standard_type_idx)
        assert out.shape == (1, 1)

    def test_output_is_float32(self) -> None:
        model = _make_model()
        fps, props, target_idx, standard_type_idx, _, _ = _make_batch()
        out = model(fps, props, target_idx, standard_type_idx)
        assert out.dtype == torch.float32

    def test_different_targets_produce_different_outputs(self) -> None:
        # Same fingerprint, props and standard type, different target → different prediction
        model = _make_model(n_targets=5).eval()
        fp = torch.rand(1, 2048)
        props = torch.randn(1, N_MOL_PROP_FEATURES)
        std_type = torch.tensor([0])
        out_0 = model(fp, props, torch.tensor([0]), std_type)
        out_1 = model(fp, props, torch.tensor([1]), std_type)
        assert not torch.allclose(out_0, out_1)

    def test_different_standard_types_produce_different_outputs(self) -> None:
        # Same fingerprint, props and target, different standard type → different prediction
        model = _make_model(n_targets=5, n_standard_types=3).eval()
        fp = torch.rand(1, 2048)
        props = torch.randn(1, N_MOL_PROP_FEATURES)
        target = torch.tensor([0])
        out_0 = model(fp, props, target, torch.tensor([0]))
        out_1 = model(fp, props, target, torch.tensor([1]))
        assert not torch.allclose(out_0, out_1)


# ---------------------------------------------------------------------------
# AffinityModel — training_step and validation_step
# ---------------------------------------------------------------------------


class TestAffinityModelSteps:
    def test_training_step_returns_scalar_loss(self) -> None:
        model = _make_model()
        batch = _make_batch()
        loss = model.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_loss_decreases_on_repeated_steps(self) -> None:
        # Overfit sanity check: loss should decrease after several gradient steps
        # on a fixed tiny batch
        model = _make_model(n_targets=2, hidden_dim=32, embed_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        fps = torch.zeros(4, 2048)
        props = torch.zeros(4, N_MOL_PROP_FEATURES)
        target_idx = torch.tensor([0, 0, 1, 1])
        standard_type_idx = torch.tensor([0, 0, 0, 0])
        labels = torch.tensor([5.0, 5.0, 8.0, 8.0])
        assay_ids = ["A", "A", "A", "A"]
        batch = (fps, props, target_idx, standard_type_idx, labels, assay_ids)

        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx=0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss did not decrease"

    def test_validation_step_runs_without_error(self) -> None:
        model = _make_model(n_targets=10)
        fps = torch.rand(6, 2048)
        props = torch.randn(6, N_MOL_PROP_FEATURES)
        target_idx = torch.zeros(6, dtype=torch.long)
        standard_type_idx = torch.zeros(6, dtype=torch.long)
        labels = torch.rand(6)
        assay_ids = ["A"] * 6
        batch = (fps, props, target_idx, standard_type_idx, labels, assay_ids)
        model.validation_step(batch, batch_idx=0)  # should not raise

    def test_validation_step_accumulates_state(self) -> None:
        model = _make_model()
        batch = _make_batch(batch_size=6)
        model.validation_step(batch, batch_idx=0)
        assert len(model._val_preds) == 1  # one batch tensor appended
        assert len(model._val_assay_ids) == 6  # 6 assay id strings

    def test_accumulated_state_cleared_after_validation_epoch_end(self) -> None:
        model = _make_model()
        model.validation_step(_make_batch(batch_size=6), batch_idx=0)
        model.on_validation_epoch_end()
        assert model._val_preds == []
        assert model._val_labels == []
        assert model._val_assay_ids == []

    def test_pearson_r_uses_full_epoch_not_per_batch(self) -> None:
        # Assay "A" has 2 samples in each of 2 batches (4 total).
        # min_assay_size=3, so per-batch each would be skipped (only 2 each).
        # Accumulated across the epoch all 4 qualify → a real r is computed.
        model = _make_model(n_targets=1)
        fp = torch.zeros(2, 2048)
        props = torch.zeros(2, N_MOL_PROP_FEATURES)
        t = torch.zeros(2, dtype=torch.long)
        s = torch.zeros(2, dtype=torch.long)
        batch1 = (fp, props, t, s, torch.tensor([1.0, 2.0]), ["A", "A"])
        batch2 = (fp, props, t, s, torch.tensor([3.0, 4.0]), ["A", "A"])
        model.validation_step(batch1, batch_idx=0)
        model.validation_step(batch2, batch_idx=1)
        # 4 samples accumulated across both batches
        assert sum(p.numel() for p in model._val_preds) == 4

    def test_test_step_accumulates_and_clears(self) -> None:
        model = _make_model()
        batch = _make_batch(batch_size=6)
        model.test_step(batch, batch_idx=0)
        assert len(model._test_preds) == 1
        model.on_test_epoch_end()
        assert model._test_preds == []
