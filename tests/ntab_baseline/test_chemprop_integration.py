"""Integration tests: Dataset + DataLoader + Model with Chemprop enabled."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from ntab_baseline.chemprop_utils import collate_batch, smiles_to_molgraph
from ntab_baseline.constants import FP_SIZE, N_MOL_PROP_FEATURES
from ntab_baseline.dataset import AffinityDataset
from ntab_baseline.model import AffinityModel

SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC", "CC(C)O"]


def _make_chemprop_dataset(n: int = 6) -> AffinityDataset:
    rng = np.random.default_rng(42)
    fps_matrix = torch.from_numpy(rng.random((n, FP_SIZE)).astype(np.float32))
    props_matrix = torch.from_numpy(
        rng.standard_normal((n, N_MOL_PROP_FEATURES)).astype(np.float32)
    )
    mol_graphs = [smiles_to_molgraph(s) for s in SMILES[:n]]
    return AffinityDataset(
        fps_matrix=fps_matrix,
        props_matrix=props_matrix,
        fp_indices=list(range(n)),
        target_indices=[0] * n,
        standard_type_indices=[0] * n,
        labels=rng.random(n).astype(np.float32) * 4 + 4,
        assay_ids=[f"A_{i % 2}" for i in range(n)],
        mol_graphs=mol_graphs,
    )


def _make_chemprop_model(
    use_fps: bool = False,
    use_mol_props: bool = False,
    chemprop_d_h: int = 32,
) -> AffinityModel:
    return AffinityModel(
        n_targets=1,
        hidden_dim=32,
        target_embed_dim=8,
        min_assay_size=2,
        use_fps=use_fps,
        use_mol_props=use_mol_props,
        use_chemprop=True,
        chemprop_d_h=chemprop_d_h,
        lr=1e-3,
    )


class TestChempropDataloaderIntegration:
    def test_dataloader_with_collate_batch(self) -> None:
        ds = _make_chemprop_dataset(n=4)
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_batch)
        fps, props, t_idx, st_idx, labels, aids, bmg = next(iter(loader))
        assert fps.shape == (2, FP_SIZE)
        assert bmg is not None

    def test_full_forward_pass_through_dataloader(self) -> None:
        ds = _make_chemprop_dataset(n=4)
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_batch)
        model = _make_chemprop_model()
        fps, props, t_idx, st_idx, labels, aids, bmg = next(iter(loader))
        out = model(fps, props, t_idx, st_idx, bmg=bmg)
        assert out.shape == (4, 1)

    def test_training_step_with_chemprop(self) -> None:
        ds = _make_chemprop_dataset(n=4)
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_batch)
        model = _make_chemprop_model()
        batch = next(iter(loader))
        loss = model.training_step(batch, batch_idx=0)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_combined_fps_and_chemprop(self) -> None:
        ds = _make_chemprop_dataset(n=4)
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_batch)
        model = _make_chemprop_model(use_fps=True, use_mol_props=True)
        batch = next(iter(loader))
        loss = model.training_step(batch, batch_idx=0)
        assert loss.shape == ()

    def test_gradient_flows_through_chemprop(self) -> None:
        ds = _make_chemprop_dataset(n=4)
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_batch)
        model = _make_chemprop_model()
        batch = next(iter(loader))
        loss = model.training_step(batch, batch_idx=0)
        loss.backward()
        mp_grads = [
            p.grad for p in model.chemprop_mp.parameters() if p.grad is not None
        ]
        assert len(mp_grads) > 0, "No gradients flowed to chemprop_mp"

    def test_loss_decreases_with_chemprop(self) -> None:
        ds = _make_chemprop_dataset(n=6)
        loader = DataLoader(ds, batch_size=6, shuffle=False, collate_fn=collate_batch)
        model = _make_chemprop_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        batch = next(iter(loader))

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx=0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss did not decrease with chemprop"
