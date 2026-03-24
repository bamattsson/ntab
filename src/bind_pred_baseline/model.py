"""Binding prediction model (Lightning module)."""

import torch
import torch.nn as nn
import lightning as L

from bind_pred_baseline.constants import FP_SIZE, MIN_ASSAY_SIZE, N_MOL_PROP_FEATURES
from bind_pred_baseline.model_utils import pearson_r_per_assay



class AffinityModel(L.LightningModule):
    """MLP baseline for protein-ligand affinity prediction.

    Architecture:
        fp_encoder:   Linear(FP_SIZE, hidden_dim) → BatchNorm → GELU
        concatenate:  [fp_enc | mol_props | target_embedding]
        head (×2):    Linear(hidden_dim) → BatchNorm → GELU
                      Linear(hidden_dim, 1)
        output:       head(combined) + target_bias + assay_type_bias

    Args:
        n_targets: Number of unique training targets (size of embedding table).
        n_standard_types: Number of assay types (IC50, Ki, Kd).
        hidden_dim: Width of hidden layers.
        target_embed_dim: Dimensionality of the target embedding.
        min_assay_size: Minimum compounds per assay for Pearson r metric.
        use_fps: Whether to include ECFP4 fingerprint as input.
        use_mol_props: Whether to include physicochemical properties as input.
        lr: Learning rate.
    """

    def __init__(
        self,
        n_targets: int,
        n_standard_types: int = 3,
        hidden_dim: int = 2048,
        target_embed_dim: int = 256,
        min_assay_size: int = MIN_ASSAY_SIZE,
        use_fps: bool = True,
        use_mol_props: bool = True,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.min_assay_size = min_assay_size
        self.lr = lr
        self.use_fps = use_fps
        self.use_mol_props = use_mol_props

        self._val_preds: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._val_assay_ids: list[str] = []
        self._test_preds: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []
        self._test_assay_ids: list[str] = []
        head_input_dim = target_embed_dim
        if use_fps:
            self.fp_encoder = nn.Sequential(
                nn.Linear(FP_SIZE, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
            head_input_dim += hidden_dim
        if use_mol_props:
            head_input_dim += N_MOL_PROP_FEATURES
        self.target_embedding = nn.Embedding(n_targets, target_embed_dim)
        self.target_bias = nn.Embedding(n_targets, 1)
        self.assay_type_bias = nn.Embedding(n_standard_types, 1)
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._loss = nn.MSELoss()

    def forward(self, fps: torch.Tensor, mol_props: torch.Tensor, target_idx: torch.Tensor, standard_type_idx: torch.Tensor) -> torch.Tensor:
        """Return predicted pchembl values, shape (batch, 1)."""
        tensors_to_head = []
        if self.use_fps:
            fp_enc = self.fp_encoder(torch.log1p(fps))
            tensors_to_head.append(fp_enc)
        t_emb = self.target_embedding(target_idx)
        tensors_to_head.append(t_emb)
        if self.use_mol_props:
            tensors_to_head.append(mol_props)
        combined = torch.cat(tensors_to_head, dim=1)
        return self.head(combined) + self.target_bias(target_idx) + self.assay_type_bias(standard_type_idx)

    def _shared_step(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], int]:
        fps, mol_props, target_idx, standard_type_idx, labels, assay_ids = batch
        preds = self(fps, mol_props, target_idx, standard_type_idx).squeeze(1)
        loss = self._loss(preds, labels)
        return loss, preds, labels, assay_ids, labels.size(0)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss, _, _, _, batch_size = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        loss, preds, labels, assay_ids, batch_size = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self._val_preds.append(preds.detach().cpu())
        self._val_labels.append(labels.detach().cpu())
        self._val_assay_ids.extend(assay_ids)

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        r = pearson_r_per_assay(
            torch.cat(self._val_preds),
            torch.cat(self._val_labels),
            self._val_assay_ids,
            self.min_assay_size,
        )
        if not torch.isnan(r):
            self.log("val_pearson_r", r, prog_bar=True)
        self._val_preds.clear()
        self._val_labels.clear()
        self._val_assay_ids.clear()

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        loss, preds, labels, assay_ids, batch_size = self._shared_step(batch)
        self.log("test_loss", loss, on_epoch=True, batch_size=batch_size)
        self._test_preds.append(preds.detach().cpu())
        self._test_labels.append(labels.detach().cpu())
        self._test_assay_ids.extend(assay_ids)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        r = pearson_r_per_assay(
            torch.cat(self._test_preds),
            torch.cat(self._test_labels),
            self._test_assay_ids,
            self.min_assay_size,
        )
        if not torch.isnan(r):
            self.log("test_pearson_r", r)
        self._test_preds.clear()
        self._test_labels.clear()
        self._test_assay_ids.clear()

    def predict_step(self, batch: tuple, batch_idx: int) -> dict:
        fps, mol_props, target_idx, standard_type_idx, names, uniprot_ids = batch
        preds = self(fps, mol_props, target_idx, standard_type_idx).squeeze(1)
        return {
            "ligand_name": list(names),
            "uniprot_id": list(uniprot_ids),
            "pred_pchembl": preds.detach().cpu(),
        }

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if "bias" in name or "bn" in name or "assay_type_bias" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 1e-5}, {"params": no_decay, "weight_decay": 0.0}],
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
