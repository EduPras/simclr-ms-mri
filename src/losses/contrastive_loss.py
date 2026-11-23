"""Patient-Masked NT-Xent contrastive loss and Hybrid Longitudinal SSL loss."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatientMaskedNTXentLoss(nn.Module):
    """Normalized Temperature-Scaled Cross-Entropy Loss with false-negative patient masking.

    For a batch of B aligned longitudinal patch pairs (u_i, v_i), forms 2B embeddings.
    Positive pair for u_i is v_i (and vice versa). Any off-diagonal pair (m, n) that
    belongs to the SAME patient_id is masked out of the negative denominator so the model
    is never penalized for having consistent patient-level representations.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        proj_t1: torch.Tensor,
        proj_t2: torch.Tensor,
        patient_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = proj_t1.shape[0]
        device = proj_t1.device

        u = F.normalize(proj_t1, dim=-1)
        v = F.normalize(proj_t2, dim=-1)
        z = torch.cat([u, v], dim=0)  # [2B, D]

        # Cosine similarity matrix scaled by temperature: [2B, 2B]
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Positive partner indices: i <-> i + B
        pos_indices = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=device),
                torch.arange(0, batch_size, device=device),
            ],
            dim=0,
        )

        # Mask out self-similarity on diagonal
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        invalid_mask = diag_mask.clone()

        # Mask out same-patient false negatives (while keeping the true positive partner unmasked!)
        if patient_labels is not None:
            labels_2b = torch.cat([patient_labels, patient_labels], dim=0)  # [2B]
            same_patient = labels_2b.unsqueeze(0) == labels_2b.unsqueeze(1)  # [2B, 2B]

            pos_pair_mask = torch.zeros_like(same_patient, dtype=torch.bool)
            row_idx = torch.arange(2 * batch_size, device=device)
            pos_pair_mask[row_idx, pos_indices] = True

            # Only mask same-patient entries if at least 1 valid negative remains per row
            candidate_invalid = diag_mask | (same_patient & (~pos_pair_mask))
            valid_neg_counts = (~candidate_invalid).sum(dim=1) - 1  # subtract positive partner
            rows_with_negs = valid_neg_counts > 0
            invalid_mask = torch.where(
                rows_with_negs.unsqueeze(1),
                candidate_invalid,
                diag_mask,
            )

        logits = sim_matrix.masked_fill(invalid_mask, -1e4)
        loss = F.cross_entropy(logits, pos_indices)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            top1_acc = (preds == pos_indices).float().mean()

        return {
            "nt_xent_loss": loss,
            "top1_acc": top1_acc,
        }


class HybridLongitudinalSSLLoss(nn.Module):
    """Combines Patient-Masked NT-Xent loss with auxiliary Δt temporal difference regression."""

    def __init__(
        self,
        temperature: float = 0.1,
        lambda_delta_t: float = 0.25,
    ) -> None:
        super().__init__()
        self.nt_xent = PatientMaskedNTXentLoss(temperature=temperature)
        self.lambda_delta_t = float(lambda_delta_t)
        self.delta_t_loss_fn = nn.SmoothL1Loss(beta=0.1)

    def forward(
        self,
        proj_t1: torch.Tensor,
        proj_t2: torch.Tensor,
        pred_delta_t: torch.Tensor,
        target_delta_t: torch.Tensor,
        patient_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        contrastive_out = self.nt_xent(proj_t1, proj_t2, patient_labels=patient_labels)
        dt_loss = self.delta_t_loss_fn(pred_delta_t.view(-1), target_delta_t.view(-1))
        total_loss = contrastive_out["nt_xent_loss"] + self.lambda_delta_t * dt_loss

        return {
            "loss": total_loss,
            "nt_xent_loss": contrastive_out["nt_xent_loss"],
            "delta_t_loss": dt_loss,
            "top1_acc": contrastive_out["top1_acc"],
        }
