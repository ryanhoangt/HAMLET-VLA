import torch
import torch.nn as nn
import torch.nn.functional as F


class MomentProjectionHead(nn.Module):
    """Maps mean-pooled moment tokens to an L2-normalized contrastive embedding.

    Architecture follows SimCLR: Linear → ReLU → Linear → L2-norm.
    The projection head is discarded after TCL pretraining; only
    moment_tokens weights are kept for subsequent HAMLET training.
    """

    def __init__(self, d_model: int = 1536, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, m_prime: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m_prime: (B, n_m, d_model)  — VLM-contextualized moment token features
        Returns:
            z: (B, proj_dim), L2-normalized
        """
        pooled = m_prime.mean(dim=1)                    # (B, d_model)
        return F.normalize(self.net(pooled), dim=-1)    # (B, proj_dim)


def tcl_loss(
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Time-Contrastive Loss with one hard negative per sample.

    Positive pair:  same observation, different augmentation (z_anchor vs z_pos).
    Hard negative:  different timestep in the same trajectory (z_neg).

    Loss per sample = -log( exp(sim(a,p)/τ) / (exp(sim(a,p)/τ) + exp(sim(a,n)/τ)) )
                    = -sim(a,p)/τ + logsumexp([sim(a,p)/τ, sim(a,n)/τ])

    Args:
        z_anchor: (B, proj_dim), L2-normalized
        z_pos:    (B, proj_dim), L2-normalized
        z_neg:    (B, proj_dim), L2-normalized
        temperature: scaling factor τ
    Returns:
        scalar loss
    """
    sim_pos = (z_anchor * z_pos).sum(dim=-1) / temperature  # (B,)
    sim_neg = (z_anchor * z_neg).sum(dim=-1) / temperature  # (B,)

    logits = torch.stack([sim_pos, sim_neg], dim=1)          # (B, 2)
    loss = -sim_pos + torch.logsumexp(logits, dim=1)         # (B,)
    return loss.mean()
