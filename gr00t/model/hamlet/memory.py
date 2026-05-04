import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    """Memory consolidation module from HAMLET.

    Stacks the T most recent moment-token sets into a flat sequence M' of
    shape (T*n_m, d), runs causal self-attention across the full L = T*n_m
    positions, and returns the last n_m rows as the history-augmented
    representation m̃'_t.  m̃'_t is then concatenated to the VLM features h_t
    before the action head:

        [a_t, ..., a_{t+k-1}] = A_ψ([h_t ; m̃'_t], s_t)

    Input:  moment_history  (B, T, n_m, d_model)
    Output: m_tilde         (B, n_m, d_model)

    The history uses stride k between entries (k = action chunk size), so
    the T entries span T*k past timesteps.  Building the history tensor and
    managing the rolling buffer is the caller's responsibility.
    """

    def __init__(
        self,
        d_model: int = 1536,
        n_moment_tokens: int = 4,
        n_heads: int = 8,
        n_layers: int = 2,
        max_history: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_moment_tokens = n_moment_tokens
        self.max_history = max_history

        # Positional embeddings over the full flattened sequence (T * n_m positions)
        max_seq_len = max_history * n_moment_tokens
        self.pos_embedding = nn.Parameter(
            0.02 * torch.randn(max_seq_len, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self._init_weights()

    def _init_weights(self):
        # Zero-init the last layer of each encoder layer's FF so the module
        # starts as a near-identity (good for fine-tuning on top of pretrained VLA)
        for layer in self.transformer.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)

    def forward(self, moment_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            moment_history: (B, T, n_m, d_model)
                T ≤ max_history; pad with zeros at the front for cold-start.
        Returns:
            m_tilde: (B, n_m, d_model)
                History-augmented moment tokens for the current timestep.
                Concatenate to backbone_features before passing to action head.
        """
        B, T, n_m, d = moment_history.shape
        assert T <= self.max_history, (
            f"History length {T} exceeds max_history {self.max_history}"
        )
        assert n_m == self.n_moment_tokens, (
            f"Expected {self.n_moment_tokens} moment tokens, got {n_m}"
        )

        L = T * n_m

        # Flatten: (B, T, n_m, d) → (B, T*n_m, d)
        x = moment_history.reshape(B, L, d)

        # Add positional encoding
        x = x + self.pos_embedding[:L].unsqueeze(0)

        # Causal mask: position i cannot attend to j > i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=x.device, dtype=x.dtype
        )

        x = self.transformer(x, mask=causal_mask, is_causal=True)  # (B, L, d)

        # Last n_m rows = current timestep's contextualized moment tokens
        m_tilde = x[:, -n_m:, :]  # (B, n_m, d_model)
        return m_tilde
