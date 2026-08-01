#!/usr/bin/env python3

"""Residual MLP predictors used by AltereGO."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file


def l2_normalize_np(
    x: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply the same per-protein L2 normalization used during training.

    Parameters
    ----------
    x
        One protein embedding with shape (features,) or a batch with
        shape (proteins, features).
    eps
        Minimum permitted normalization denominator.

    Returns
    -------
    np.ndarray
        L2-normalized float32 embeddings with shape
        (proteins, features).
    """

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.ndim != 2:
        raise ValueError(
            "Embedding input must have shape (features,) or "
            "(proteins, features)."
        )

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        eps,
    )

    return x / norms


class ResidualFFNBlock(nn.Module):
    """Residual feedforward block from the trained AltereGO models."""

    def __init__(
        self,
        dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = x + residual

        return x


class ResidualMLP(nn.Module):
    """
    Residual multilayer perceptron used for standalone GO prediction.

    The module structure and names match the original training code so
    that the released Safetensors checkpoints load without remapping.
    """

    def __init__(
        self,
        input_dim: int,
        n_labels: int,
        hidden_dim: int = 2048,
        n_blocks: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [
                ResidualFFNBlock(
                    hidden_dim,
                    dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.output = nn.Linear(
            hidden_dim,
            n_labels,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.output(x)

        return logits


def load_residual_mlp(
    weights_path: Union[str, Path],
    input_dim: int,
    n_labels: int = 8446,
    device: Union[str, torch.device] = "cpu",
) -> ResidualMLP:
    """
    Reconstruct a trained AltereGO predictor from Safetensors weights.
    """

    weights_path = Path(weights_path)
    device = torch.device(device)

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights were not found: {weights_path}"
        )

    model = ResidualMLP(
        input_dim=input_dim,
        n_labels=n_labels,
        hidden_dim=2048,
        n_blocks=3,
        dropout=0.2,
    )

    state_dict = load_file(
        str(weights_path),
        device="cpu",
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


@torch.inference_mode()
def predict_probabilities(
    model: ResidualMLP,
    embeddings: np.ndarray,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Generate unpropagated GO probabilities from protein embeddings.

    This function reproduces the original inference procedure:

        L2 normalization -> residual MLP -> sigmoid

    GO hierarchy propagation is performed separately after prediction.
    """

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    device = torch.device(device)

    embeddings = l2_normalize_np(
        embeddings
    )

    expected_input_dim = model.input_proj[0].in_features

    if embeddings.shape[1] != expected_input_dim:
        raise ValueError(
            f"The model expects {expected_input_dim} embedding features, "
            f"but received {embeddings.shape[1]}."
        )

    model.to(device)
    model.eval()

    predictions = []

    for start in range(
        0,
        embeddings.shape[0],
        batch_size,
    ):
        stop = start + batch_size

        xb = torch.from_numpy(
            embeddings[start:stop]
        ).to(device)

        probabilities = torch.sigmoid(
            model(xb)
        )

        predictions.append(
            probabilities.cpu().numpy()
        )

    return np.vstack(
        predictions
    ).astype(
        np.float32,
        copy=False,
    )
