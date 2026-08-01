#!/usr/bin/env python3

"""Fixed score-level fusion used by AltereGO."""

from typing import Dict

import numpy as np


def fuse_probabilities(
    probabilities: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """Combine model probabilities using fixed linear-fusion weights."""

    if set(probabilities) != set(weights):
        raise ValueError(
            "The available model predictions do not match the configured "
            "fusion weights."
        )

    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(
            "Fusion weights must sum to 1."
        )

    fused = None
    expected_shape = None

    for model_name, weight in weights.items():
        scores = np.asarray(
            probabilities[model_name],
            dtype=np.float32,
        )

        if scores.ndim == 1:
            scores = scores.reshape(1, -1)

        if scores.ndim != 2:
            raise ValueError(
                f"{model_name} predictions must be a two-dimensional array."
            )

        if expected_shape is None:
            expected_shape = scores.shape
            fused = np.zeros(
                expected_shape,
                dtype=np.float32,
            )

        elif scores.shape != expected_shape:
            raise ValueError(
                "All model prediction arrays must have identical shapes."
            )

        fused += np.float32(weight) * scores

    return fused
