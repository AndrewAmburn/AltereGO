#!/usr/bin/env python3

"""Core AltereGO prediction pipeline."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from .fusion import fuse_probabilities
from .model import (
    load_residual_mlp,
    predict_probabilities,
)
from .ontology import (
    load_go_parents,
    load_label_space,
    propagate_scores,
)


class AltereGOPredictor:
    """
    Run trained AltereGO predictors from precomputed protein embeddings.

    Supported modes
    ---------------
    sequence_only
        ESM2-3B + ProtT5 + Ankh.

    structure_only
        ProstT5-3Di alone.

    sequence_and_structure
        ESM2-3B + ProtT5 + Ankh + ProstT5-3Di.
    """

    def __init__(
        self,
        release_directory: Union[str, Path],
        device: Optional[str] = None,
    ):
        self.release_directory = Path(
            release_directory
        ).resolve()

        self.model_directory = (
            self.release_directory / "model"
        )

        self.resource_directory = (
            self.release_directory / "resources"
        )

        config_path = (
            self.model_directory
            / "prediction_modes.json"
        )

        with config_path.open() as handle:
            self.config = json.load(handle)

        self.label_space = load_label_space(
            self.resource_directory
            / "go_label_space.csv"
        )

        self.parents = load_go_parents(
            self.resource_directory
            / "go-basic.obo",
            self.label_space,
        )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

    @staticmethod
    def determine_mode(
        sequence_supplied: bool,
        structure_supplied: bool,
    ) -> str:
        """Determine the validated prediction mode from supplied inputs."""

        if sequence_supplied and structure_supplied:
            return "sequence_and_structure"

        if sequence_supplied:
            return "sequence_only"

        if structure_supplied:
            return "structure_only"

        raise ValueError(
            "At least one protein sequence or structure must be supplied."
        )

    def predict_from_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        mode: str,
        batch_size: int = 1024,
    ) -> Dict[str, object]:
        """
        Generate hierarchy-consistent AltereGO predictions.

        Each standalone model output is propagated before fusion to
        reproduce the original model-development workflow.
        """

        if mode not in self.config["modes"]:
            raise ValueError(
                f"Unsupported prediction mode: {mode}"
            )

        mode_config = self.config["modes"][mode]
        required_models = mode_config["models"]

        missing = [
            model_name
            for model_name in required_models
            if model_name not in embeddings
        ]

        if missing:
            raise ValueError(
                "Missing required embeddings: "
                + ", ".join(missing)
            )

        unexpected = sorted(
            set(embeddings) - set(required_models)
        )

        if unexpected:
            raise ValueError(
                "Unexpected embeddings for prediction mode "
                f"{mode}: {', '.join(unexpected)}"
            )

        standalone_scores = {}
        protein_count = None

        for model_name in required_models:
            model_config = self.config["models"][model_name]

            model_embeddings = np.asarray(
                embeddings[model_name],
                dtype=np.float32,
            )

            if model_embeddings.ndim == 1:
                model_embeddings = model_embeddings.reshape(1, -1)

            if model_embeddings.ndim != 2:
                raise ValueError(
                    f"{model_name} embeddings must be two-dimensional."
                )

            if protein_count is None:
                protein_count = model_embeddings.shape[0]

            elif model_embeddings.shape[0] != protein_count:
                raise ValueError(
                    "All embedding arrays must contain the same number "
                    "of proteins in the same order."
                )

            expected_dimension = model_config[
                "input_dimension"
            ]

            if model_embeddings.shape[1] != expected_dimension:
                raise ValueError(
                    f"{model_name} expects {expected_dimension} features, "
                    f"but received {model_embeddings.shape[1]}."
                )

            weights_path = (
                self.model_directory
                / model_config["weights_file"]
            )

            model = load_residual_mlp(
                weights_path=weights_path,
                input_dim=expected_dimension,
                n_labels=len(self.label_space),
                device=self.device,
            )

            scores = predict_probabilities(
                model=model,
                embeddings=model_embeddings,
                device=self.device,
                batch_size=batch_size,
            )

            standalone_scores[model_name] = propagate_scores(
                scores,
                self.label_space,
                self.parents,
            )

            del model

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        fused_scores = fuse_probabilities(
            probabilities=standalone_scores,
            weights=mode_config["weights"],
        )

        fused_scores = propagate_scores(
            fused_scores,
            self.label_space,
            self.parents,
        )

        threshold = float(
            mode_config["decision_threshold"]
        )

        binary_predictions = (
            fused_scores >= threshold
        ).astype(np.uint8)

        return {
            "mode": mode,
            "threshold": threshold,
            "scores": fused_scores,
            "binary_predictions": binary_predictions,
            "standalone_scores": standalone_scores,
            "label_space": self.label_space,
        }
