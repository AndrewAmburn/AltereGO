#!/usr/bin/env python3

"""Generate Ankh-Large protein-level embeddings for AltereGO."""

import warnings
from typing import List, Optional, Union

import numpy as np
import torch


MODEL_NAME = "ankh-large"
EMBEDDING_DIMENSION = 1536
MAX_TOKEN_LENGTH = 1024


def preprocess_sequence(
    sequence: str,
) -> List[str]:
    """
    Reproduce the Ankh preprocessing used for the training embeddings.

    The tokenizer receives each amino-acid character as a separate word.
    """

    sequence = str(sequence).strip().upper()

    if not sequence:
        raise ValueError(
            "The supplied protein sequence is empty."
        )

    return list(sequence)


class AnkhEmbedder:
    """Lazy-loaded Ankh-Large protein embedding generator."""

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load Ankh-Large when first needed."""

        if self.model is not None:
            return

        try:
            import ankh
        except ImportError as exc:
            raise ImportError(
                "Ankh embedding generation requires the ankh package."
            ) from exc

        model, tokenizer = (
            ankh.load_large_model()
        )

        # The production workflow retained float32 because float16
        # produced non-finite embeddings.
        model = model.float()
        model.eval()
        model.to(self.device)

        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def embed(
        self,
        sequence: str,
    ) -> np.ndarray:
        """
        Generate one 1,536-dimensional Ankh-Large embedding.

        This preserves the original float32 inference, 1,024-token
        truncation, attention-mask pooling, and special-token handling.
        """

        self.load()

        raw_sequence = str(
            sequence
        ).strip().upper()

        if len(raw_sequence) + 1 > MAX_TOKEN_LENGTH:
            warnings.warn(
                "Ankh input exceeds the training-time maximum of "
                "1,024 tokens and will be truncated to reproduce the "
                "original AltereGO embedding procedure.",
                RuntimeWarning,
                stacklevel=2,
            )

        processed_sequence = preprocess_sequence(
            raw_sequence
        )

        tokens = self.tokenizer(
            [processed_sequence],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            is_split_into_words=True,
            return_tensors="pt",
        )

        input_ids = tokens[
            "input_ids"
        ].to(
            self.device
        )

        attention_mask = tokens[
            "attention_mask"
        ].to(
            self.device
        )

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if hasattr(
            model_outputs,
            "last_hidden_state",
        ):
            hidden = (
                model_outputs.last_hidden_state
            )

        elif isinstance(
            model_outputs,
            (tuple, list),
        ):
            hidden = model_outputs[0]

        else:
            hidden = model_outputs

        mask = attention_mask[0].bool()

        embedding = (
            hidden[0][mask]
            .mean(dim=0)
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        if embedding.shape != (
            EMBEDDING_DIMENSION,
        ):
            raise RuntimeError(
                "Unexpected Ankh embedding shape: "
                f"{embedding.shape}; expected "
                f"({EMBEDDING_DIMENSION},)."
            )

        if not np.isfinite(
            embedding
        ).all():
            raise RuntimeError(
                "Ankh produced a non-finite protein embedding."
            )

        return embedding
