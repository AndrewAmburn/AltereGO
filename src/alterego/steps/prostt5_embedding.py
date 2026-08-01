#!/usr/bin/env python3

"""Generate ProstT5 embeddings from Foldseek 3Di strings."""

import warnings
from typing import Optional, Union

import numpy as np
import torch


MODEL_NAME = "Rostlab/ProstT5"
EMBEDDING_DIMENSION = 1024
MAX_TOKEN_LENGTH = 1024
DIRECTION_PREFIX = "<fold2AA>"


def preprocess_3di(
    structural_sequence: str,
) -> str:
    """Format a Foldseek 3Di string as used during model development."""

    structural_sequence = (
        str(structural_sequence)
        .strip()
        .lower()
    )

    if not structural_sequence:
        raise ValueError(
            "The Foldseek 3Di sequence is empty."
        )

    return (
        DIRECTION_PREFIX
        + " "
        + " ".join(
            list(structural_sequence)
        )
    )


class ProstT5Embedder:
    """Lazy-loaded ProstT5 3Di embedding generator."""

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        if dtype is None:
            dtype = (
                torch.float16
                if self.device.type == "cuda"
                else torch.float32
            )

        if (
            self.device.type == "cpu"
            and dtype == torch.float16
        ):
            raise ValueError(
                "ProstT5 float16 inference requires CUDA. "
                "Use float32 for CPU inference."
            )

        self.dtype = dtype
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load ProstT5 when first needed."""

        if self.model is not None:
            return

        try:
            from transformers import (
                T5EncoderModel,
                T5Tokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "ProstT5 requires transformers and sentencepiece."
            ) from exc

        self.tokenizer = T5Tokenizer.from_pretrained(
            MODEL_NAME,
            do_lower_case=False,
        )

        self.model = T5EncoderModel.from_pretrained(
            MODEL_NAME,
        )

        self.model.to(self.device)

        if self.dtype == torch.float16:
            self.model.half()
        else:
            self.model.float()

        self.model.eval()

    @torch.inference_mode()
    def embed(
        self,
        structural_sequence: str,
    ) -> np.ndarray:
        """Generate one 1,024-dimensional ProstT5-3Di embedding."""

        self.load()

        structural_sequence = (
            str(structural_sequence)
            .strip()
            .lower()
        )

        if len(structural_sequence) + 2 > MAX_TOKEN_LENGTH:
            warnings.warn(
                "The ProstT5 input exceeds the training-time maximum "
                "of 1,024 tokens and will be truncated.",
                RuntimeWarning,
                stacklevel=2,
            )

        processed = preprocess_3di(
            structural_sequence
        )

        tokens = self.tokenizer(
            [processed],
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
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

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state[0]
        mask = attention_mask[0].bool()

        embedding = (
            hidden[mask]
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
                "Unexpected ProstT5 embedding shape: "
                f"{embedding.shape}; expected "
                f"({EMBEDDING_DIMENSION},)."
            )

        if not np.isfinite(
            embedding
        ).all():
            raise RuntimeError(
                "ProstT5 produced a non-finite embedding."
            )

        return embedding
