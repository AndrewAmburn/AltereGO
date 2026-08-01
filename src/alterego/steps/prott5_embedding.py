#!/usr/bin/env python3

"""Generate ProtT5 protein-level embeddings for AltereGO."""

import re
import warnings
from typing import Optional, Union

import numpy as np
import torch


MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
EMBEDDING_DIMENSION = 1024
MAX_TOKEN_LENGTH = 1024


def preprocess_sequence(sequence: str) -> str:
    """
    Reproduce the ProtT5 preprocessing used for the training embeddings.

    U, Z, O, and B are replaced with X, and residues are separated by
    spaces before tokenization.
    """

    sequence = str(sequence).strip().upper()

    if not sequence:
        raise ValueError(
            "The supplied protein sequence is empty."
        )

    sequence = re.sub(
        r"[UZOB]",
        "X",
        sequence,
    )

    return " ".join(
        list(sequence)
    )


class ProtT5Embedder:
    """Lazy-loaded ProtT5 protein embedding generator."""

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
                "ProtT5 float16 inference requires a CUDA device. "
                "Use float32 for CPU inference."
            )

        self.dtype = dtype
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load the pretrained ProtT5 encoder when first needed."""

        if self.model is not None:
            return

        try:
            from transformers import (
                T5EncoderModel,
                T5Tokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "ProtT5 embedding generation requires transformers "
                "and sentencepiece."
            ) from exc

        self.tokenizer = T5Tokenizer.from_pretrained(
            MODEL_NAME,
            do_lower_case=False,
        )

        self.model = T5EncoderModel.from_pretrained(
            MODEL_NAME,
            dtype=self.dtype,
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed(
        self,
        sequence: str,
    ) -> np.ndarray:
        """
        Generate one 1,024-dimensional protein-level ProtT5 embedding.

        Tokenization, truncation, masking, and mean pooling reproduce the
        production script used to construct the training embeddings.
        """

        self.load()

        raw_sequence = str(
            sequence
        ).strip().upper()

        if len(raw_sequence) + 1 > MAX_TOKEN_LENGTH:
            warnings.warn(
                "ProtT5 input exceeds the training-time maximum of "
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
            padding="longest",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(
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
                "Unexpected ProtT5 embedding shape: "
                f"{embedding.shape}; expected "
                f"({EMBEDDING_DIMENSION},)."
            )

        return embedding
