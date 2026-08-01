#!/usr/bin/env python3

"""Generate ESM2-3B protein-level embeddings for AltereGO."""

import gc
from typing import List, Optional, Union

import numpy as np
import torch


MODEL_NAME = "esm2_t36_3B_UR50D"
REPRESENTATION_LAYER = 36
EMBEDDING_DIMENSION = 2560
MAX_RESIDUES_PER_CHUNK = 1022

VALID_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWY"
)


def clean_sequence(sequence: str) -> str:
    """
    Reproduce the sequence cleaning used to construct the training embeddings.

    Characters other than the 20 canonical amino acids are removed.
    """

    sequence = str(sequence).upper().strip()

    cleaned = "".join(
        amino_acid
        for amino_acid in sequence
        if amino_acid in VALID_AMINO_ACIDS
    )

    if not cleaned:
        raise ValueError(
            "The sequence contains no canonical amino-acid residues "
            "after cleaning."
        )

    return cleaned


def split_sequence(
    sequence: str,
    max_length: int = MAX_RESIDUES_PER_CHUNK,
) -> List[str]:
    """Split a cleaned sequence into nonoverlapping residue chunks."""

    if max_length < 1:
        raise ValueError(
            "max_length must be at least 1."
        )

    return [
        sequence[start:start + max_length]
        for start in range(0, len(sequence), max_length)
    ]


class ESM2Embedder:
    """Lazy-loaded ESM2-3B protein embedding generator."""

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
        self.batch_converter = None

    def load(self) -> None:
        """Load the pretrained ESM2-3B model when first needed."""

        if self.model is not None:
            return

        try:
            import esm
        except ImportError as exc:
            raise ImportError(
                "ESM2 embedding generation requires the fair-esm package. "
                "Install it with: pip install fair-esm"
            ) from exc

        model, alphabet = (
            esm.pretrained.esm2_t36_3B_UR50D()
        )

        model.eval()
        model.to(self.device)

        self.model = model
        self.batch_converter = (
            alphabet.get_batch_converter()
        )

    @torch.inference_mode()
    def _embed_chunk(
        self,
        sequence_chunk: str,
    ) -> np.ndarray:
        data = [
            ("protein", sequence_chunk)
        ]

        _, _, tokens = self.batch_converter(
            data
        )

        tokens = tokens.to(
            self.device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            "cuda",
            enabled=(self.device.type == "cuda"),
        ):
            results = self.model(
                tokens,
                repr_layers=[
                    REPRESENTATION_LAYER
                ],
                return_contacts=False,
            )

        representations = results[
            "representations"
        ][REPRESENTATION_LAYER]

        embedding = (
            representations[
                0,
                1:len(sequence_chunk) + 1,
            ]
            .mean(dim=0)
            .detach()
            .cpu()
            .float()
            .numpy()
        )

        del tokens
        del results
        del representations

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return embedding

    def embed(
        self,
        sequence: str,
    ) -> np.ndarray:
        """
        Generate one 2,560-dimensional protein-level ESM2 embedding.

        Long proteins are divided into nonoverlapping 1,022-residue chunks.
        Residues are mean-pooled within each chunk, after which chunk
        embeddings are averaged as in the original embedding workflow.
        """

        self.load()

        sequence = clean_sequence(
            sequence
        )

        chunks = split_sequence(
            sequence
        )

        chunk_embeddings = []

        for chunk in chunks:
            chunk_embeddings.append(
                self._embed_chunk(chunk)
            )

            gc.collect()

        embedding = np.mean(
            np.stack(
                chunk_embeddings,
                axis=0,
            ),
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        if embedding.shape != (
            EMBEDDING_DIMENSION,
        ):
            raise RuntimeError(
                "Unexpected ESM2 embedding shape: "
                f"{embedding.shape}; expected "
                f"({EMBEDDING_DIMENSION},)."
            )

        return embedding
