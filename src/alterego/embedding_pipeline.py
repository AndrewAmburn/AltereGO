#!/usr/bin/env python3

"""Generate the embeddings required for each AltereGO prediction mode."""

import gc
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from .steps.ankh_embedding import AnkhEmbedder
from .steps.esm2_embedding import ESM2Embedder
from .steps.foldseek_3di import structure_to_3di
from .steps.prostt5_embedding import ProstT5Embedder
from .steps.prott5_embedding import ProtT5Embedder


def read_single_fasta(
    fasta_path: Union[str, Path],
) -> Tuple[str, str]:
    """Read exactly one protein sequence from a FASTA file."""

    fasta_path = Path(fasta_path)

    if not fasta_path.is_file():
        raise FileNotFoundError(
            f"Sequence file was not found: {fasta_path}"
        )

    records = []
    identifier = None
    sequence_parts = []

    with fasta_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if identifier is not None:
                    records.append(
                        (
                            identifier,
                            "".join(sequence_parts),
                        )
                    )

                identifier = (
                    line[1:].strip().split()[0]
                    or fasta_path.stem
                )
                sequence_parts = []

            else:
                if identifier is None:
                    identifier = fasta_path.stem

                sequence_parts.append(line)

    if identifier is not None:
        records.append(
            (
                identifier,
                "".join(sequence_parts),
            )
        )

    if not records:
        raise ValueError(
            f"No protein sequence was found in {fasta_path}."
        )

    if len(records) != 1:
        raise ValueError(
            "The initial AltereGO release accepts one protein per FASTA "
            "file. Use separate runs for multiple proteins."
        )

    protein_id, sequence = records[0]

    sequence = (
        sequence
        .replace(" ", "")
        .replace("\t", "")
        .upper()
    )

    if not sequence:
        raise ValueError(
            "The supplied protein sequence is empty."
        )

    return protein_id, sequence


def release_model(
    embedder,
    device: torch.device,
) -> None:
    """Release a foundation model before loading the next one."""

    if hasattr(embedder, "model"):
        embedder.model = None

    if hasattr(embedder, "tokenizer"):
        embedder.tokenizer = None

    if hasattr(embedder, "batch_converter"):
        embedder.batch_converter = None

    del embedder
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()


def generate_embeddings(
    sequence: Optional[str] = None,
    structure_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    foldseek_executable: str = "foldseek",
) -> Dict[str, np.ndarray]:
    """
    Generate the embeddings required by the supplied input modalities.

    Sequence input produces ESM2-3B, ProtT5, and Ankh embeddings.
    Structure input produces a Foldseek-3Di/ProstT5 embedding.
    """

    if sequence is None and structure_path is None:
        raise ValueError(
            "At least one sequence or structure must be supplied."
        )

    if structure_path is not None:
        structure_path = Path(structure_path).resolve()

        if not structure_path.is_file():
            raise FileNotFoundError(
                f"Structure file was not found: {structure_path}"
            )

        if shutil.which(foldseek_executable) is None:
            raise FileNotFoundError(
                "Foldseek was not found. Install Foldseek, add it to PATH, "
                "or provide its executable using --foldseek."
            )

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)
    embeddings = {}

    if sequence is not None:
        print("Generating ESM2-3B embedding...", flush=True)

        embedder = ESM2Embedder(
            device=device
        )

        embeddings["esm2"] = (
            embedder.embed(sequence)
            .reshape(1, -1)
        )

        release_model(
            embedder,
            device,
        )

        print("Generating ProtT5 embedding...", flush=True)

        embedder = ProtT5Embedder(
            device=device
        )

        embeddings["prott5"] = (
            embedder.embed(sequence)
            .reshape(1, -1)
        )

        release_model(
            embedder,
            device,
        )

        print("Generating Ankh-Large embedding...", flush=True)

        embedder = AnkhEmbedder(
            device=device
        )

        embeddings["ankh"] = (
            embedder.embed(sequence)
            .reshape(1, -1)
        )

        release_model(
            embedder,
            device,
        )

    if structure_path is not None:
        print(
            "Converting structure to Foldseek 3Di tokens...",
            flush=True,
        )

        structural_sequence = structure_to_3di(
            structure_path=structure_path,
            foldseek_executable=foldseek_executable,
        )

        print("Generating ProstT5-3Di embedding...", flush=True)

        embedder = ProstT5Embedder(
            device=device
        )

        embeddings["prostt5_3di"] = (
            embedder.embed(
                structural_sequence
            )
            .reshape(1, -1)
        )

        release_model(
            embedder,
            device,
        )

    return embeddings
