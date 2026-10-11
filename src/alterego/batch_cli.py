#!/usr/bin/env python3

"""Batch sequence-only inference for AltereGO."""

import os

# AltereGO uses PyTorch exclusively.
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .embedding_pipeline import release_model
from .pipeline import AltereGOPredictor
from .steps.ankh_embedding import AnkhEmbedder
from .steps.esm2_embedding import ESM2Embedder
from .steps.prott5_embedding import ProtT5Embedder


FastaRecord = Tuple[str, str]


def read_fasta(fasta_path: Path) -> List[FastaRecord]:
    """Read one or more protein records while preserving FASTA order."""

    fasta_path = Path(fasta_path)

    if not fasta_path.is_file():
        raise FileNotFoundError(
            f"Sequence file was not found: {fasta_path}"
        )

    records: List[FastaRecord] = []
    identifier = None
    sequence_parts: List[str] = []

    def store_record() -> None:
        nonlocal identifier
        nonlocal sequence_parts

        if identifier is None:
            return

        sequence = (
            "".join(sequence_parts)
            .replace(" ", "")
            .replace("\t", "")
            .upper()
        )

        if not sequence:
            raise ValueError(
                f"FASTA record {identifier!r} has an empty sequence."
            )

        records.append((identifier, sequence))

        identifier = None
        sequence_parts = []

    with fasta_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                store_record()

                header = line[1:].strip()

                identifier = (
                    header.split()[0]
                    if header
                    else fasta_path.stem
                )

            else:
                if identifier is None:
                    identifier = fasta_path.stem

                sequence_parts.append(line)

    store_record()

    if not records:
        raise ValueError(
            f"No protein sequences were found in {fasta_path}."
        )

    identifiers = [
        protein_id
        for protein_id, _ in records
    ]

    duplicate_ids = sorted(
        {
            protein_id
            for protein_id in identifiers
            if identifiers.count(protein_id) > 1
        }
    )

    if duplicate_ids:
        raise ValueError(
            "Duplicate FASTA identifiers are not allowed: "
            + ", ".join(duplicate_ids)
        )

    return records


def embed_records(
    records: Sequence[FastaRecord],
    embedder_class,
    model_name: str,
    device: torch.device,
    progress_every: int,
) -> np.ndarray:
    """
    Embed all sequences while loading the foundation model only once.

    Sequences are processed individually to preserve the validated
    single-protein embedding procedure and minimize peak GPU memory.
    """

    print(
        f"Loading {model_name} and embedding "
        f"{len(records):,} sequences...",
        flush=True,
    )

    embedder = embedder_class(device=device)
    embeddings = []

    try:
        for index, (protein_id, sequence) in enumerate(
            records,
            start=1,
        ):
            try:
                embedding = embedder.embed(sequence)
            except Exception as exc:
                raise RuntimeError(
                    f"{model_name} embedding failed for "
                    f"{protein_id!r}."
                ) from exc

            embeddings.append(embedding)

            if (
                index == 1
                or index == len(records)
                or index % progress_every == 0
            ):
                print(
                    f"{model_name}: "
                    f"{index:,}/{len(records):,}",
                    flush=True,
                )

    finally:
        release_model(
            embedder,
            device,
        )

    return np.stack(
        embeddings,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )


def generate_batch_embeddings(
    records: Sequence[FastaRecord],
    device: torch.device,
    progress_every: int,
):
    """Generate sequence-only AltereGO embeddings in FASTA order."""

    embeddings = {}

    embeddings["esm2"] = embed_records(
        records=records,
        embedder_class=ESM2Embedder,
        model_name="ESM2-3B",
        device=device,
        progress_every=progress_every,
    )

    embeddings["prott5"] = embed_records(
        records=records,
        embedder_class=ProtT5Embedder,
        model_name="ProtT5",
        device=device,
        progress_every=progress_every,
    )

    embeddings["ankh"] = embed_records(
        records=records,
        embedder_class=AnkhEmbedder,
        model_name="Ankh-Large",
        device=device,
        progress_every=progress_every,
    )

    return embeddings


def select_go_terms(
    requested_go_terms,
    label_space,
):
    """Resolve optional output filtering against the label space."""

    label_to_index = {
        go_id: index
        for index, go_id in enumerate(label_space)
    }

    if requested_go_terms is None:
        selected_terms = list(label_space)

    else:
        selected_terms = list(
            dict.fromkeys(requested_go_terms)
        )

        missing = [
            go_id
            for go_id in selected_terms
            if go_id not in label_to_index
        ]

        if missing:
            raise ValueError(
                "Requested GO terms were not found in the "
                "AltereGO label space: "
                + ", ".join(missing)
            )

    selected_indices = np.asarray(
        [
            label_to_index[go_id]
            for go_id in selected_terms
        ],
        dtype=np.int64,
    )

    return selected_terms, selected_indices


def write_long_tsv(
    output_path: Path,
    predicted_output_path: Path,
    protein_ids,
    scores,
    binary_predictions,
    selected_terms,
    selected_indices,
    label_table,
) -> None:
    """Write ranked long-format output without building one enormous table."""

    label_table = label_table.copy()
    label_table["go_id"] = (
        label_table["go_id"].astype(str)
    )

    metadata_lookup = (
        label_table
        .drop_duplicates("go_id")
        .set_index("go_id")
    )

    label_metadata = pd.DataFrame(
        {
            "go_id": selected_terms,
        }
    )

    for column in ["name", "definition"]:
        if column in metadata_lookup.columns:
            label_metadata[column] = [
                (
                    metadata_lookup.at[go_id, column]
                    if go_id in metadata_lookup.index
                    else ""
                )
                for go_id in selected_terms
            ]

    for protein_index, protein_id in enumerate(protein_ids):
        frame = label_metadata.copy()

        frame.insert(
            0,
            "protein_id",
            protein_id,
        )

        frame["score"] = scores[
            protein_index,
            selected_indices,
        ]

        frame["predicted"] = binary_predictions[
            protein_index,
            selected_indices,
        ].astype(bool)

        frame = frame.sort_values(
            "score",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True)

        frame.insert(
            1,
            "rank",
            np.arange(
                1,
                len(frame) + 1,
                dtype=np.int64,
            ),
        )

        write_mode = (
            "w"
            if protein_index == 0
            else "a"
        )

        write_header = (
            protein_index == 0
        )

        frame.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode=write_mode,
            header=write_header,
        )

        predicted_frame = frame[
            frame["predicted"]
        ]

        predicted_frame.to_csv(
            predicted_output_path,
            sep="\t",
            index=False,
            mode=write_mode,
            header=write_header,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alterego-batch",
        description=(
            "Run sequence-only AltereGO inference on a "
            "multi-record protein FASTA file."
        ),
    )

    parser.add_argument(
        "--sequence",
        type=Path,
        required=True,
        help="FASTA file containing one or more protein sequences.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Long-format TSV output path.",
    )

    parser.add_argument(
        "--go-terms",
        nargs="+",
        help=(
            "Optional GO identifiers to write to the TSV. "
            "The NPZ file always retains the complete label space."
        ),
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Write only the NPZ score matrix. When --go-terms is "
            "provided, the NPZ contains only those GO terms. Long "
            "TSV and JSON outputs are skipped."
        ),
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        help="Inference device.",
    )

    parser.add_argument(
        "--mlp-batch-size",
        type=int,
        default=256,
        help="Batch size for downstream residual MLP inference.",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Report embedding progress every N proteins.",
    )

    parser.add_argument(
        "--release-directory",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--exploratory",
        action="store_true",
        help=(
            "Use an exploratory decision threshold of 0.50 instead of "
            "the validated mode-specific threshold. Continuous scores "
            "and fusion weights are unchanged."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        parser.error(
            "CUDA was requested, but no CUDA device is available."
        )

    if args.mlp_batch_size < 1:
        parser.error(
            "--mlp-batch-size must be at least 1."
        )

    if args.progress_every < 1:
        parser.error(
            "--progress-every must be at least 1."
        )

    records = read_fasta(
        args.sequence
    )

    protein_ids = [
        protein_id
        for protein_id, _ in records
    ]

    print(
        f"Loaded {len(records):,} FASTA records.",
        flush=True,
    )
    print(
        "Prediction mode: sequence_only",
        flush=True,
    )

    device = torch.device(
        args.device
    )

    embeddings = generate_batch_embeddings(
        records=records,
        device=device,
        progress_every=args.progress_every,
    )

    predictor = AltereGOPredictor(
        release_directory=args.release_directory,
        device=args.device,
    )

    result = predictor.predict_from_embeddings(
        embeddings=embeddings,
        mode="sequence_only",
        batch_size=args.mlp_batch_size,
        exploratory=args.exploratory,
    )

    scores = np.asarray(
        result["scores"],
        dtype=np.float32,
    )

    binary_predictions = np.asarray(
        result["binary_predictions"],
        dtype=np.uint8,
    )

    selected_terms, selected_indices = select_go_terms(
        requested_go_terms=args.go_terms,
        label_space=result["label_space"],
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predicted_output_path = args.output.with_name(
        f"{args.output.stem}_predicted"
        f"{args.output.suffix}"
    )

    npz_output_path = args.output.with_suffix(
        ".npz"
    )

    metadata_path = args.output.with_suffix(
        args.output.suffix + ".json"
    )

    if args.compact:
        npz_label_space = np.asarray(
            selected_terms,
            dtype=str,
        )

        npz_scores = scores[
            :,
            selected_indices,
        ]

        npz_binary_predictions = binary_predictions[
            :,
            selected_indices,
        ]

    else:
        label_table = pd.read_csv(
            args.release_directory
            / "resources"
            / "go_label_space.csv"
        )

        write_long_tsv(
            output_path=args.output,
            predicted_output_path=predicted_output_path,
            protein_ids=protein_ids,
            scores=scores,
            binary_predictions=binary_predictions,
            selected_terms=selected_terms,
            selected_indices=selected_indices,
            label_table=label_table,
        )

        npz_label_space = np.asarray(
            result["label_space"],
            dtype=str,
        )

        npz_scores = scores

        npz_binary_predictions = binary_predictions

    np.savez_compressed(
        npz_output_path,
        protein_ids=np.asarray(
            protein_ids,
            dtype=str,
        ),
        label_space=npz_label_space,
        scores=np.asarray(
            npz_scores,
            dtype=np.float32,
        ),
        binary_predictions=np.asarray(
            npz_binary_predictions,
            dtype=np.uint8,
        ),
        exploratory=np.asarray(args.exploratory, dtype=np.bool_),
        threshold=np.asarray(
            result["threshold"],
            dtype=np.float32,
        ),
    )

    if args.compact:
        print(
            "Compact score matrix saved to: "
            f"{npz_output_path}",
            flush=True,
        )

    else:
        predicted_counts = {
            protein_id: int(
                binary_predictions[index].sum()
            )
            for index, protein_id in enumerate(protein_ids)
        }

        metadata = {
            "mode": "sequence_only",
            "protein_count": len(protein_ids),
            "go_term_count": len(result["label_space"]),
            "decision_threshold": float(result["threshold"]),
            "exploratory": bool(args.exploratory),
            "mlp_batch_size": args.mlp_batch_size,
            "foundation_model_loading": (
                "Each foundation model was loaded once; "
                "sequences were embedded individually in FASTA order."
            ),
            "go_terms_written": selected_terms,
            "predicted_term_counts": predicted_counts,
            "output_file": str(args.output),
            "npz_file": str(npz_output_path),
        }

        with metadata_path.open("w") as handle:
            json.dump(
                metadata,
                handle,
                indent=2,
            )
            handle.write("\n")

        print(
            f"Predictions saved to: {args.output}",
            flush=True,
        )
        print(
            f"Selected predictions saved to: "
            f"{predicted_output_path}",
            flush=True,
        )
        print(
            f"Score matrix saved to: {npz_output_path}",
            flush=True,
        )
        print(
            f"Metadata saved to: {metadata_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
