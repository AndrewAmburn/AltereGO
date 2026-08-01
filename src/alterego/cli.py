#!/usr/bin/env python3

"""Command-line interface for AltereGO."""

import os

# AltereGO uses PyTorch exclusively. Prevent Transformers from loading
# unnecessary TensorFlow or Flax backends.
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .embedding_pipeline import (
    generate_embeddings,
    read_single_fasta,
)
from .pipeline import AltereGOPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alterego",
        description=(
            "Predict Gene Ontology Molecular Function terms using "
            "sequence, structure, or combined AltereGO inference."
        ),
    )

    parser.add_argument(
        "--sequence",
        type=Path,
        help="FASTA file containing one protein sequence.",
    )

    parser.add_argument(
        "--structure",
        type=Path,
        help="One protein structure in PDB or mmCIF format.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output TSV file for ranked GO predictions.",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        help="Device used for foundation-model and MLP inference.",
    )

    parser.add_argument(
        "--foldseek",
        default="foldseek",
        help="Foldseek executable name or absolute path.",
    )

    parser.add_argument(
        "--release-directory",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help=argparse.SUPPRESS,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.sequence is None and args.structure is None:
        parser.error(
            "At least one of --sequence or --structure is required."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA was requested, but no CUDA device is available."
        )

    sequence = None
    protein_id = None

    if args.sequence is not None:
        protein_id, sequence = read_single_fasta(
            args.sequence
        )

    if protein_id is None:
        protein_id = args.structure.stem

    mode = AltereGOPredictor.determine_mode(
        sequence_supplied=(
            args.sequence is not None
        ),
        structure_supplied=(
            args.structure is not None
        ),
    )

    print(f"Prediction mode: {mode}", flush=True)

    embeddings = generate_embeddings(
        sequence=sequence,
        structure_path=args.structure,
        device=args.device,
        foldseek_executable=args.foldseek,
    )

    predictor = AltereGOPredictor(
        release_directory=args.release_directory,
        device=args.device,
    )

    result = predictor.predict_from_embeddings(
        embeddings=embeddings,
        mode=mode,
    )

    label_table = pd.read_csv(
        args.release_directory
        / "resources"
        / "go_label_space.csv"
    )

    scores = result["scores"][0]
    binary = result[
        "binary_predictions"
    ][0]

    output = pd.DataFrame(
        {
            "protein_id": protein_id,
            "go_id": result["label_space"],
            "score": scores,
            "predicted": binary.astype(bool),
        }
    )

    for column in [
        "name",
        "definition",
    ]:
        if column in label_table.columns:
            output[column] = label_table[column].values

    preferred_columns = [
        column
        for column in [
            "protein_id",
            "go_id",
            "name",
            "definition",
            "score",
            "predicted",
        ]
        if column in output.columns
    ]

    output = output[
        preferred_columns
    ].sort_values(
        "score",
        ascending=False,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        sep="\t",
        index=False,
    )

    predicted_output_path = args.output.with_name(
        f"{args.output.stem}_predicted{args.output.suffix}"
    )

    predicted_output = output[
        output["predicted"]
    ].copy()

    predicted_output.to_csv(
        predicted_output_path,
        sep="\t",
        index=False,
    )

    metadata = {
        "protein_id": protein_id,
        "mode": result["mode"],
        "decision_threshold": result["threshold"],
        "predicted_terms": int(binary.sum()),
        "output_file": str(
            args.output.resolve()
        ),
    }

    metadata_path = args.output.with_suffix(
        args.output.suffix + ".json"
    )

    with metadata_path.open("w") as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )
        handle.write("\n")

    print(
        f"Predicted terms: {metadata['predicted_terms']}",
        flush=True,
    )
    print(
        f"Selected predictions saved to: {predicted_output_path}",
        flush=True,
    )

    print(f"Predictions saved to: {args.output}", flush=True)
    print(f"Metadata saved to: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
