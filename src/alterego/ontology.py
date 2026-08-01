#!/usr/bin/env python3

from pathlib import Path
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd


def load_label_space(
    label_space_path: Union[str, Path],
) -> List[str]:
    """Load GO terms in the exact output-column order used during training."""

    label_space_path = Path(label_space_path)

    label_table = pd.read_csv(label_space_path)

    if "go_id" not in label_table.columns:
        raise ValueError(
            f"{label_space_path} does not contain a go_id column."
        )

    label_space = (
        label_table["go_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(label_space) != 8446:
        raise ValueError(
            "The trained AltereGO predictors require exactly 8,446 "
            f"GO terms, but {len(label_space):,} were loaded."
        )

    if len(set(label_space)) != len(label_space):
        raise ValueError("The GO label space contains duplicate terms.")

    return label_space


def load_go_parents(
    obo_path: Union[str, Path],
    label_space: List[str],
) -> Dict[str, Set[str]]:
    """
    Load direct in-label-space is_a parents from a Gene Ontology OBO file.

    This reproduces the relationship type used in the original AltereGO
    standalone and fusion scripts.
    """

    obo_path = Path(obo_path)
    allowed_terms = set(label_space)

    parents = {
        go_term: set()
        for go_term in label_space
    }

    current_id = None
    current_parents = []

    def store_current_term():
        if current_id in parents:
            parents[current_id].update(
                parent
                for parent in current_parents
                if parent in allowed_terms
            )

    with obo_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if line == "[Term]":
                store_current_term()
                current_id = None
                current_parents = []

            elif line.startswith("id: GO:"):
                current_id = line.split("id: ", 1)[1]

            elif line.startswith("is_a: GO:"):
                parent = (
                    line.split("is_a: ", 1)[1]
                    .split()[0]
                )
                current_parents.append(parent)

            elif line == "":
                store_current_term()
                current_id = None
                current_parents = []

    store_current_term()

    return parents


def propagate_scores(
    scores: np.ndarray,
    label_space: List[str],
    parents: Dict[str, Set[str]],
) -> np.ndarray:
    """
    Propagate descendant scores to direct parents until convergence.

    Each parent receives at least the maximum score assigned to any
    descendant, reproducing the original AltereGO implementation.
    """

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    if scores.ndim != 2:
        raise ValueError(
            "Scores must have shape (terms,) or (proteins, terms)."
        )

    if scores.shape[1] != len(label_space):
        raise ValueError(
            f"Score matrix has {scores.shape[1]:,} columns, but the "
            f"label space contains {len(label_space):,} terms."
        )

    label_to_index = {
        go_term: index
        for index, go_term in enumerate(label_space)
    }

    propagated = scores.copy()

    changed = True

    while changed:
        changed = False

        for child, direct_parents in parents.items():
            child_index = label_to_index[child]
            child_scores = propagated[:, child_index]

            for parent in direct_parents:
                parent_index = label_to_index[parent]

                updated = np.maximum(
                    propagated[:, parent_index],
                    child_scores,
                )

                if not np.array_equal(
                    updated,
                    propagated[:, parent_index],
                ):
                    propagated[:, parent_index] = updated
                    changed = True

    return propagated
