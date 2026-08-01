#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from alterego.pipeline import AltereGOPredictor


RELEASE = Path(__file__).resolve().parents[1]
FIXTURE = (
    RELEASE
    / "tests"
    / "data"
    / "A0A096LP01_reproduction.npz"
)


def test_fourway_reproduction():
    fixture = np.load(FIXTURE)

    embeddings = {
        "esm2": fixture["esm2"].reshape(1, -1),
        "prott5": fixture["prott5"].reshape(1, -1),
        "ankh": fixture["ankh"].reshape(1, -1),
        "prostt5_3di": fixture[
            "prostt5_3di"
        ].reshape(1, -1),
    }

    predictor = AltereGOPredictor(
        release_directory=RELEASE,
        device="cpu",
    )

    result = predictor.predict_from_embeddings(
        embeddings=embeddings,
        mode="sequence_and_structure",
    )

    expected_scores = fixture[
        "expected_fourway_scores"
    ].reshape(1, -1)

    expected_binary = fixture[
        "expected_fourway_binary"
    ].reshape(1, -1)

    assert np.allclose(
        result["scores"],
        expected_scores,
        atol=5e-6,
        rtol=1e-6,
    )

    assert np.array_equal(
        result["binary_predictions"],
        expected_binary,
    )


if __name__ == "__main__":
    test_fourway_reproduction()
    print("Downstream reproduction test: PASS")
