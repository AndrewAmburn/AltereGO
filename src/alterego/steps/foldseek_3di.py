#!/usr/bin/env python3

"""Convert a protein structure into a Foldseek 3Di sequence."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Union


def read_fasta(
    fasta_path: Union[str, Path],
) -> List[Tuple[str, str]]:
    """Read FASTA records without external dependencies."""

    fasta_path = Path(fasta_path)

    records = []
    current_identifier = None
    current_sequence = []

    with fasta_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current_identifier is not None:
                    records.append(
                        (
                            current_identifier,
                            "".join(current_sequence),
                        )
                    )

                current_identifier = (
                    line[1:].strip().split()[0]
                )
                current_sequence = []

            else:
                current_sequence.append(line)

    if current_identifier is not None:
        records.append(
            (
                current_identifier,
                "".join(current_sequence),
            )
        )

    return records


def run_command(
    command: List[str],
) -> None:
    """Run Foldseek and provide a useful error if it fails."""

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Foldseek command failed:\n"
            + " ".join(command)
            + "\n\nStandard output:\n"
            + result.stdout
            + "\nStandard error:\n"
            + result.stderr
        )


def structure_to_3di(
    structure_path: Union[str, Path],
    foldseek_executable: str = "foldseek",
) -> str:
    """
    Convert one PDB or mmCIF protein structure to one Foldseek 3Di string.

    The commands reproduce the original AltereGO structural-token workflow:
    createdb -> lndb -> convert2fasta.
    """

    structure_path = Path(
        structure_path
    ).resolve()

    if not structure_path.is_file():
        raise FileNotFoundError(
            f"Structure file was not found: {structure_path}"
        )

    executable = shutil.which(
        foldseek_executable
    )

    if executable is None:
        raise FileNotFoundError(
            "Foldseek was not found on PATH. Install Foldseek or provide "
            "the path to its executable."
        )

    with tempfile.TemporaryDirectory(
        prefix="alterego_foldseek_"
    ) as temporary_directory:
        temporary_directory = Path(
            temporary_directory
        )

        database = (
            temporary_directory
            / "structure_db"
        )

        fasta_path = (
            temporary_directory
            / "structure_3di.fasta"
        )

        run_command(
            [
                executable,
                "createdb",
                str(structure_path),
                str(database),
            ]
        )

        run_command(
            [
                executable,
                "lndb",
                str(database) + "_h",
                str(database) + "_ss_h",
            ]
        )

        run_command(
            [
                executable,
                "convert2fasta",
                str(database) + "_ss",
                str(fasta_path),
            ]
        )

        records = read_fasta(
            fasta_path
        )

        records = [
            (identifier, sequence)
            for identifier, sequence in records
            if sequence
        ]

        if not records:
            raise RuntimeError(
                "Foldseek did not produce a 3Di sequence."
            )

        if len(records) != 1:
            raise ValueError(
                "The structure produced multiple Foldseek 3Di records. "
                "The initial AltereGO release expects one protein chain "
                "per structure file."
            )

        return records[0][1].strip().lower()
