# AltereGO

AltereGO predicts Gene Ontology (GO) Molecular Function annotations by combining independently pretrained sequence- and structure-derived protein foundation models through score-level late fusion.

## Prediction modes

AltereGO automatically selects one of three validated inference modes according to the supplied inputs.

| Input | Models | Fusion weights | Default decision threshold |
|---|---|---|---:|
| Sequence | ESM2-3B, ProtT5, Ankh-Large | 0.40, 0.25, 0.35 | 0.62 |
| Structure | ProstT5-3Di | 1.00 | 0.64 |
| Sequence and structure | ESM2-3B, ProtT5, Ankh-Large, ProstT5-3Di | 0.35, 0.25, 0.25, 0.15 | 0.60 |

Sequence and structure arguments are individually optional, but at least one must be provided.

The initial release accepts one protein sequence and one single-chain structure per run. When both inputs are supplied, the user is responsible for ensuring that they represent the same protein.

## Exploratory prediction selection

By default, AltereGO uses the validated decision threshold for the supplied
input mode: 0.62 for sequence-only, 0.64 for structure-only, and 0.60 for
combined inference.

Add `--exploratory` to select GO terms whose final hierarchy-propagated score
is greater than or equal to 0.50. This optional setting supports exploration
of additional candidate functions.

Embeddings, normalization, fusion weights, GO propagation, continuous scores,
and score rankings remain unchanged. The 0.50 cutoff is an exploratory score
threshold, not a calibrated 50/50 probability of biological function.
Selecting more candidate terms may increase recall while reducing precision.

### Sequence-only exploratory prediction

    alterego --sequence protein.fasta --output exploratory_sequence.tsv --exploratory

### Structure-only exploratory prediction

    alterego --structure protein.pdb --output exploratory_structure.tsv --exploratory

Foldseek must be on PATH or supplied through `--foldseek`.

### Combined exploratory prediction

    alterego --sequence protein.fasta --structure protein.pdb --output exploratory_combined.tsv --exploratory

### Batch exploratory prediction

    alterego-batch --sequence proteins.fasta --output exploratory_batch.tsv --device cuda --exploratory

The batch command supports sequence-only inference and loads each foundation
model once for all input records. Run `--device cuda` on a GPU-equipped node.

The full TSV retains all scores. Its `predicted` column and the selected
`_predicted.tsv` use the active decision threshold. JSON metadata records
`decision_threshold` and the Boolean `exploratory` flag. Batch NPZ files also
record the active threshold, binary predictions, and exploratory flag,
including with `--compact`. Compact mode skips TSV and JSON outputs.

Omit `--exploratory` to retain the validated default threshold. Use separate
output paths when retaining both default and exploratory results.

## Installation

### Clone with Git LFS

The trained residual MLP weights are stored using Git Large File Storage (Git LFS). Install and initialize Git LFS before cloning:

    git lfs install
    git clone <repository-url>
    cd AltereGO

Verify that the model files were downloaded rather than retained as Git LFS pointer files:

    git lfs pull
    ls -lh model/*.safetensors

### Validated GPU environment

The provided environment was validated with Python 3.10, PyTorch 2.11.0, CUDA 12.6, and the dependency versions specified in `environment.yml`.

Create and activate the environment:

    conda env create -f environment.yml
    conda activate alterego

The environment installs AltereGO in editable mode. The cloned repository must therefore remain available after installation because it contains the trained model weights and GO resources.

### Existing Python environment

AltereGO can also be installed into an existing compatible environment:

    pip install -e .

For GPU use, install the PyTorch build appropriate for the local CUDA environment before installing AltereGO.

## Foundation-model downloads

The original pretrained ESM2-3B, ProtT5, Ankh-Large, and ProstT5 checkpoints are not duplicated in this repository. They are obtained through their respective software packages and model repositories on first use.

The first inference run therefore requires internet access unless the foundation-model checkpoints are already present in the local model cache. These checkpoints require substantial disk space.

## Foldseek

Foldseek is required for structure-based prediction and must either be available on `PATH` or supplied using `--foldseek`.

The structural workflow was validated using Foldseek commit:

    8dc75c74ad0eddab73cfd905963d13bf74dc012b

## Sequence-only prediction

    alterego \
      --sequence protein.fasta \
      --output predictions.tsv

Sequence-only prediction generates ESM2-3B, ProtT5, and Ankh-Large embeddings and applies the validated three-way fusion configuration.

## Structure-only prediction

    alterego \
      --structure protein.pdb \
      --output predictions.tsv

The supplied structure is converted into Foldseek 3Di tokens and embedded using ProstT5.

If Foldseek is not available on `PATH`:

    alterego \
      --structure protein.pdb \
      --foldseek /path/to/foldseek \
      --output predictions.tsv

PDB and mmCIF structure files are supported. The initial release expects the structure file to produce one Foldseek 3Di record.

## Combined sequence and structure prediction

    alterego \
      --sequence protein.fasta \
      --structure protein.pdb \
      --output predictions.tsv

This mode applies the published four-way late-fusion configuration.

## Device selection

AltereGO automatically uses CUDA when an available GPU is detected and otherwise falls back to CPU.

A device can be selected explicitly:

    alterego \
      --sequence protein.fasta \
      --output predictions.tsv \
      --device cuda

or:

    alterego \
      --sequence protein.fasta \
      --output predictions.tsv \
      --device cpu

GPU inference is strongly recommended. The foundation models are loaded sequentially to reduce peak memory consumption.

CPU inference is supported but can be slow and memory-intensive, particularly for sequence-based prediction, which requires three large foundation models. Structure-only CPU inference is generally more manageable.

## Output

The output table contains all 8,446 Molecular Function GO terms ranked from highest to lowest prediction score.

Output columns include:

- `protein_id`: input protein identifier
- `go_id`: Gene Ontology identifier
- `name`: GO term name
- `definition`: GO term definition
- `score`: hierarchy-propagated prediction score
- `predicted`: whether the score meets the active threshold (the validated default or 0.50 with `--exploratory`)

A JSON metadata file is written alongside the prediction table and records:

- Protein identifier
- Prediction mode
- Decision threshold
- Whether exploratory selection was enabled
- Number of threshold-selected terms

The complete ranked output is retained because biologically informative candidate functions may rank highly even when they do not exceed the global binary decision threshold.

## Training-time embedding behavior

The inference pipeline reproduces the embedding procedures used during model development:

- ESM2-3B residue representations are extracted from layer 36 and mean-pooled. Sequences longer than 1,022 residues are processed in nonoverlapping chunks.
- ProtT5 inputs use a maximum tokenized length of 1,024 and are truncated beyond that limit.
- Ankh-Large inputs use a maximum tokenized length of 1,024 and are truncated beyond that limit.
- ProstT5 inputs use Foldseek 3Di strings, the `<fold2AA>` prefix, and a maximum tokenized length of 1,024.
- Protein-level embeddings are L2-normalized before residual MLP inference.
- Prediction scores are propagated through the GO hierarchy before fusion and again after final fusion.

## Model files

The release contains four trained residual MLP predictors in Safetensors format:

    model/esm2_mlp.safetensors
    model/prott5_mlp.safetensors
    model/ankh_mlp.safetensors
    model/prostt5_3di_mlp.safetensors

The accompanying `weights_manifest.json` provides SHA-256 checksums and model metadata.

## Reproduction testing

A compact fixture derived from held-out test protein `A0A096LP01` verifies the four downstream predictors, L2 normalization, GO propagation, four-way fusion, and thresholding without rerunning the foundation models.

Run:

    python3 tests/test_reproduction.py

The expected result is:

    Downstream reproduction test: PASS

## License

AltereGO is distributed under the MIT License. See `LICENSE` for details.

Because the software was developed in the context of funded research, institutional authorization should be confirmed before changing the repository from private to public.

## Citation

A manuscript citation will be added upon publication. Until then, please cite the AltereGO repository and associated manuscript preprint when available.

## Included example

AltereGO includes an example protein to verify correct installation and end-to-end sequence-and-structure inference.

### Example protein

**Q97TX9 — CRISPR-associated exonuclease Cas4**

- UniProt: [Q97TX9](https://www.uniprot.org/uniprotkb/Q97TX9/entry)
- Organism: *Saccharolobus solfataricus* strain P2
- Sequence length: 202 amino acids
- Protein status: UniProtKB reviewed (Swiss-Prot)
- Structure input: AlphaFold Protein Structure Database model `AF-Q97TX9-F1-model_v6`
- Reference study: Zhang J, Kasciukovic T, White MF. *The CRISPR associated protein Cas4 is a 5′ to 3′ DNA exonuclease with an iron-sulfur cluster.* PLOS ONE (2012). [https://doi.org/10.1371/journal.pone.0047232](https://doi.org/10.1371/journal.pone.0047232)

Cas4 is a single-stranded DNA exonuclease associated with CRISPR adaptation. It contains a nuclease domain and binds a [4Fe–4S] cluster.

### Example directory

    example/Q97TX9/

The directory contains:

    Q97TX9.fasta
    AF-Q97TX9-F1-model_v6.pdb
    example_summary.json
    expected_output/
    run_output/

### Run the example

From the AltereGO repository root:

    conda activate alterego
    pip install -e .

    alterego \
      --sequence example/Q97TX9/Q97TX9.fasta \
      --structure example/Q97TX9/AF-Q97TX9-F1-model_v6.pdb \
      --output example/Q97TX9/run_output/predictions.tsv

If Foldseek is not available on `PATH`:

    alterego \
      --sequence example/Q97TX9/Q97TX9.fasta \
      --structure example/Q97TX9/AF-Q97TX9-F1-model_v6.pdb \
      --foldseek /path/to/foldseek \
      --output example/Q97TX9/run_output/predictions.tsv

### Expected result

At the validated four-way decision threshold of 0.60, the reference AltereGO prediction recovered 23 parent-propagated true Molecular Function terms with no false-positive terms. Two true terms were not recovered.

Reference per-protein performance was:

- Precision: 1.000
- Recall: 0.920
- F1: 0.958
- True-positive terms: 23
- False-positive terms: 0
- False-negative terms: 2

Recovered functions include:

- Single-stranded DNA 5′–3′ exonuclease activity
- DNA exonuclease and nuclease activity
- Hydrolase activity acting on ester bonds
- Iron–sulfur cluster binding
- 4 iron, 4 sulfur cluster binding
- Metal-ion binding

The two missed propagated terms were manganese-ion binding and transition-metal-ion binding.

The `expected_output` directory contains the reference outputs from a successful run. Newly generated results are written to `run_output` and can be compared with these files.

Small continuous-score differences may occur across CPU and GPU hardware or numerical precision modes. The threshold-selected predictions should remain consistent under the validated dependency configuration.

### Purpose of the example

This example verifies:

- FASTA parsing
- Foldseek 3Di conversion
- ESM2-3B, ProtT5, Ankh-Large, and ProstT5 embedding generation
- Loading of the four Safetensors residual MLP predictors
- GO hierarchy propagation
- Four-way score-level late fusion
- Ranked and thresholded Molecular Function output

## Batch sequence prediction

AltereGO supports sequence prediction from multi-record FASTA files through the `alterego-batch` command. Each FASTA identifier must be unique. The sequence foundation models are loaded once and reused across all records, avoiding repeated model-loading overhead.

```bash
alterego-batch \
  --sequence candidates.fasta \
  --output batch_predictions.tsv \
  --device cuda
```

The batch workflow applies the validated sequence-only fusion of ESM2-3B, ProtT5, and Ankh-Large. Records are processed sequentially within each foundation model to preserve the validated inference behavior and limit peak memory usage.

The command produces:

- `batch_predictions.tsv`: GO terms ranked by score for every protein.
- `batch_predictions_predicted.tsv`: terms meeting the active sequence-only decision threshold.
- `batch_predictions.npz`: complete score and binary-prediction matrices for all 8,446 GO terms.
- `batch_predictions.tsv.json`: metadata describing the run.

The human-readable TSV output can optionally be restricted to selected GO terms:

```bash
alterego-batch \
  --sequence candidates.fasta \
  --output selected_predictions.tsv \
  --go-terms GO:0004518 GO:0051539 \
  --device cuda
```

The NPZ archive retains the complete 8,446-term label space even when `--go-terms` is used. CPU batch inference is supported with `--device cpu`, although GPU inference is recommended for larger FASTA files.
