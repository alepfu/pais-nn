# PAIS-NN

PAIS-NN is a tool for classifying prokaryotic insertion sequences using cluster labels defined by the [Prokaryotic Atlas of Insertion Sequences (PAIS)](https://pais.probst-lab.uni-due.de). It applies a neural network model to predict PAIS cluster labels based on k-mer composition and then estimates the ecosystem proportions of a sample using an expectation-maximization (EM) approach. If you use PAIS-NN in your work, please cite the associated paper: (TODO: add citation).

__Features__
- Sequence embedding using k-mer frequencies
- PAIS cluster prediction via a calibrated feedforward neural network
- Expectation-Maximization (EM) estimation of ecosystem proportions
- Support for multiple input sites and configurable output formats
- Plotting of ecosystem composition results

## Installation

Environment managed with Conda (see paisnn_environment.yaml), major packages include: 
- `torch`
- `CUDA 12.8`
- `scikit-learn`
- `pandas`
- `seaborn`
- `biopython`

```bash
# Firsts, clone the repository to your local machine
git clone https://github.com/alepfu/pais-nn.git
cd pais-nn

# Second, setup and activate Conda environment
conda env create -f paisnn_environment.yaml
conda activate paisnn
```

## Usage
```bash
python pais-nn.py \
	-m paisnn_min_size_10_clusters.pth \
	-s test_data/river_estuary \
	-e paisnn_ecosystem_priors.csv
```
__Arguments__
| Flag                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `-m`, `--model`       | Path to trained `.pth` model (required)                       |
| `-s`, `--sitedir`     | Path to site directory with sequences (required)              |
| `-e`, `--ecosyspriors` | Path to `.csv` with ecosystem prior distributions (required)  |
| `-p`, `--plotfmt`     | Output plot format: `png` (default) or `svg`                  |
| `-c`, `--confthresh`  | Confidence threshold for filtering predictions (default: 0.5) |
| `-v`, `--verbosity`   | Logging verbosity: 0 = silent, 1 = info, 2 = debug (default)  |

__Inputs__
- Site directory: Should contain subdirectories of (gzipped) FASTA files per site.
- Model file: A serialized PyTorch model bundle (.pth) including weights and label encoder.
- Ecosystem priors file: Prior probabilities for ecosystems used in EM. CSV-file with columns: ecosys_label, cluster_label.

__Outputs__
- cluster_predictions.csv: Sequence-level predictions with confidence scores.
- ecosystem_proportions.csv: Estimated ecosystem proportions per site/sample.
- ecosystem_proportions.png/svg: Grouped barplot visualization of estimated proportions over sites.

__Expected directory structure__
```
site_dir/
├── site_A
│   ├── sample1.fasta
│   ├── sample2.fasta
|	└── ...
├── site_B
│   ├── sample1.fasta
│   ├── sample2.fasta
|	└── ...
└── ...
```

__Example Data__
- `test_data/river_estuary` contains metagenomic samples collected from three different sites (BR2, BR1, and BAY) along the Brisbane River estuary ([Prabhu et al. 2024](https://doi.org/10.1093/ismeco/ycae067)).

## License
[MIT](https://github.com/alepfu/pais-nn/blob/main/LICENSE)
