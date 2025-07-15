#!/usr/bin/env python3

"""
Script: pais-nn.py
Author: Alexander Pfundner
Created: 2025-07-11

Description:
    Applies the PAIS neural network model to assign IS cluster labels to new sequences and
    estimates ecosystem proportions using expectation-maximization.

Usage:
    python pais-nn.py -m <model-file> -s <site-dir> -e <ecosystem-priors-file>

Arguments:
    -m, --model         Path to the trained PAIS neural network model to use for prediction. (required)
    -s, --sitedir       Path to the site directory containing the new sequences to classify. (required)
    -e, --ecosyspriors  Path to the CSV file containing ecosystem distribution priors. (required)
    -p, --plotfmt       Format for output plots. Options: 'png' (default) or 'svg'.
    -c, --confthresh    Confidence threshold for filtering predictions. Default: 0.5
    -v, --verbosity     Verbosity level: 0 = silent, 1 = info, 2 = debug (default).

Dependencies:
    - See paisnn_environment.yaml
"""

import logging
import argparse
from pathlib import Path
import torch.nn as nn
from Bio import SeqIO
import pickle
import io
import gzip
import torch
import torch.nn.functional as F
import itertools
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

OUT_PREDS = 'cluster_predictions.csv'
OUT_PROPS = 'ecosystem_proportions.csv'
OUT_PLOT_PREF = 'ecosystem_proportions'
CONFIDENCE_THRESHOLD = 0.5

logger = logging.getLogger("main")
sns.set_style("whitegrid")


class FNN(nn.Module):
    """
    Feedforward Neural Network with 4 hidden layers, batch normalization, ReLU activation
    and temperature scaling for post-calibration.
    """
    def __init__(self, input_size, output_size,
                 hidden_size1=512, hidden_size2=1024,
                 hidden_size3=512, hidden_size4=256):
        super(FNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, hidden_size4),
            nn.BatchNorm1d(hidden_size4),
            nn.ReLU(),
            nn.Linear(hidden_size4, output_size)
        )
        self.register_buffer("temperature", torch.ones(1))  # Buffer is saved in state_dict

    def forward(self, x):
        logits = self.net(x)
        return logits / self.temperature


class KmerEmbedder:
    def __init__(self, k=3, alphabet=None):
        """Initialize the KmerEmbedder with a given alphabet and k-mer length."""
        if alphabet is None:
            alphabet = ['A', 'C', 'G', 'T']
        self.alphabet = alphabet
        self.k = k
        self.all_possible_kmers = self.generate_all_possible_kmers()
        self.embedding_length = len(self.all_possible_kmers)

    def generate_kmers(self, sequence):
        """Generate k-mers from a sequence."""
        return [sequence[i:i + self.k] for i in range(len(sequence) - self.k + 1)]

    def generate_all_possible_kmers(self):
        """
        Generate all possible k-mers for the given alphabet.
        """
        return sorted([''.join(kmer) for kmer in itertools.product(self.alphabet, repeat=self.k)])

    def embed_sequences(self, sequences):
        """Convert sequences into a normalized k-mer frequency matrix."""
        kmers_list = []
        for seq in sequences:
            kmers = self.generate_kmers(seq)
            kmers_list.append(" ".join(kmers))  # Join k-mers as space-separated strings

        # Use CountVectorizer with predefined vocabulary to count k-mer occurrences
        vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\b\w+\b',
                                     vocabulary=self.all_possible_kmers, lowercase=False)
        kmer_matrix = vectorizer.fit_transform(kmers_list)

        df_kmers = pd.DataFrame(kmer_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Normalize counts to frequencies by dividing by row sum
        df_kmers = df_kmers.div(df_kmers.sum(axis=1), axis=0)

        return df_kmers


def collect_site_seqs(input_dir):
    """Collect sequences, their corresponding filenames, and sequence IDs for each site."""
    results = {}
    for site_dir in Path(input_dir).iterdir():
        if not site_dir.is_dir():
            continue

        site_name = site_dir.name
        sequences = []
        filenames_per_seq = []
        sequence_ids = []

        for file_path in sorted(site_dir.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    seqs = list(SeqIO.parse(f, 'fasta'))
            else:
                seqs = list(SeqIO.parse(file_path, 'fasta'))

            sequences.extend(str(record.seq) for record in seqs)
            filenames_per_seq.extend([file_path.name] * len(seqs))
            sequence_ids.extend([record.id for record in seqs])

        results[site_name] = (sequences, filenames_per_seq, sequence_ids)

    logger.debug("No. of sequences per site:")
    for site, (sequences, filenames_per_seq, sequence_ids) in results.items():
        logger.debug(f"  {site}: {len(sequences)}")

    return results


def parse_plot_format(value):
    """Checks if the passed plot format is valid."""
    fmt = value.strip().lower()
    if fmt not in {"png", "svg"}:
        raise argparse.ArgumentTypeError(f"Invalid plot format '{value}'. Allowed values are 'png' or 'svg'.")
    return fmt


def calc_ecosys_props(df_pred, reference_csv):
    """Estimate ecosystem proportions using EM."""
    # Estimate P(cluster | ecosystem) from reference data
    P, ecosystem_to_idx, cluster_to_idx = estimate_p_cluster_given_ecosystem(reference_csv)

    # Group predictions by sample and apply EM to each
    props = {}
    for sample_id, group in df_pred.groupby('sample_id'):
        cluster_labels = group['cluster'].tolist()
        proportions = em_ecosystem_mixture(cluster_labels, P, ecosystem_to_idx, cluster_to_idx)
        props[sample_id] = proportions
    return props


def estimate_p_cluster_given_ecosystem(csv_path):
    """Load CSV and estimate P(cluster | ecosystem) matrix."""
    df = pd.read_csv(csv_path)

    # Get sorted unique ecosystem and cluster labels
    ecosystems = sorted(df['ecosys_label'].unique())
    clusters = sorted(df['cluster_label'].unique())

    # Create mapping dictionaries for indexing into the probability matrix
    ecosystem_to_idx = {eco: i for i, eco in enumerate(ecosystems)}
    cluster_to_idx = {cl: i for i, cl in enumerate(clusters)}

    # Initialize count matrix of shape [n_ecosystems, n_clusters]
    P = np.zeros((len(ecosystems), len(clusters)))

    # Count how often each (ecosystem, cluster) pair occurs
    # Produces a Series with a multi-index
    grouped = df.groupby(['ecosys_label', 'cluster_label']).size()

    # Fill the count matrix with the observed counts
    for (eco, cl), count in grouped.items():
        i = ecosystem_to_idx[eco]   # row index for ecosystem
        j = cluster_to_idx[cl]      # column index for cluster
        P[i, j] = count             # set count

    # Normalize each row to convert counts to conditional probabilities
    P = P / P.sum(axis=1, keepdims=True)

    return P, ecosystem_to_idx, cluster_to_idx


def em_ecosystem_mixture(cluster_labels, P_cluster_given_ecosys, ecosys_to_idx, cluster_to_idx, max_iter=100, tol=1e-6):
    """Perform EM to estimate ecosystem mixture proportions for a sample."""
    # Initialize ecosystem proportions uniformly
    n_ecosystems = len(ecosys_to_idx)
    theta = np.ones(n_ecosystems) / n_ecosystems

    # Map cluster labels to their indices in the probability matrix
    cluster_ids = [cluster_to_idx[cl] for cl in cluster_labels if cl in cluster_to_idx]
    cluster_ids = np.array(cluster_ids)

    # EM loop
    for _ in range(max_iter):
        # E-step: compute posterior weights of ecosystems for each cluster observation
        weights = np.zeros((len(cluster_ids), n_ecosystems))
        for i in range(n_ecosystems):
            weights[:, i] = theta[i] * P_cluster_given_ecosys[i, cluster_ids]

        # Normalize weights for each data point
        # Transforms raw likelihood-weighted values into posterior probabilities
        weights /= weights.sum(axis=1, keepdims=True)

        # M-step: update ecosystem proportions based on weights
        theta_new = weights.sum(axis=0) / len(cluster_ids)

        # Check for convergence
        if np.linalg.norm(theta_new - theta) < tol:
            break
        theta = theta_new

    # Return ecosystem proportions as dictionary
    proportions = {eco: theta[i] for eco, i in ecosys_to_idx.items()}
    return proportions


def plot_ecosystem_props(df_props, plot_format):
    """Grouped bar plot with error bars."""
    # Convert to percentages
    df_props['proportion'] = round(df_props['proportion'] * 100, 2)

    # Handle ecosystems with long names
    df_props['ecosystem'] = df_props['ecosystem'].replace({
        'Saline and Alkaline inland systems': 'Saline/Alkaline\ninland systems',
        'Hydrocarbon-associated': 'Hydrocarbon-\nassociated',
        'Industrial environments': 'Industrial\nenvironments',
        'Synthetic communities': 'Synthetic\ncommunities',
        'Fungi-associated': 'Fungi-\nassociated',
        'Human-associated': 'Human-\nassociated',
        'Animal-associated': 'Animal-\nassociated',
        'Plant-associated': 'Plant-\nassociated',
        'Thermal springs': 'Thermal\nsprings',
        'Built environment': 'Built\nenvironment'
    })

    # Plotting
    plt.figure(figsize=(16, 6))
    sns.barplot(data=df_props, x="ecosystem", y="proportion", hue="site", errorbar="se", capsize=0.1,
                order=df_props.groupby('ecosystem')['proportion'].mean().sort_values(ascending=False).index)
    plt.ylabel('Proportion (± SE)', fontsize=14)
    plt.xlabel('')
    plt.xticks(fontsize=8)
    plt.title('Ecosystem proportions over sites', fontsize=16)
    plt.legend(title="Site", title_fontsize='14')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))  # Add %-sign to tick labels
    plt.tight_layout()
    plt.savefig(f'{OUT_PLOT_PREF}.{plot_format}')


def setup_logger(verbosity):
    """Set up logger with specified verbosity level (0 = silent, 1 = info, 2 = debug)."""
    logger.handlers.clear()
    if verbosity <= 0:
        logger.setLevel(logging.CRITICAL + 1)  # No output
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def apply_conf_thresh(df_preds, conf_thresh):
    """Filters out low confidence predictions."""
    logger.debug(f"No. predictions before conf. threshold: {len(df_preds)}")
    logger.debug(f"Average probability before conf. threshold: {df_preds['prob'].mean():.4f}")
    result = df_preds[df_preds['prob'] >= conf_thresh].copy()
    logger.debug(f"No. predictions after conf. threshold: {len(result)}")
    logger.debug(f"No. predictions removed: {len(df_preds) - len(result)}")
    return result


def setup_parser():
    """Initialze argument parser."""
    parser = argparse.ArgumentParser(
        description="PAIS-NN | Predict PAIS cluster labels and estimate ecosystem proportions")
    parser.add_argument('-m', '--model', required=True, help='Model to use (mandatory)')
    parser.add_argument('-s', '--sitedir', required=True, help='Site directory (mandatory)')
    parser.add_argument('-e', '--ecosyspriors', required=True, help='Ecosystem priors CSV-file (mandatory)')
    parser.add_argument('-p', '--plotfmt', default='png', type=parse_plot_format,
                        help="Plot format, either 'png' or 'svg'.")
    parser.add_argument('-c', '--confthresh', default=CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold for prediction filtering (default {CONFIDENCE_THRESHOLD})")
    parser.add_argument('-v', '--verbosity', default='2', help="Verbosity level, 0 = silent, 1 = info, 2 = debug")
    return parser


class CPUUnpickler(pickle.Unpickler):
    """Custom loading to handle CUDA tensors on CPU-only machines."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def main():
    parser = setup_parser()
    args = parser.parse_args()
    setup_logger(int(args.verbosity))

    # Collect sequences for every subdirectory
    logger.info('Loading sequences...')
    site_seqs = collect_site_seqs(args.sitedir)

    # Embed sequences
    logger.info('Embedding sequences...')
    embedder = KmerEmbedder(k=3)
    site_seqs_embedded = {}
    for site, (seqs, filenames, seq_ids) in site_seqs.items():
        embeddings = embedder.embed_sequences(seqs)
        embeddings_tensor = torch.tensor(embeddings.values, dtype=torch.float32)
        site_seqs_embedded[site] = (embeddings_tensor, filenames, seq_ids)

    # Load model
    logger.info('Loading model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f'Using device: {device}')
    with open(args.model, 'rb') as f:
        if device.type == 'cpu':
            bundle = CPUUnpickler(f).load()
        else:
            bundle = pickle.load(f)
    model_weights = bundle['model_state_dict']
    le = bundle['label_encoder']
    fnn = FNN(embedder.embedding_length, len(le.classes_))
    fnn.load_state_dict(model_weights)
    fnn.to(device)
    fnn.eval()

    # Run predictions
    logger.info('Running predictions...')
    preds = []
    for site, (tensors, filenames, seq_ids) in site_seqs_embedded.items():
        with torch.no_grad():
            tensors = tensors.to(device)
            probs = F.softmax(fnn(tensors), dim=1)
            top_probs, top_indices = torch.max(probs, dim=1)
            top_probs = top_probs.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            top_clusters = le.inverse_transform(top_indices)

        for i in range(len(top_probs)):
            preds.append({
                "site": site,
                "sample_id": filenames[i],
                "seq_id": seq_ids[i],
                "cluster": top_clusters[i],
                "prob": top_probs[i]
            })

    # Output predictions to CSV-file
    df_preds = pd.DataFrame(preds)
    df_preds = apply_conf_thresh(df_preds, args.confthresh)
    df_preds.to_csv(OUT_PREDS, index=False)
    logger.info(f'Cluster predictions written to {OUT_PREDS}')

    # Run EM calculations per site
    logger.info('Running EM calculations...')
    site_props = {}
    for site, preds in df_preds.groupby('site'):
        site_props[site] = calc_ecosys_props(preds, args.ecosyspriors)

    # Combine results into long format dataframe and output to CSV-fle
    records = []
    for site, samples in site_props.items():
        for sample_id, ecosys_props in samples.items():
            for ecosys_label, proportion in ecosys_props.items():
                records.append({
                    "site": site,
                    "sample_id": sample_id,
                    "ecosystem": ecosys_label,
                    "proportion": proportion
                })
    df_props = pd.DataFrame(records)
    df_props.to_csv(OUT_PROPS, index=False)
    logger.info(f'Ecosystem proportions written to {OUT_PREDS}')

    # Plot ecosystem proportions over sites
    plot_ecosystem_props(df_props, args.plotfmt)
    logger.info(f'Ecosystem proportions plot saved to {OUT_PLOT_PREF}.{args.plotfmt}')


if __name__ == "__main__":
    main()
