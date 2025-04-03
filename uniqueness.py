import argparse
import numpy as np
import os
from itertools import combinations
from tqdm import tqdm  # Progress bar


def parse_args():
    parser = argparse.ArgumentParser(description="Measure uniqueness of neurons with pairwise Jaccard Index")
    parser.add_argument('--activations_dir', type=str, required=True)
    parser.add_argument('--k', type=int, default=16)
    return parser.parse_args()


def jaccard_index(set1, set2):
    """Compute Jaccard index between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


if __name__ == "__main__":
    args = parse_args()

    hai_indices_path = os.path.join(args.activations_dir, f"hai_indices_{args.k}.npy")
    worst_scores_path = os.path.join(args.activations_dir, f"hai_indices_{args.k}_worst.npy")

    hai_indices = np.load(hai_indices_path)  # (num_neurons, k)
    worst_scores = np.load(worst_scores_path)  # (num_neurons)

    print(f"Loaded HAI indices from {hai_indices_path}")
    print(f"Loaded worst scores from {worst_scores_path}")

    # Correct mask condition
    mask = worst_scores != 0  # Keep only neurons that are NOT "dead"
    hai_indices = hai_indices[mask]

    print(f"Removed {np.count_nonzero(~mask)} dead (or almost dead) neurons")
    print(f"Remaining neurons: {hai_indices.shape[0]}")

    # Compute pairwise Jaccard index
    num_neurons = hai_indices.shape[0]
    jaccard_scores = []
    index_pairs = []

    total_pairs = (num_neurons * (num_neurons - 1)) // 2  # Number of unique pairs

    for i, j in tqdm(combinations(range(num_neurons), 2), total=total_pairs, desc="Computing Jaccard Index"):
        set1, set2 = set(hai_indices[i]), set(hai_indices[j])
        jaccard = jaccard_index(set1, set2)
        jaccard_scores.append(jaccard)
        index_pairs.append((i, j))

    jaccard_scores = np.array(jaccard_scores)

    # Sort Jaccard scores
    sorted_indices = np.argsort(jaccard_scores)  # Ascending order

    # Extract top and bottom 10 index pairs
    top_10 = [(index_pairs[i], jaccard_scores[i]) for i in sorted_indices[-10:]]  # Top 10 highest Jaccard
    bottom_10 = [(index_pairs[i], jaccard_scores[i]) for i in sorted_indices[:10]]  # Bottom 10 lowest Jaccard

    print(f"Total pairs: {total_pairs}")
    # Count how many have Jaccard Index > 0.1 * i
    for i in np.arange(0, 1, 0.1):
        high_similarity_count = np.count_nonzero(jaccard_scores > i)
        print(f"Pairs with Jaccard index > {i:.1f}: {high_similarity_count}")
        print(f"Ratio: {high_similarity_count / total_pairs:.4f}")

    print(f"Top 10 Indexes (Most Similar Pairs): {top_10}")
    print(f"Bottom 10 Indexes (Most Unique Pairs): {bottom_10}")
