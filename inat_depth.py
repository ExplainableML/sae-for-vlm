import numpy as np
import tqdm
import os
import torch
from utils import get_dataset
import argparse
from itertools import combinations
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Measure hierarchy in iNaturalist trained Matryoshka SAE")
    parser.add_argument('--activations_dir', type=str, required=True)
    parser.add_argument('--hai_indices_path', type=str, required=True)
    parser.add_argument("--data_path", default="/shared-network/inat2021", type=str)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--group_fractions', type=float, nargs='+', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    args.batch_size = 1  # not used
    args.num_workers = 0  # not used
    args.dataset_name = 'inat'  # fixed

    # Get HAI indices
    hai_indices = np.load(args.hai_indices_path)
    print(f"Loaded HAI indices found at {args.hai_indices_path}")

    # Get HAI worst scores
    worst_scores_path = f"{args.hai_indices_path[:-4]}_worst.npy"
    worst_scores = np.load(worst_scores_path)
    print(f"Loaded worst scores of HAI found at {worst_scores_path}")

    # Assign path to each of top k images
    hai_classes = []
    num_neurons = hai_indices.shape[0]
    ds, _ = get_dataset(args, preprocess=None, processor=None, split=args.split)

    for neuron in range(num_neurons):
        hai_classes_neuron = []
        for i in range(args.k):
            class_index = hai_indices[neuron, i]
            image_path = ds.imgs[class_index][0]
            class_name = image_path.split(os.path.sep)[-2].split("_")[1:]
            hai_classes_neuron.append(class_name)
        hai_classes.append(hai_classes_neuron)


    # Compute pairwise LCA
    def get_lca(x, y):
        lca = 0
        for a, b in zip(x, y):
            if a == b:
                lca += 1
            else:
                break
        return lca


    lcas_majority = []
    lcas_mean = []

    for neuron in range(num_neurons):
        lcas_neuron = []
        hai_classes_neuron = hai_classes[neuron]
        for x, y in combinations(hai_classes_neuron, 2):
            lca = get_lca(x, y)
            lcas_neuron.append(lca)

        if lcas_neuron:
            lcas_mean.append(round(sum(lcas_neuron) / len(lcas_neuron)))
            lcas_majority.append(Counter(lcas_neuron).most_common(1)[0][0])

    # Compute avg depth of LCA
    assert np.isclose(sum(args.group_fractions), 1.0), "group_fractions must sum to 1.0"
    group_sizes = [int(f * num_neurons) for f in args.group_fractions[:-1]]
    group_sizes.append(num_neurons - sum(group_sizes))  # Ensure it adds up to num_neurons

    start_idx = 0
    depths_majority = []
    depths_mean = []

    for group_idx, group_size in enumerate(group_sizes):
        end_idx = start_idx + group_size
        valid_mask = worst_scores[start_idx:end_idx] != 0.0
        valid_lcas_majority = np.compress(valid_mask, lcas_majority[start_idx:end_idx])
        valid_lcas_mean = np.compress(valid_mask, lcas_mean[start_idx:end_idx])
        depths_majority.append(np.mean(valid_lcas_majority))
        depths_mean.append(np.mean(valid_lcas_mean))
        start_idx = end_idx

        num_excluded = np.sum(~valid_mask)
        percentage_excluded = (num_excluded / group_size) * 100
        print(f"Group {group_idx}: {percentage_excluded:.2f}% neurons excluded")

    print("Group-wise Average WordNet Depths (mean):", depths_mean)
    print("Group-wise Average WordNet Depths (majority):", depths_majority)