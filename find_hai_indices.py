import numpy as np
import tqdm
import os
import torch
from torchvision import transforms
from utils import get_dataset
import argparse
from datasets.activations import ActivationsDataset, ChunkedActivationsDataset
from torch.utils.data import DataLoader, Subset
from itertools import combinations
from collections import Counter
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Find indices of highest activating images from activations")
    parser.add_argument('--activations_dir', type=str, required=True)
    parser.add_argument("--dataset_name", default="imagenet", type=str)
    parser.add_argument("--data_path", default="/shared-network/inat2021", type=str)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--chunk_size', type=int)
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    args.batch_size = 1  # not used
    args.num_workers = 0  # not used

    hai_indices_path = os.path.join(args.activations_dir, f"hai_indices_{args.k}")
    if os.path.exists(hai_indices_path):
        print(f"HAI indices already saved at {hai_indices_path}")
    else:
        print("Computing HAI indices", flush=True)
        print(f"Loading activations from {args.activations_dir}", flush=True)
        activations_dataset = ActivationsDataset(args.activations_dir, device=torch.device("cpu"))
        print(f"Dataset loaded. Total samples: {len(activations_dataset)}", flush=True)

        activations_dataloader = DataLoader(activations_dataset, batch_size=args.chunk_size, shuffle=False, num_workers=16)
        num_samples = len(activations_dataset)

        first_batch = next(iter(activations_dataloader))
        num_neurons = first_batch.shape[1]
        print(f"Number of neurons detected: {num_neurons}")
        num_chunks = math.ceil(num_neurons / args.chunk_size)
        print(f"Processing {num_chunks} chunks of {args.chunk_size} neurons each...", flush=True)

        importants = []
        worst_hais = []
        pbar = tqdm.tqdm(list(range(num_chunks)))
        for i in pbar:
            neuron_start = i * args.chunk_size
            neuron_end = min((i + 1) * args.chunk_size, num_neurons)
            activations_chunks = np.zeros((num_samples, neuron_end - neuron_start))
            for j, activations_chunk in enumerate(activations_dataloader):
                sample_start = j * args.chunk_size
                sample_end = min((j + 1) * args.chunk_size, num_samples)
                activations_chunk = activations_chunk.numpy()
                activations_chunks[sample_start:sample_end, :] = activations_chunk[:, neuron_start:neuron_end]
            for neuron in range(neuron_end - neuron_start):
                neuron_activations = activations_chunks[:, neuron]
                important = np.argsort(neuron_activations)[-args.k:]
                importants.append(important)
                worst_hai = neuron_activations[important[0]]
                worst_hais.append(worst_hai)

        hai_indices = np.array(importants)
        print(f"hai_indices.shape(): {hai_indices.shape}")
        np.save(hai_indices_path, hai_indices)
        print(f"Saved HAI indices to: {hai_indices_path}")

        worst_hai_indices_path = os.path.join(args.activations_dir, f"hai_indices_{args.k}_worst")
        worst_hais = np.array(worst_hais)
        np.save(worst_hai_indices_path, worst_hais)
        print(f"Saved worst HAI indices to: {worst_hai_indices_path}")