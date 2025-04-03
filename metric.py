import torch
import os.path
import argparse
from datasets.activations import ActivationsDataset
import os

from torch.utils.data import DataLoader, Subset
import tqdm
import torch.nn.functional as F

def get_args_parser():
    parser = argparse.ArgumentParser("Measure monosemanticity via weighted pairwise cosine similarity", add_help=False)
    parser.add_argument("--embeddings_path")
    parser.add_argument("--activations_dir")
    parser.add_argument("--output_subdir")
    parser.add_argument("--device", default="cpu")
    return parser

def main(args):
    # Load embeddings
    embeddings = torch.load(args.embeddings_path, map_location=torch.device(args.device))
    print(f"Loaded embeddings found at {args.embeddings_path}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Load activations
    activations_dataset = ActivationsDataset(args.activations_dir, device=torch.device(args.device), take_every=1)
    activations_dataloader = DataLoader(activations_dataset, batch_size=len(activations_dataset), shuffle=False)
    activations = next(iter(activations_dataloader))
    print(f"Loaded activations found at {args.activations_dir}")
    print(f"Activations shape: {activations.shape}")

    # Scale to 0-1 per neuron
    min_values = activations.min(dim=0, keepdim=True)[0]
    max_values = activations.max(dim=0, keepdim=True)[0]
    activations = (activations - min_values) / (max_values - min_values)

    # embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    num_images, embed_dim = embeddings.shape
    num_neurons = activations.shape[1]

    # Initialize accumulators
    weighted_cosine_similarity_sum = torch.zeros(num_neurons, device=torch.device(args.device))
    weight_sum = torch.zeros(num_neurons, device=torch.device(args.device))
    batch_size = 100  # Set batch size

    for i in tqdm.tqdm(range(num_images), desc="Processing image pairs"):
        for j_start in range(i + 1, num_images, batch_size):  # Process in batches
            j_end = min(j_start + batch_size, num_images)

            embeddings_i = embeddings[i].cuda()  # (embedding_dim)
            embeddings_j = embeddings[j_start:j_end].cuda()  # (batch_size, embedding_dim)
            activations_i = activations[i].cuda()  # (num_neurons)
            activations_j = activations[j_start:j_end].cuda()  # (batch_size, num_neurons)

            # Compute cosine similarity
            cosine_similarities = F.cosine_similarity(
                embeddings_i.unsqueeze(0).expand(j_end - j_start, -1),  # Expanding to (batch_size, embedding_dim)
                embeddings_j,
                dim=1
            )

            # Compute weights and weighted similarities
            # Expanding activations_i to (1, num_neurons)
            weights = activations_i.unsqueeze(0) * activations_j  # (batch_size, num_neurons)
            weighted_cosine_similarities = weights * cosine_similarities.unsqueeze(1)  # (batch_size, num_neurons)

            weighted_cosine_similarities = torch.sum(weighted_cosine_similarities, dim=0)  # (num_neurons)
            weighted_cosine_similarity_sum += weighted_cosine_similarities.cpu()

            weights = torch.sum(weights, dim=0)  # (num_neurons)
            weight_sum += weights.cpu()

    monosemanticity = torch.where(weight_sum != 0, weighted_cosine_similarity_sum / weight_sum, torch.nan)

    os.makedirs(os.path.join(args.activations_dir, args.output_subdir), exist_ok=True)
    torch.save(monosemanticity, os.path.join(args.activations_dir, args.output_subdir, "all_neurons_scores.pth"))

    is_nan = torch.isnan(monosemanticity)
    nan_count = is_nan.sum()
    monosemanticity_mean = torch.mean(monosemanticity[~is_nan])
    monosemanticity_std = torch.std(monosemanticity[~is_nan])

    print(f"Monosemanticity: {monosemanticity_mean.item()} +- {monosemanticity_std.item()}")
    print(f"Dead neurons:", nan_count.item())
    print(f"Total neurons:", num_neurons)

    # Filter out NaNs
    valid_indices = ~torch.isnan(monosemanticity)
    valid_monosemanticity = monosemanticity[valid_indices]
    valid_indices = torch.nonzero(valid_indices).squeeze()

    # Get top 10 highest and lowest monosemantic neurons
    top_10_values, top_10_indices = torch.topk(valid_monosemanticity, 10)
    bottom_10_values, bottom_10_indices = torch.topk(valid_monosemanticity, 10, largest=False)

    # Map indices back to original positions
    top_10_indices = valid_indices[top_10_indices]
    bottom_10_indices = valid_indices[bottom_10_indices]

    # Print results
    print("Top 10 most monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(top_10_indices, top_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item()}")

    print("\nBottom 10 least monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(bottom_10_indices, bottom_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item()}")

    # Save to file
    output_path = os.path.join(args.activations_dir, args.output_subdir, "metric_stats_new.txt")
    with open(output_path, "w") as file:
        file.write(f"Monosemanticity: {monosemanticity_mean.item()} +- {monosemanticity_std.item()}\n")
        file.write(f"Dead neurons: {nan_count.item()}\n")
        file.write(f"Total neurons: {num_neurons}\n\n")

        file.write("Top 10 most monosemantic neurons:\n")
        for idx, val in zip(top_10_indices, top_10_values):
            file.write(f"Neuron {idx.item()} - {val.item()}\n")

        file.write("\nBottom 10 least monosemantic neurons:\n")
        for idx, val in zip(bottom_10_indices, bottom_10_values):
            file.write(f"Neuron {idx.item()} - {val.item()}\n")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
