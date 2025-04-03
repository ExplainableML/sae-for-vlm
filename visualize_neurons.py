import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tqdm
import os
import torch
from torchvision import transforms
from utils import get_dataset
import argparse
from datasets.activations import ActivationsDataset, ChunkedActivationsDataset
from torch.utils.data import DataLoader, Subset
from math import isclose


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols, "Number of images must match rows * cols."
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize top-k activating images for neurons.")
    parser.add_argument('--activations_dir', type=str, required=True)
    parser.add_argument('--subset', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=16)
    parser.add_argument("--dataset_name", default="imagenet", type=str)
    parser.add_argument("--data_path", default="/shared-network/inat2021", type=str)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--visualization_size', type=int, default=224)
    parser.add_argument('--mode', type=str, default='ends')
    parser.add_argument('--group_fractions', type=float, nargs='+')
    parser.add_argument('--hai_indices_path', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    args.batch_size = 1  # not used
    args.num_workers = 0  # not used

    # Select highest activating images
    print(f"Loading activations from {args.activations_dir}")

    importants = np.load(args.hai_indices_path)
    print(f"Loaded HAI indices found at {args.hai_indices_path}", flush=True)
    num_neurons = importants.shape[0]

    # all-at-once version
    # activations_dataset = ActivationsDataset(args.activations_dir, device=torch.device("cpu"))
    # activations_dataloader = DataLoader(activations_dataset, batch_size=len(activations_dataset), shuffle=False)
    # activations = next(iter(activations_dataloader)).numpy()
    # activations = activations[::int(1.0 / args.subset), :]
    # importants, aliveness = [], []
    #
    # for neuron in tqdm.trange(activations.shape[1]):
    #     neuron_activations = activations[:, neuron]
    #     important = np.argsort(neuron_activations)[-args.top_k:]
    #     aliveness.append(neuron_activations.sum())
    #     importants.append(important)

    # chunked version
    # importants, aliveness = [], []
    # chunk_size = 1000
    # activations_dataset = ChunkedActivationsDataset(args.activations_dir, device=torch.device("cpu"))
    # activations_dataloader = DataLoader(activations_dataset, batch_size=chunk_size, shuffle=False)
    # num_samples = len(activations_dataset)
    # num_neurons = next(iter(activations_dataloader)).shape[1]
    # print(f"{num_samples // chunk_size} chunks to process...", flush=True)
    # pbar = tqdm.tqdm(list(range(num_neurons // chunk_size)))
    # for i in pbar:
    #     neuron_start = i * chunk_size
    #     neuron_end = min((i + 1) * chunk_size, num_neurons)
    #     activations_chunks = np.zeros((num_samples, neuron_end - neuron_start))
    #     for j, activations_chunk in enumerate(activations_dataloader):
    #         sample_start = j * chunk_size
    #         # sample_end = (j + 1) * chunk_size
    #         sample_end = min((j + 1) * chunk_size, num_samples)
    #         activations_chunk = activations_chunk.numpy()
    #         activations_chunks[sample_start:sample_end, :] = activations_chunk[:, neuron_start:neuron_end]
    #
    #     for neuron in range(neuron_end - neuron_start):
    #         neuron_activations = activations_chunks[:, neuron]
    #         important = np.argsort(neuron_activations)[-args.top_k:]
    #         aliveness.append(neuron_activations.sum())
    #         importants.append(important)

        # uncomment to visualize only some of the neurons
        # break

    # Visualize selected images
    def _convert_to_rgb(image):
        return image.convert("RGB")

    visualization_preprocess = transforms.Compose([
        transforms.Resize(size=224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        _convert_to_rgb,
    ])

    ds, dl = get_dataset(args, preprocess=visualization_preprocess, processor=None, split=args.split, subset=args.subset)

    os.makedirs(os.path.join(args.activations_dir, 'hai'), exist_ok=True)

    if args.mode == "ends":
        for end in ["bot", "top"]:
            with open(os.path.join(args.activations_dir, f"{end}_neurons.txt")) as f:
                neurons = [int(x) for x in f if x.strip()]

            for i, neuron_id in enumerate(neurons):
                print(f"Visualizing neuron {neuron_id}")
                important = importants[neuron_id]
                images = [ds[i][0] for i in important]
                s = int(np.sqrt(args.top_k))
                grid_image = image_grid(images[::-1], rows=s, cols=s)
                plt.imshow(grid_image)
                plt.axis('off')
                plt.savefig(os.path.join(args.activations_dir, 'hai', f'{end}{i}.png'), bbox_inches='tight', pad_inches=0)
    elif args.mode == "tree":
        assert isclose(sum(args.group_fractions), 1.0), "group_fractions must sum to 1.0"
        group_sizes = [int(f * num_neurons) for f in args.group_fractions[:-1]]
        group_sizes.append(num_neurons - sum(group_sizes))

        start_idx = 0
        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size
            group_neurons = range(start_idx, end_idx)

            for neuron_id, absolute_id in enumerate(group_neurons[:5000]):
                print(f"Visualizing neuron {neuron_id} (absolute {absolute_id}) in group {group_idx}", flush=True)

                important = importants[absolute_id]
                images = [ds[i][0] for i in important]
                s = int(np.sqrt(args.top_k))
                grid_image = image_grid(images[::-1], rows=s, cols=s)

                plt.imshow(grid_image)
                plt.axis('off')
                filename = f"group_{group_idx}_neuron_{neuron_id}_absolute_{absolute_id}.png"
                output_path = os.path.join(args.activations_dir, 'tree', filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close()  # Close the plot to free memory

            start_idx = end_idx
