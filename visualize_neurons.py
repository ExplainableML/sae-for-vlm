import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
from torchvision import transforms
from utils import get_dataset
import argparse
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
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=16)
    parser.add_argument("--dataset_name", default="imagenet", type=str)
    parser.add_argument("--data_path", default="/shared-network/inat2021", type=str)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--visualization_size', type=int, default=224)
    parser.add_argument('--group_fractions', type=float, nargs='+')
    parser.add_argument('--hai_indices_path', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    args.batch_size = 1  # not used
    args.num_workers = 0  # not used

    importants = np.load(args.hai_indices_path)
    print(f"Loaded HAI indices found at {args.hai_indices_path}", flush=True)
    num_neurons = importants.shape[0]

    # Visualize selected images
    def _convert_to_rgb(image):
        return image.convert("RGB")

    visualization_preprocess = transforms.Compose([
        transforms.Resize(size=224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        _convert_to_rgb,
    ])

    ds, dl = get_dataset(args, preprocess=visualization_preprocess, processor=None, split=args.split, subset=1)

    os.makedirs(os.path.join(args.output_dir, 'hai'), exist_ok=True)

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
            output_path = os.path.join(args.output_dir, 'tree', filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()  # Close the plot to free memory

        start_idx = end_idx
