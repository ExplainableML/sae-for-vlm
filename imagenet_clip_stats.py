import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path
from datasets.activations import ActivationsDataset

from torch.utils.data import DataLoader, Subset
import tqdm
import matplotlib.pyplot as plt
from utils import get_dataset, get_model
import torch.nn.functional as F

def get_args_parser():
    parser = argparse.ArgumentParser("Get clip stats", add_help=False)
    parser.add_argument("--embeddings_path")
    parser.add_argument("--device", default="cpu")
    return parser

def main(args):
    # Load embeddings
    # embeddings = torch.load(args.embeddings_path, map_location=torch.device(args.device))
    embeddings = torch.load("embeddings_dir/imagenet_val_embeddings_clip-vit-base-patch32.pt", map_location=torch.device('cuda'))
    print(f"Embeddings shape: {embeddings.shape}")
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute cosine similarity using einsum (efficient)
    dot_products = torch.einsum('ik,jk->ij', embeddings, embeddings)
    print(dot_products.shape)
    print(torch.max(dot_products).item())
    print(torch.min(dot_products).item())
    print(torch.mean(dot_products).item())

# >>> print(torch.max(dot_products).item())
# 1.0000015497207642
# >>> print(torch.min(dot_products).item())
# -0.016243668273091316
# >>> print(torch.mean(dot_products).item())
# 0.4831811189651489
