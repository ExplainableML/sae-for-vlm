import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import tqdm
import argparse
from pathlib import Path
from torchvision.datasets import ImageFolder
from utils import get_dataset, get_model
from torchvision.transforms import ToTensor
from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import BatchTopKSAE, MatroyshkaBatchTopKSAE


def get_args_parser():
    parser = argparse.ArgumentParser("Save activations used to train SAE", add_help=False)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--sae_model", default=None, type=str)
    parser.add_argument("--model_name", default="clip", type=str)
    parser.add_argument("--attachment_point", default="post_mlp_residual", type=str)
    parser.add_argument("--layer", default=-1, type=int)
    parser.add_argument("--sae_path", default=None, type=str)
    parser.add_argument("--dataset_name", default="imagenet", type=str)
    parser.add_argument("--data_path", default="/shared-network/inat2021", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--output_dir", default="./output_dir", type=str)
    parser.add_argument("--cls_only", default=False, action="store_true")
    parser.add_argument("--mean_pool", default=False, action="store_true")
    parser.add_argument("--take_every", default=1, type=int)
    parser.add_argument("--random_k", default=-1, type=int)
    parser.add_argument("--save_every", default=50_000, type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser

def save_activations(activations, count, split, save_count, args):
    activations_tensor = torch.cat(activations, dim=0)
    if args.take_every > 1:
        # Pick every n-th activation in the batch
        activations_tensor = activations_tensor[::args.take_every, :, :]

    if args.layer == -1:
        # Tokens already pooled
        activations_tensor = activations_tensor
    elif args.cls_only:
        # Keep only CLS token
        activations_tensor =  activations_tensor[:, 0, :]
    elif args.mean_pool:
        # Mean pool tokens into one data point
        activations_tensor =  torch.mean(activations_tensor, dim=1)
    elif args.random_k != -1:
        # Treat each token as a separate data point but pick random k tokens from each image
        batch_size, seq_len, hidden_dim = activations_tensor.shape
        indices = torch.randint(0, seq_len, (batch_size, args.random_k))
        activations_tensor = torch.stack([activations_tensor[i, indices[i], :] for i in range(batch_size)])
        activations_tensor = activations_tensor.reshape(-1, hidden_dim)
    else:
        # Treat each token as a separate data point and use all the tokens
        activations_tensor = activations_tensor.reshape(activations_tensor.shape[0] * activations_tensor.shape[1],
                                                        activations_tensor.shape[2])

    filename = f"{args.dataset_name}_{split}_activations_{args.model_name}_{args.layer}_{args.attachment_point}_part{save_count + 1}.pt"
    save_path = os.path.join(args.output_dir, filename)
    torch.save(torch.tensor(activations_tensor.cpu().numpy()), save_path)
    print(f"Saved the activations at count {count} to {save_path}")

def collect_activations(args):
    model, processor = get_model(args)

    if args.sae_model is not None:
        if args.sae_model == "standard":
            sae = AutoEncoder.from_pretrained(args.sae_path).to(args.device)
        if args.sae_model == "batch_top_k":
            sae = BatchTopKSAE.from_pretrained(args.sae_path).to(args.device)
        if args.sae_model == "matroyshka_batch_top_k":
            sae = MatroyshkaBatchTopKSAE.from_pretrained(args.sae_path).to(args.device)
        print(f"Attached SAE from {args.sae_path}")
    else:
        sae = None
        print(f"No SAE attached. Saving original activations")

    model.attach(args.attachment_point, args.layer, sae=sae)

    ds, dl = get_dataset(args, preprocess=None, processor=processor, split=args.split)
    activations = []
    count = 0
    save_count = 0
    pbar = tqdm.tqdm(dl)
    for image in pbar:

        with torch.no_grad():
            model.encode(image)
            activations.extend(model.register[f"{args.attachment_point}_{args.layer}"])

        count += image['pixel_values'].shape[0]
        pbar.set_postfix({'Processed data points': count})

        if count >= args.save_every * (save_count + 1):
            save_activations(activations, count, args.split, save_count, args)
            activations = []
            save_count += 1

    if activations:
        save_activations(activations, count, args.split, save_count, args)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    collect_activations(args)