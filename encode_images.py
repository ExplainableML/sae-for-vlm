import torch
import os.path
import argparse
import tqdm
from utils import get_dataset, get_model

def get_args_parser():
    parser = argparse.ArgumentParser("Encode images", add_help=False)
    parser.add_argument("--embeddings_path")
    parser.add_argument("--model_name", default="clip", type=str)
    parser.add_argument("--dataset_name", default="imagenet", type=str)
    parser.add_argument("--data_path", default="/shared-network/inat2021", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    if os.path.exists(args.embeddings_path):
        print(f"Embeddings already saved at {args.embeddings_path}")
    else:
        model, processor = get_model(args)
        ds, dl = get_dataset(args, preprocess=None, processor=processor, split=args.split)

        embeddings = []
        pbar = tqdm.tqdm(dl)
        for image in pbar:
            with torch.no_grad():
                output = model.encode(image)
                embeddings.append(output.detach().cpu())

        embeddings = torch.cat(embeddings, dim=0)
        os.makedirs(os.path.dirname(args.embeddings_path), exist_ok=True)
        torch.save(embeddings, args.embeddings_path)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Saved embeddings to {args.embeddings_path}")

