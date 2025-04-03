import numpy as np
import tqdm
import os
import torch
import torch.nn.functional as F
from models.llava import Llava
from utils import IdentitySAE, get_text_model
import argparse
from dictionary_learning.trainers import MatroyshkaBatchTopKSAE
from PIL import Image
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Compute CLIP-based score for steering accuracy")
    parser.add_argument('--hai_indices_path', type=str, required=True)
    parser.add_argument('--embeddings_path', type=str, required=True)
    parser.add_argument('--sae_path', type=str, default=None)
    parser.add_argument('--images_path', type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pre_zero", action=argparse.BooleanOptionalAction)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--neuron_prefix', type=int, default=None)
    parser.add_argument("--steer", action=argparse.BooleanOptionalAction)

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    args.batch_size = 1  # not used
    args.num_workers = 0  # not used

    # Get HAI indices
    hai_indices = torch.from_numpy(np.load(args.hai_indices_path)).to(args.device)
    print(f"Loaded HAI indices found at {args.hai_indices_path}")
    print(f"hai_indices shape: {hai_indices.shape}")

    # Get image embeddings
    embeddings = torch.load(args.embeddings_path).to(args.device)
    print(f"Loaded embeddings found at {args.embeddings_path}")
    print(f"embeddings shape: {embeddings.shape}")

    # Compute mean image embedding of HAI per neuron
    hai_embeddings = embeddings[hai_indices]
    print(f"hai_embeddings shape: {hai_embeddings.shape}")  # (num_neurons, k, embedding_dim)
    hai_embeddings = hai_embeddings.mean(dim=1)
    print(f"hai_embeddings shape: {hai_embeddings.shape}")  # (num_neurons, embedding_dim)

    # Load LLaVA model
    llava = Llava(args.device)
    if args.sae_path:
        sae = MatroyshkaBatchTopKSAE.from_pretrained(args.sae_path).to(args.device)
        print(f"Attached SAE from {args.sae_path}")
    else:
        sae = IdentitySAE()
        print(f"Attached Identity SAE (scoring original neurons)")

    # Filter neurons if prefix is given
    num_neurons = hai_embeddings.shape[0]
    if args.neuron_prefix:
        neuron_indices = list(range(args.neuron_prefix))
    else:
        neuron_indices = list(range(num_neurons))
    print(f"Evaluating on {len(neuron_indices)} neurons")

    # Label images while clamping each neuron
    image_files = [f for f in os.listdir(args.images_path) if f.endswith(('png', 'jpg', 'jpeg', '.JPEG'))]
    print(f"Found {len(image_files)} images in {args.images_path}")
    text = "What is shown in this image? Use exactly one word!"
    labels = {neuron: [] for neuron in neuron_indices}
    if args.steer:
        print("Steering")
    else:
        print("Not steering")
    for neuron in tqdm.tqdm(neuron_indices, desc="Processing neurons"):
        if args.steer:
            llava.attach_and_fix(sae=sae, neurons_to_fix={neuron: 100}, pre_zero=args.pre_zero)
        for image_file in image_files:
            image_path = os.path.join(args.images_path, image_file)
            image = Image.open(image_path)
            label = llava.prompt(text, image, max_tokens=5)[0].split(" ")[0]
            labels[neuron].append((image_file, label))

    # Save labels
    labels_path = os.path.join(os.path.dirname(args.output_path), "labels.txt")
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(labels_path, "w") as f:
        for neuron, image_labels in labels.items():
            for image_file, label in image_labels:
                f.write(f"{neuron},{image_file},{label}\n")
    print(f"Labels saved to {labels_path}")

    # Compute text embeddings for neuron-clamped LLaVA labels
    text_encoder, tokenizer = get_text_model(args)
    label_embeddings = torch.zeros(len(neuron_indices), len(image_files), hai_embeddings.shape[1]).to(args.device)
    for i, (neuron, image_labels) in tqdm.tqdm(enumerate(labels.items()), desc="Computing text embeddings"):
        for j, label in enumerate(image_labels):
            with torch.no_grad():
                inputs = tokenizer([label[1]], padding=True, return_tensors="pt").to(args.device)
                outputs = text_encoder(**inputs)
                label_embeddings[i, j] = outputs.text_embeds

    # Compute cosine similarities
    cosine_similarities = []
    for i in range(label_embeddings.shape[1]):
        cosine_similarities.append(F.cosine_similarity(hai_embeddings[neuron_indices], label_embeddings[:, i], dim=1))
    cosine_similarities = torch.cat(cosine_similarities)
    print(cosine_similarities.shape)
    torch.save(cosine_similarities, os.path.join(os.path.dirname(args.output_path), "scores_per_neuron"))
    mean_cosine_similarity = cosine_similarities.mean().item()
    std_cosine_similarity = cosine_similarities.std().item()

    print("Mean Cosine Similarity:", mean_cosine_similarity)
    print("Standard Deviation Cosine Similarity:", std_cosine_similarity)

    # Save results
    with open(os.path.join(os.path.dirname(args.output_path), "metric.txt"), "w") as f:
        f.write(f"Mean Cosine Similarity: {mean_cosine_similarity}\n")
        f.write(f"Standard Deviation Cosine Similarity: {std_cosine_similarity}\n")

print("Done")