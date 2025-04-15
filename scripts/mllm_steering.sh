#!/bin/bash

DATASET_PATH="${IMAGENET_PATH}"

# 1. Save original activations
for SPLIT in "train" "val"; do
  python save_activations.py \
    --batch_size 32 \
    --model_name "clip-vit-large-patch14-336" \
    --attachment_point "post_mlp_residual" \
    --layer 22 \
    --dataset_name "imagenet" \
    --split "${SPLIT}" \
    --data_path "${DATASET_PATH}" \
    --num_workers 8 \
    --output_dir "./activations_dir/raw/random_k_2/imagenet_${SPLIT}_activations_clip-vit-large-patch14-336_22_post_mlp_residual" \
    --random_k 2 \
    --save_every 100
done

# 2. Train SAE
python sae_train.py \
  --sae_model "matroyshka_batch_top_k" \
  --activations_dir "activations_dir/raw/random_k_2/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual" \
  --val_activations_dir "activations_dir/raw/random_k_2/imagenet_val_activations_clip-vit-large-patch14-336_22_post_mlp_residual" \
  --checkpoints_dir "checkpoints_dir/matroyshka_batch_top_k_20_x64/random_k_2/" \
  --expansion_factor 64 \
  --steps 110000 \
  --save_steps 20000 \
  --log_steps 10000 \
  --batch_size 4096 \
  --k 20 \
  --auxk_alpha 0.03 \
  --decay_start 109999 \
  --group_fractions 0.0625 0.125 0.25 0.5625

# 3. Save SAE activations
python save_activations.py \
  --batch_size 32 \
  --model_name "clip-vit-large-patch14-336" \
  --attachment_point "post_mlp_residual" \
  --layer 22 \
  --dataset_name "imagenet" \
  --split "train" \
  --data_path "${DATASET_PATH}" \
  --num_workers 8 \
  --output_dir "./activations_dir/matroyshka_batch_top_k_20_x64/mean_pool/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual" \
  --mean_pool \
  --save_every 100 \
  --sae_model "matroyshka_batch_top_k" \
  --sae_path "./checkpoints_dir/matroyshka_batch_top_k_20_x64/random_k_2/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual_matroyshka_batch_top_k_20_x64/trainer_0/checkpoints/ae_100000.pt"

# 4. Visualize neurons
python find_hai_indices.py \
  --activations_dir "./activations_dir/matroyshka_batch_top_k_20_x64/mean_pool/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual" \
  --dataset_name "imagenet" \
  --data_path "${DATASET_PATH}" \
  --split "train" \
  --k 16 \
  --chunk_size 1000

python visualize_neurons.py \
  --output_dir "./activations_dir/matroyshka_batch_top_k_20_x64/mean_pool/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual" \
  --top_k 16 \
  --dataset_name "imagenet" \
  --data_path "${DATASET_PATH}" \
  --split "train" \
  --group_fractions 0.0625 0.125 0.25 0.5625 \
  --hai_indices_path "./activations_dir/matroyshka_batch_top_k_20_x64/mean_pool/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual/hai_indices_16.npy"

# 5. Compute steering score
python encode_images.py \
  --embeddings_path "embeddings_dir/imagenet_train_embeddings_clip-vit-base-patch32.pt" \
  --model_name "clip-vit-base-patch32" \
  --dataset_name "imagenet" \
  --split "train" \
  --data_path "${DATASET_PATH}" \
  --batch_size 128

python imagenet_subset.py \
  --imagenet_root "${DATASET_PATH}" \
  --output_dir "./images_imagenet"

# no steering (1000 images, 10 neurons)
python steering_score.py \
  --hai_indices_path "./activations_dir/matroyshka_batch_top_k_20_x64/mean_pool/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual/hai_indices_16.npy" \
  --embeddings_path "./embeddings_dir/imagenet_train_embeddings_clip-vit-base-patch32.pt" \
  --sae_path "./checkpoints_dir/matroyshka_batch_top_k_20_x64/random_k_2/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual_matroyshka_batch_top_k_20_x64/trainer_0/checkpoints/ae_100000.pt" \
  --images_path "./images_imagenet/" \
  --no-pre_zero \
  --model_name "clip-vit-base-patch32" \
  --neuron_prefix 10 \
  --no-steer \
  --output_path "./llava_results_dir/1000/no_steering/"

# steering (1000 images, 10 neurons)
python steering_score.py \
  --hai_indices_path "./activations_dir/matroyshka_batch_top_k_20_x64/mean_pool/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual/hai_indices_16.npy" \
  --embeddings_path "./embeddings_dir/imagenet_train_embeddings_clip-vit-base-patch32.pt" \
  --sae_path "./checkpoints_dir/matroyshka_batch_top_k_20_x64/random_k_2/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual_matroyshka_batch_top_k_20_x64/trainer_0/checkpoints/ae_100000.pt" \
  --images_path "./images_imagenet/" \
  --no-pre_zero \
  --model_name "clip-vit-base-patch32" \
  --neuron_prefix 10 \
  --steer \
  --output_path "./llava_results_dir/1000/steering/"

# steering (1 image, 1000 neurons)
python steering_score.py \
  --hai_indices_path "./activations_dir/matroyshka_batch_top_k_20_x64/mean_pool/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual/hai_indices_16.npy" \
  --embeddings_path "./embeddings_dir/imagenet_train_embeddings_clip-vit-base-patch32.pt" \
  --sae_path "./checkpoints_dir/matroyshka_batch_top_k_20_x64/random_k_2/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual_matroyshka_batch_top_k_20_x64/trainer_0/checkpoints/ae_100000.pt" \
  --images_path "./images/" \
  --no-pre_zero \
  --model_name "clip-vit-base-patch32" \
  --neuron_prefix 1000 \
  --no-steer \
  --output_path "./llava_results_dir/1/no_steering/"

python steering_score.py \
  --hai_indices_path "./activations_dir/matroyshka_batch_top_k_20_x64/mean_pool/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual/hai_indices_16.npy" \
  --embeddings_path "./embeddings_dir/imagenet_train_embeddings_clip-vit-base-patch32.pt" \
  --sae_path "./checkpoints_dir/matroyshka_batch_top_k_20_x64/random_k_2/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual_matroyshka_batch_top_k_20_x64/trainer_0/checkpoints/ae_100000.pt" \
  --images_path "./images/" \
  --no-pre_zero \
  --model_name "clip-vit-base-patch32" \
  --neuron_prefix 1000 \
  --steer \
  --output_path "./llava_results_dir/1/steering/"

# 6. Compute baseline scores
python similarity_baseline.py

# 7. Finding qualitative examples (e.g. steering pencil neuron)
python steering_qualitative.py
