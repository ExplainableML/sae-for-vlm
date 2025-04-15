#!/bin/bash

DATASET_PATH="${INAT_PATH}"

# 1. Save original activations
for SPLIT in "train" "val"; do
  python save_activations.py \
    --batch_size 32 \
    --model_name "clip-vit-large-patch14-336" \
    --attachment_point "post_projection" \
    --layer "-1" \
    --dataset_name "inat" \
    --split "${SPLIT}" \
    --data_path "${DATASET_PATH}" \
    --num_workers 8 \
    --output_dir "./activations_dir/raw/inat_${SPLIT}_activations_clip-vit-large-patch14-336_-1_post_projection" \
    --cls_only \
    --save_every 100
done

# 2. Train SAE
python sae_train.py \
  --sae_model "matroyshka_batch_top_k" \
  --activations_dir "activations_dir/raw/inat_train_activations_clip-vit-large-patch14-336_-1_post_projection" \
  --val_activations_dir "activations_dir/raw/inat_val_activations_clip-vit-large-patch14-336_-1_post_projection" \
  --checkpoints_dir "checkpoints_dir/matroyshka_batch_top_k_20_x2" \
  --expansion_factor 2 \
  --steps 110000 \
  --save_steps 20000 \
  --log_steps 10000 \
  --batch_size 4096 \
  --k 20 \
  --auxk_alpha 0.03 \
  --decay_start 109999 \
  --group_fractions 0.002 0.009 0.035 0.189 0.765

# 3. Save SAE activations
python save_activations.py \
  --batch_size 32 \
  --model_name "clip-vit-large-patch14-336" \
  --attachment_point "post_projection" \
  --layer "-1" \
  --dataset_name "inat" \
  --split "train" \
  --data_path "${DATASET_PATH}" \
  --num_workers 8 \
  --output_dir "./activations_dir/matroyshka_batch_top_k_20_x2/inat_train_activations_clip-vit-large-patch14-336_-1_post_projection" \
  --cls_only \
  --save_every 100 \
  --sae_model "matroyshka_batch_top_k" \
  --sae_path "./checkpoints_dir/matroyshka_batch_top_k_20_x2/inat_train_activations_clip-vit-large-patch14-336_-1_post_projection_matroyshka_batch_top_k_20_x2/trainer_0/checkpoints/ae_100000.pt"

# 4. Compute LCA depth per level
python find_hai_indices.py \
  --activations_dir "./activations_dir/matroyshka_batch_top_k_20_x2/inat_train_activations_clip-vit-large-patch14-336_-1_post_projection" \
  --dataset_name "inat" \
  --data_path "${DATASET_PATH}" \
  --split "train" \
  --k 16 \
  --chunk_size 1000

python inat_depth.py \
  --activations_dir "./activations_dir/matroyshka_batch_top_k_20_x2/inat_train_activations_clip-vit-large-patch14-336_-1_post_projection" \
  --hai_indices_path "./activations_dir/matroyshka_batch_top_k_20_x2/inat_train_activations_clip-vit-large-patch14-336_-1_post_projection/hai_indices_16.npy" \
  --data_path "${DATASET_PATH}" \
  --split "train" \
  --k 16 \
  --group_fractions 0.002 0.009 0.035 0.189 0.765
