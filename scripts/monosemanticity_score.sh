#!/bin/bash

DATASET="imagenet"
DATASET_PATH="${IMAGENET_PATH}"
LAYERS=("-1" 11 17 22 23)
MODEL_NAME="clip-vit-large-patch14-336"  # "siglip-so400m-patch14-384"
EXPANSION_FACTORS=(64 16 8 4 2 1)
VISION_ENCODER="clip-vit-base-patch32"  # "dinov2-base"

# 1. Save original activations
for LAYER in "${LAYERS[@]}"; do
  if [ "${LAYER}" == "-1" ]; then
    POINT="post_projection"
  else
    POINT="post_mlp_residual"
  fi
  for SPLIT in "train" "val"; do
    python save_activations.py \
      --batch_size 32 \
      --model_name "${MODEL_NAME}" \
      --attachment_point "${POINT}" \
      --layer "${LAYER}" \
      --dataset_name "${DATASET}" \
      --split "${SPLIT}" \
      --data_path "${DATASET_PATH}" \
      --num_workers 8 \
      --output_dir "./activations_dir/raw/${DATASET}_${SPLIT}_activations_${MODEL_NAME}_${LAYER}_${POINT}" \
      --cls_only \
      --save_every 100
  done
done

# 2. Train SAE
for LAYER in "${LAYERS[@]}"; do
  if [ "${LAYER}" == "-1" ]; then
    POINT="post_projection"
  else
    POINT="post_mlp_residual"
  fi
  for EXPANSION_FACTOR in "${EXPANSION_FACTORS[@]}"; do
    python sae_train.py \
      --sae_model "matroyshka_batch_top_k" \
      --activations_dir "activations_dir/raw/${DATASET}_train_activations_${MODEL_NAME}_${LAYER}_${POINT}" \
      --val_activations_dir "activations_dir/raw/${DATASET}_val_activations_${MODEL_NAME}_${LAYER}_${POINT}" \
      --checkpoints_dir "checkpoints_dir/matroyshka_batch_top_k_20_x${EXPANSION_FACTOR}" \
      --expansion_factor "${EXPANSION_FACTOR}" \
      --steps 110000 \
      --save_steps 20000 \
      --log_steps 10000 \
      --batch_size 4096 \
      --k 20 \
      --auxk_alpha 0.03 \
      --decay_start 109999 \
      --group_fractions 0.0625 0.125 0.25 0.5625

    python sae_train.py \
      --sae_model "batch_top_k" \
      --activations_dir "activations_dir/raw/${DATASET}_train_activations_${MODEL_NAME}_${LAYER}_${POINT}" \
      --val_activations_dir "activations_dir/raw/${DATASET}_val_activations_${MODEL_NAME}_${LAYER}_${POINT}" \
      --checkpoints_dir "checkpoints_dir/batch_top_k_20_x${EXPANSION_FACTOR}" \
      --expansion_factor "${EXPANSION_FACTOR}" \
      --steps 110000 \
      --save_steps 20000 \
      --log_steps 10000 \
      --batch_size 4096 \
      --k 20 \
      --auxk_alpha 0.03 \
      --decay_start 109999
  done
done

# 3. Save SAE activations
for LAYER in "${LAYERS[@]}"; do
  if [ "${LAYER}" == "-1" ]; then
    POINT="post_projection"
  else
    POINT="post_mlp_residual"
  fi
  for EXPANSION_FACTOR in "${EXPANSION_FACTORS[@]}"; do
    for SAE_MODEL in "matroyshka_batch_top_k" "batch_top_k"; do
      python save_activations.py \
        --batch_size 32 \
        --model_name "${MODEL_NAME}" \
        --attachment_point "${POINT}" \
        --layer "${LAYER}" \
        --dataset_name "${DATASET}" \
        --split "val" \
        --data_path "${DATASET_PATH}" \
        --num_workers 8 \
        --output_dir "./activations_dir/${SAE_MODEL}_20_x${EXPANSION_FACTOR}/${DATASET}_val_activations_${MODEL_NAME}_${LAYER}_${POINT}" \
        --cls_only \
        --save_every 100 \
        --sae_model "${SAE_MODEL}" \
        --sae_path "./checkpoints_dir/${SAE_MODEL}_20_x${EXPANSION_FACTOR}/${DATASET}_train_activations_${MODEL_NAME}_${LAYER}_${POINT}_${SAE_MODEL}_20_x${EXPANSION_FACTOR}/trainer_0/checkpoints/ae_100000.pt"
    done
  done
done

# 4. Save vision encoder embeddings
python encode_images.py \
  --embeddings_path "embeddings_dir/${DATASET}_val_embeddings_${VISION_ENCODER}.pt" \
  --model_name "${VISION_ENCODER}" \
  --dataset_name "${DATASET}" \
  --split "val" \
  --data_path "${DATASET_PATH}" \
  --batch_size 128

# 5. Compute Monosemanticity Score
for LAYER in "${LAYERS[@]}"; do
  if [ "${LAYER}" == "-1" ]; then
    POINT="post_projection"
  else
    POINT="post_mlp_residual"
  fi
  # SAE neurons
  for EXPANSION_FACTOR in "${EXPANSION_FACTORS[@]}"; do
    for SAE_MODEL in "matroyshka_batch_top_k" "batch_top_k"; do
      python metric.py \
        --activations_dir "activations_dir/${SAE_MODEL}_20_x${EXPANSION_FACTOR}/${DATASET}_val_activations_${MODEL_NAME}_${LAYER}_${POINT}" \
        --embeddings_path ${EMBEDDINGS_PATH} \
        --output_subdir "ms_${VISION_ENCODER}"
    done
  done
  # original neurons
  python metric.py \
    --activations_dir "activations_dir/raw/${DATASET}_val_activations_${MODEL_NAME}_${LAYER}_${POINT}" \
    --embeddings_path ${EMBEDDINGS_PATH} \
    --output_subdir "ms_${VISION_ENCODER}"
done

# 6. Visualize neurons (of selected SAE, using training set)
python save_activations.py \
  --batch_size 32 \
  --model_name "${MODEL_NAME}" \
  --attachment_point "post_projection" \
  --layer "-1" \
  --dataset_name "${DATASET}" \
  --split "train" \
  --data_path "${DATASET_PATH}" \
  --num_workers 8 \
  --output_dir "./activations_dir/matroyshka_batch_top_k_20_x4/${DATASET}_train_activations_${MODEL_NAME}_-1_post_projection" \
  --cls_only \
  --save_every 100 \
  --sae_model "matroyshka_batch_top_k" \
  --sae_path "./checkpoints_dir/matroyshka_batch_top_k_20_x4/${DATASET}_train_activations_${MODEL_NAME}_-1_post_projection_matroyshka_batch_top_k_20_x4/trainer_0/checkpoints/ae_100000.pt"

python find_hai_indices.py \
  --activations_dir "./activations_dir/matroyshka_batch_top_k_20_x4/${DATASET}_train_activations_${MODEL_NAME}_-1_post_projection" \
  --dataset_name "${DATASET}" \
  --data_path "${DATASET_PATH}" \
  --split "train" \
  --k 16 \
  --chunk_size 1000

python visualize_neurons.py \
  --output_dir "./activations_dir/matroyshka_batch_top_k_20_x4/${DATASET}_train_activations_${MODEL_NAME}_-1_post_projection" \
  --top_k 16 \
  --dataset_name "${DATASET}" \
  --data_path "${DATASET_PATH}" \
  --split "train" \
  --group_fractions 0.0625 0.125 0.25 0.5625 \
  --hai_indices_path "./activations_dir/matroyshka_batch_top_k_20_x4/${DATASET}_train_activations_${MODEL_NAME}_-1_post_projection/hai_indices_16.npy"
