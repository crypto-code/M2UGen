#!/usr/bin/bash

LLAMA_PATH="$1"
OUTPUT_DIR="$2"

mkdir -p "$OUTPUT_DIR"

accelerate launch main_train.py --batch_size 1 --accum_iter 1 --stage 1 --music_decoder musicgen \
 --epochs 5 --split_epoch 1 --warmup_epochs 0 --lr 1e-4 --min_lr 1e-6 --weight_decay 0.05 \
 --mert_path m-a-p/MERT-v1-330M --vit_path google/vit-base-patch16-224 \
 --vivit_path google/vivit-b-16x2-kinetics400 --music_decoder_path facebook/musicgen-small \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR"

accelerate launch main_train.py --batch_size 1 --accum_iter 1 --stage 2 --music_decoder musicgen \
 --epochs 5 --split_epoch 1 --warmup_epochs 0 --lr 1e-4 --min_lr 1e-6 --weight_decay 0.05 \
 --mert_path m-a-p/MERT-v1-330M --vit_path google/vit-base-patch16-224 \
 --vivit_path google/vivit-b-16x2-kinetics400 --music_decoder_path facebook/musicgen-small \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR"

 accelerate launch main_train.py --batch_size 1 --accum_iter 1 --stage 3 --music_decoder musicgen \
 --epochs 5 --split_epoch 1 --warmup_epochs 0 --lr 1e-4 --min_lr 1e-6 --weight_decay 0.05 \
 --mert_path m-a-p/MERT-v1-330M --vit_path google/vit-base-patch16-224 \
 --vivit_path google/vivit-b-16x2-kinetics400 --music_decoder_path facebook/musicgen-small \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR"

