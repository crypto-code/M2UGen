#!/usr/bin/bash

LLAMA_PATH="$1"
OUTPUT_DIR="$2"

mkdir -p "$OUTPUT_DIR"

 python3 -u -m torch.distributed.launch --master_port=1114 --nproc_per_node=1 --use_env \
 main_train.py --batch_size 1 --accum_iter 4 --stage 1 --music_decoder audioldm2 \
 --epochs 5 --split_epoch 1 --warmup_epochs 0 --lr 1e-4 --min_lr 1e-6 --weight_decay 0.05 \
 --mert_path m-a-p/MERT-v1-330M --vit_path google/vit-base-patch16-224 \
 --vivit_path google/vivit-b-16x2 --music_decoder_path cvssp/audioldm2-music \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR"

 python3 -u -m torch.distributed.launch --master_port=1114 --nproc_per_node=1 --use_env \
  main_train.py --batch_size 1 --accum_iter 4 --stage 2 --music_decoder audioldm2 \
  --epochs 5 --split_epoch 1 --warmup_epochs 0 --lr 1e-4 --min_lr 1e-6 --weight_decay 0.05 \
 --mert_path m-a-p/MERT-v1-330M --vit_path google/vit-base-patch16-224 \
 --vivit_path google/vivit-b-16x2 --music_decoder_path cvssp/audioldm2-music \
  --llama_path "$LLAMA_PATH" \
  --output_dir "$OUTPUT_DIR"

 python3 -u -m torch.distributed.launch --master_port=1114 --nproc_per_node=1 --use_env \
 main_train.py --batch_size 1 --accum_iter 4 --stage 3 --music_decoder audioldm2 \
 --epochs 2 --split_epoch 1 --warmup_epochs 0 --lr 1e-4 --min_lr 1e-6 --weight_decay 0.05 \
 --mert_path m-a-p/MERT-v1-330M --vit_path google/vit-base-patch16-224 \
 --vivit_path google/vivit-b-16x2 --music_decoder_path cvssp/audioldm2-music \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR"
