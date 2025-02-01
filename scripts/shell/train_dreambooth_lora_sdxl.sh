#!/bin/bash

# Get script dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Training script
TRAINING_SCRIPT="$SCRIPT_DIR/../training/train_dreambooth_lora_sdxl.py"
# Dir containing training images
INSTANCE_DATA_DIR="$SCRIPT_DIR/../../images_training/ava"
# Output model directory
OUTPUT_DIR="$SCRIPT_DIR/../../lora_output/ava"

# Activate virtual environment (if applicable)
source "$SCRIPT_DIR/../venv/bin/activate" || echo "No virtual environment found. Skipping..."

# Run script with params
python "$TRAINING_SCRIPT" \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="$INSTANCE_DATA_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --mixed_precision="no" \
  --optimizer="AdamW" \
  --instance_prompt="a photo of sks ava" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=200 \
  --validation_epochs=25 \
  --seed="0"
