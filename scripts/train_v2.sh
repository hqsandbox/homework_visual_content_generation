#!/bin/bash
# ============================================================
# 备选方案: 使用 accelerate + train_lora_w_ti.py 训练
# 适合需要更细粒度控制的场景
# ============================================================

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/processed"
export OUTPUT_DIR="./outputs/lora_output_v2"

# ---------- 训练参数 ----------
# learnable_property: "style" 或 "object"
LEARNABLE_PROPERTY="style"

# placeholder_token: 用于代表学到的概念
PLACEHOLDER_TOKEN="<my_style>"

# initializer_token: 初始化 token, 用一个语义接近的词
# 风格: 可以用 "painting", "art", "illustration" 等
# 角色: 可以用 "person", "woman", "man", "character" 等
INITIALIZER_TOKEN="art"

echo "============================================"
echo "  LoRA Training (accelerate version)"
echo "============================================"

cd lora/training_scripts

accelerate launch train_lora_w_ti.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=../../$INSTANCE_DIR \
  --output_dir=../../$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=2000 \
  --placeholder_token="$PLACEHOLDER_TOKEN" \
  --learnable_property="$LEARNABLE_PROPERTY" \
  --initializer_token="$INITIALIZER_TOKEN" \
  --save_steps=500 \
  --unfreeze_lora_step=500 \
  --lora_rank=4 \
  --seed=42 \
  --resize=True

cd ../..

echo ""
echo "训练完成! 模型保存在: $OUTPUT_DIR"
