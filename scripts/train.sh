#!/bin/bash
# ============================================================
# LoRA + Pivotal Tuning Inversion 训练脚本
# 使用 lora_pti CLI 进行训练 (推荐方式，效果最好)
# ============================================================

# ---------- 基本配置 (请根据实际情况修改) ----------
# 基础模型: Stable Diffusion v1.5
export MODEL_NAME="runwayml/stable-diffusion-v1-5"

# 训练数据目录 (放置预处理好的 512x512 图片)
export INSTANCE_DIR="./data/processed"

# 输出目录
export OUTPUT_DIR="./outputs/lora_output"

# ---------- 训练参数 ----------
# use_template: "style" 用于风格学习, "object" 用于物体/角色学习
# 如果你训练的是某种画风 (如动漫风格、水彩风格), 用 "style"
# 如果你训练的是某个具体角色/物体, 用 "object"
USE_TEMPLATE="object"

# placeholder_tokens: 用于表示学到的概念的特殊 token
# 使用 <s1>|<s2> 两个 token 效果更好
PLACEHOLDER_TOKENS="<s1>|<s2>"

# lora_rank: LoRA 的秩, 越大模型表达力越强但文件越大
# 推荐: 1-4 用于风格, 4-8 用于角色
LORA_RANK=8

# 训练步数
MAX_TRAIN_STEPS_TI=1000    # Textual Inversion 阶段步数
MAX_TRAIN_STEPS_TUNING=1000 # LoRA 微调阶段步数
SAVE_STEPS=200              # 每隔多少步保存一次 checkpoint

# 学习率
LR_UNET=1e-4       # UNet 学习率
LR_TEXT=1e-5        # Text Encoder 学习率
LR_TI=5e-4          # Textual Inversion 学习率

# ---------- 开始训练 ----------
echo "============================================"
echo "  LoRA Training with Pivotal Tuning"
echo "============================================"

# 确保 lora_diffusion 包已安装 (提供 lora_pti 命令)
if ! command -v lora_pti &> /dev/null; then
  echo "[INFO] lora_pti 未找到，正在从 ./lora 目录安装 lora_diffusion 包..."
  pip install -e ./lora
fi

echo "Model:      $MODEL_NAME"
echo "Data:       $INSTANCE_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Template:   $USE_TEMPLATE"
echo "LoRA Rank:  $LORA_RANK"
echo "TI Steps:   $MAX_TRAIN_STEPS_TI"
echo "Tune Steps: $MAX_TRAIN_STEPS_TUNING"
echo "============================================"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --scale_lr \
  --learning_rate_unet=$LR_UNET \
  --learning_rate_text=$LR_TEXT \
  --learning_rate_ti=$LR_TI \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --placeholder_tokens="$PLACEHOLDER_TOKENS" \
  --use_template="$USE_TEMPLATE" \
  --save_steps=$SAVE_STEPS \
  --max_train_steps_ti=$MAX_TRAIN_STEPS_TI \
  --max_train_steps_tuning=$MAX_TRAIN_STEPS_TUNING \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001 \
  --continue_inversion \
  --continue_inversion_lr=1e-4 \
  --device="cuda:0" \
  --lora_rank=$LORA_RANK \
  --seed=42

echo ""
echo "训练完成! 模型保存在: $OUTPUT_DIR"
echo "最终 LoRA 权重: $OUTPUT_DIR/final_lora.safetensors"
