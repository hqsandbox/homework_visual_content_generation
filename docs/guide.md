# LoRA 微调 Stable Diffusion 完整指南

## 项目概述

本项目使用 **LoRA (Low-Rank Adaptation)** + **Pivotal Tuning Inversion** 方法微调 Stable Diffusion v1.5 模型，在自定义数据集（100张图片，from CyberHarem/yae_miko_genshin）上学习特定的画风或角色。

### 项目结构

```
sft-for-diffusion-model/
├── data/
│   ├── raw/                 # 原始图片（你收集的100张图）
│   └── processed/           # 预处理后的图片（512x512）
├── scripts/
│   ├── prepare_data.py      # 数据预处理脚本
│   ├── train.sh             # 训练脚本（推荐，使用 lora_pti）
│   ├── train_v2.sh          # 备选训练脚本（使用 accelerate）
│   ├── inference.py          # 推理生成脚本
│   ├── visualize_results.py  # 可视化脚本
│   └── prompts.txt          # 推理用的 prompt 列表
├── outputs/
│   ├── lora_output/         # 训练输出（LoRA权重、checkpoint）
│   └── generated/           # 生成的图片
├── lora/                    # cloneofsimo/lora 仓库
└── docs/
    ├── requirements.md      # 作业要求
    └── guide.md             # 本指南
```

---

## 环境要求

- **GPU**: NVIDIA GPU，显存 >= 12GB（推荐 RTX 3090/4090 或 A100）
- **Python**: 3.8+
- **CUDA**: 11.7+
- **磁盘空间**: ~10GB（模型权重约 5GB + 数据 + 输出）

---

## Step 1: 环境安装

```bash
# 1. 创建 conda 虚拟环境（推荐）
conda create -n lora python=3.10 -y
conda activate lora

# 2. 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 lora_diffusion 包（从本地 lora 目录安装）
cd lora
pip install -e .
cd ..

# 4. 安装其他依赖
pip install accelerate transformers diffusers safetensors
pip install wandb  # 可选，用于训练日志

# 5. 配置 accelerate（使用 train_v2.sh 时需要）
accelerate config
# 按提示选择：单GPU、fp16 即可
```

### 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from lora_diffusion import patch_pipe; print('lora_diffusion OK')"
python -c "from diffusers import StableDiffusionPipeline; print('diffusers OK')"
```

---

## Step 2: 准备数据集（100张图片）

### 2.1 图片收集建议

**推荐方案: 选择一种画风**

收集同一种画风的 100 张图片效果最好。以下是一些推荐主题：

| 主题 | 说明 | 训练 template |
|------|------|--------------|
| 动漫画风 | 同一画师/系列的动漫插画 | `style` |
| 水彩画 | 水彩风格的画作 | `style` |
| 像素艺术 | 像素风格图片 | `style` |
| 特定角色 | 某个动漫/游戏角色的图片 | `object` |
| 自然风景 | 同一风格的风景照 | `style` |

**收集图片的途径：**
- Pixiv、DeviantArt、ArtStation 等艺术网站
- Danbooru 等标签化图库
- Google 图片搜索
- 直接使用数据集（如 Kaggle 上的动漫数据集）

**注意事项：**
- 图片应该风格一致、质量较高
- 分辨率建议 >= 512x512（脚本会自动缩放裁剪）
- 支持格式：jpg、jpeg、png、bmp、webp
- 避免包含大量文字、水印的图片

### 2.2 放置原始图片

将收集好的 100 张图片全部放入 `data/raw/` 目录：

```bash
# 确认图片数量
ls data/raw/ | wc -l
# 应该输出 100 左右
```

### 2.3 预处理图片

运行预处理脚本，将所有图片统一处理为 512x512 的 RGB 格式：

```bash
python scripts/prepare_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --size 512
```

预处理完成后验证：

```bash
# 确认处理后的图片数量
ls data/processed/ | wc -l

# 查看某张图片信息
python -c "from PIL import Image; img=Image.open('data/processed/0000.jpg'); print(f'Size: {img.size}, Mode: {img.mode}')"
# 应输出: Size: (512, 512), Mode: RGB
```

---

## Step 3: 训练模型

### 3.1 方案一：lora_pti 命令行训练（推荐）

这是效果最好的方式，结合了 Textual Inversion + LoRA 微调。

**配置训练参数：** 根据你的数据集主题，编辑 `scripts/train.sh` 中的关键参数：

```bash
# 如果训练画风，保持默认:
USE_TEMPLATE="style"

# 如果训练角色/物体，改为:
USE_TEMPLATE="object"
```

**开始训练：**

```bash
bash scripts/train.sh
```

训练过程分两个阶段：
1. **Textual Inversion 阶段** (1000 步)：学习 `<s1><s2>` token 的 embedding
2. **LoRA 微调阶段** (1000 步)：微调 UNet 和 Text Encoder 的 LoRA 参数

**预计耗时：** 在 RTX 3090 上约 30-60 分钟。

### 3.2 方案二：accelerate 训练（备选）

如果 `lora_pti` 命令有问题，可以使用备选方案：

```bash
# 先编辑 scripts/train_v2.sh 中的参数
bash scripts/train_v2.sh
```

### 3.3 训练参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `lora_rank` | LoRA 矩阵的秩，越大表达力越强 | 风格: 1-4, 角色: 4-8 |
| `learning_rate_unet` | UNet 学习率 | 1e-4 |
| `learning_rate_text` | Text Encoder 学习率 | 1e-5 |
| `learning_rate_ti` | Textual Inversion 学习率 | 5e-4 |
| `max_train_steps_ti` | TI 阶段训练步数 | 500-1500 |
| `max_train_steps_tuning` | LoRA 微调步数 | 500-2000 |
| `gradient_accumulation_steps` | 梯度累积步数 | 4 (显存不够可以增大) |
| `use_template` | 训练模板类型 | "style" 或 "object" |
| `resolution` | 训练分辨率 | 512 |

### 3.4 训练输出

训练完成后，`outputs/lora_output/` 目录下会包含：

```
outputs/lora_output/
├── step_inv_200.safetensors     # TI 阶段 checkpoint
├── step_inv_400.safetensors
├── ...
├── step_200.safetensors         # LoRA 阶段 checkpoint
├── step_400.safetensors
├── ...
└── final_lora.safetensors       # 最终模型 (这是你要用的)
```

---

## Step 4: 推理生成图片

### 4.1 基本用法

```bash
# 使用单个 prompt 生成
python scripts/inference.py \
    --lora_path outputs/lora_output/final_lora.safetensors \
    --prompt "a cat in style of <s1><s2>" \
    --num_images 8 \
    --output_dir outputs/generated
```

### 4.2 批量生成

编辑 `scripts/prompts.txt`，每行写一个 prompt，然后：

```bash
python scripts/inference.py \
    --lora_path outputs/lora_output/final_lora.safetensors \
    --prompt_file scripts/prompts.txt \
    --num_images 4 \
    --output_dir outputs/generated
```

### 4.3 调节生成效果

LoRA scale (alpha) 控制风格强度：
- `alpha = 0.0`：完全不使用 LoRA（原始模型）
- `alpha = 0.5`：中等程度的风格融合
- `alpha = 0.8`：推荐值，较强的风格效果
- `alpha = 1.0`：完全应用 LoRA 风格
- `alpha > 1.0`：超过 1.0 可以强化效果，但可能过拟合

```bash
# 调节 LoRA 强度
python scripts/inference.py \
    --lora_path outputs/lora_output/final_lora.safetensors \
    --prompt "a cat in style of <s1><s2>" \
    --lora_scale_unet 0.6 \
    --lora_scale_text 0.9 \
    --num_images 4
```

### 4.4 其他可调参数

```bash
python scripts/inference.py \
    --lora_path outputs/lora_output/final_lora.safetensors \
    --prompt "a cat in style of <s1><s2>" \
    --guidance_scale 7.5 \       # CFG scale: 越高越贴近prompt, 推荐 5-12
    --num_steps 50 \             # 推理步数: 越多质量越好, 推荐 30-75
    --seed 42 \                  # 随机种子: 相同seed产生相同图片
    --negative_prompt "blurry, bad quality, worst quality"
```

---

## Step 5: 可视化结果

### 5.1 将生成的图片拼成网格

```bash
python scripts/visualize_results.py grid \
    --image_dir outputs/generated \
    --output outputs/grid.png \
    --cols 4 \
    --title "LoRA Generated Images"
```

### 5.2 对比不同 LoRA scale 的效果

```bash
python scripts/visualize_results.py compare_scales \
    --lora_path outputs/lora_output/final_lora.safetensors \
    --prompt "a cat in style of <s1><s2>" \
    --output outputs/scale_comparison.png
```

这会生成 alpha=0.0, 0.3, 0.5, 0.7, 1.0 的对比图，非常适合放到报告中展示 LoRA 的效果。

### 5.3 对比不同训练阶段的效果

```bash
python scripts/visualize_results.py compare_checkpoints \
    --checkpoint_dir outputs/lora_output \
    --prompt "a cat in style of <s1><s2>" \
    --output outputs/checkpoint_comparison.png
```

这会加载每个 checkpoint 并生成对比图，展示模型随训练的变化过程。

---

## 完整流程 Quick Start

```bash
# 0. 激活环境
conda activate lora

# 1. 准备数据（将100张图片放入 data/raw/ 后执行）
python scripts/prepare_data.py --input_dir data/raw --output_dir data/processed

# 2. 训练
bash scripts/train.sh

# 3. 生成图片
python scripts/inference.py \
    --lora_path outputs/lora_output/final_lora.safetensors \
    --prompt_file scripts/prompts.txt \
    --num_images 4

# 4. 可视化
python scripts/visualize_results.py grid \
    --image_dir outputs/generated \
    --output outputs/grid.png \
    --cols 4
```

---

## 常见问题 (FAQ)

### Q: 显存不够 (OOM) 怎么办？

1. 减小 `train_batch_size` 到 1（默认已经是 1）
2. 增大 `gradient_accumulation_steps` 到 8 或 16
3. 在 `train.sh` 中添加 `--gradient_checkpointing` 参数
4. 使用 `--enable_xformers_memory_efficient_attention` (需要安装 xformers)
5. 降低分辨率到 256（不推荐，会影响质量）

### Q: 生成的图片质量不好？

1. **过拟合**: 降低训练步数，或降低 LoRA rank
2. **欠拟合**: 增加训练步数，或提高学习率
3. **调节 scale**: 推理时尝试不同的 `lora_scale_unet` 和 `lora_scale_text`
4. **调节 guidance**: 提高 `guidance_scale` (7.5-12.0)
5. **使用负面 prompt**: 添加 "blurry, bad quality, worst quality, low resolution"

### Q: 训练多少步合适？

- 100 张图片，总步数 (TI + Tuning) 建议 1500-3000 步
- 可以通过对比不同 checkpoint 的效果来判断最佳步数
- 参考 README 中的建议：2500 步左右就能得到较好结果

### Q: style 和 object 模式有什么区别？

- **style**: 学习一种画风/艺术风格。Prompt 模板为 "a painting in the style of {}"
- **object**: 学习一个具体物体/角色。Prompt 模板为 "a photo of a {}"
- 选错不影响训练，但会影响生成效果。画风选 style，角色选 object

---

## 报告建议

报告中建议包含以下内容：

1. **数据集介绍**: 说明选择了什么主题的图片，为什么选择这个主题，展示几张代表性样例
2. **LoRA 方法原理**: 简述 LoRA 的核心思想 (W' = W + AB^T, 低秩分解减少参数量)
3. **训练配置**: 列出主要的超参数设置
4. **训练过程**: 展示 loss 曲线，不同 checkpoint 阶段的生成效果对比
5. **结果展示**: 展示最终模型生成的图片网格
6. **效果分析**:
   - 不同 LoRA scale (alpha) 的效果对比
   - 与原始模型 (alpha=0) 的对比
   - 成功和失败案例分析
7. **总结**: 对 LoRA 方法的优缺点做简要分析
