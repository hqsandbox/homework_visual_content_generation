# LoRA Fine-Tuning Stable Diffusion

Visual Content Generation course homework. Fine-tunes Stable Diffusion v1.5 with LoRA + Pivotal Tuning Inversion on a custom dataset (100 images) to learn a specific art style or character.

## Project Structure

```
├── data/
│   ├── raw/                 # Original images
│   └── processed/           # Preprocessed 512×512 images
├── scripts/
│   ├── prepare_data.py      # Data preprocessing
│   ├── train.sh             # Training (lora_pti, recommended)
│   ├── train_v2.sh          # Training (accelerate, alternative)
│   ├── inference.py         # Inference / generation
│   ├── visualize_results.py # Visualization & comparison
│   └── prompts.txt          # Prompt list
├── lora/                    # cloneofsimo/lora library
├── outputs/                 # Training weights & generated results
└── docs/                    # Guide & assignment requirements
```

## Requirements

- Python 3.8+, CUDA 11.7+, GPU ≥ 12 GB VRAM
- PyTorch, diffusers, transformers, safetensors, accelerate

## Usage

```bash
# 1. Data preprocessing (place raw images in data/raw/ first)
python scripts/prepare_data.py --input_dir data/raw --output_dir data/processed

# 2. Training
bash scripts/train.sh

# 3. Inference
python scripts/inference.py \
    --lora_path outputs/lora_output/final_lora.safetensors \
    --prompt_file scripts/prompts.txt \
    --num_images 4
```

See [docs/guide.md](docs/guide.md) for detailed parameter descriptions and FAQ.