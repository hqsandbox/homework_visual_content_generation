"""
可视化脚本: 展示训练结果和对比效果

功能:
1. 将生成的图片拼成网格图 (grid)
2. 对比不同 LoRA scale 下的生成效果
3. 对比训练不同阶段的 checkpoint 效果

用法:
    # 拼接已生成的图片为网格
    python scripts/visualize_results.py grid \
        --image_dir outputs/generated \
        --output outputs/grid.png \
        --cols 4

    # 对比不同 LoRA scale 的效果
    python scripts/visualize_results.py compare_scales \
        --lora_path outputs/lora_output/final_lora.safetensors \
        --prompt "a cat in style of <s1><s2>" \
        --output outputs/scale_comparison.png

    # 对比不同 checkpoint 的效果
    python scripts/visualize_results.py compare_checkpoints \
        --checkpoint_dir outputs/lora_output \
        --prompt "a cat in style of <s1><s2>" \
        --output outputs/checkpoint_comparison.png
"""

import argparse
import glob
import math
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont


def make_grid(images, cols=4, padding=10, bg_color=(255, 255, 255)):
    """将多张图片拼成网格。"""
    if not images:
        print("没有图片可以拼接")
        return None

    rows = math.ceil(len(images) / cols)
    w, h = images[0].size

    grid_w = cols * w + (cols + 1) * padding
    grid_h = rows * h + (rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), bg_color)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = padding + col * (w + padding)
        y = padding + row * (h + padding)
        grid.paste(img.resize((w, h)), (x, y))

    return grid


def add_title(image, title, font_size=24):
    """在图片顶部添加标题。"""
    title_h = font_size + 20
    new_img = Image.new("RGB", (image.width, image.height + title_h), (255, 255, 255))
    new_img.paste(image, (0, title_h))
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((image.width - text_w) // 2, 10), title, fill=(0, 0, 0), font=font)
    return new_img


def cmd_grid(args):
    """将目录中的图片拼成网格。"""
    image_files = sorted(
        glob.glob(os.path.join(args.image_dir, "*.png"))
        + glob.glob(os.path.join(args.image_dir, "*.jpg"))
    )

    if not image_files:
        print(f"在 {args.image_dir} 中未找到图片")
        return

    images = [Image.open(f).convert("RGB") for f in image_files]
    print(f"找到 {len(images)} 张图片，拼接为 {args.cols} 列网格...")

    grid = make_grid(images, cols=args.cols)
    if args.title:
        grid = add_title(grid, args.title)
    grid.save(args.output)
    print(f"网格图已保存: {args.output}")


def cmd_compare_scales(args):
    """对比不同 LoRA scale 下的生成效果。"""
    from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
    from lora_diffusion import patch_pipe, tune_lora_scale

    scales = [0.0, 0.3, 0.5, 0.7, 1.0]
    model_id = args.model_id

    print(f"加载模型: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    patch_pipe(pipe, args.lora_path, patch_text=True, patch_ti=True, patch_unet=True)

    images = []
    for scale in scales:
        print(f"  生成 scale={scale}...")
        tune_lora_scale(pipe.unet, scale)
        tune_lora_scale(pipe.text_encoder, scale)
        torch.manual_seed(args.seed)
        img = pipe(
            args.prompt,
            negative_prompt="blurry, bad quality",
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images[0]
        # 添加 scale 标注
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
        draw.text((10, 10), f"α={scale}", fill=(255, 255, 255), font=font)
        images.append(img)

    grid = make_grid(images, cols=len(scales))
    grid = add_title(grid, f"LoRA Scale Comparison: {args.prompt}")
    grid.save(args.output)
    print(f"对比图已保存: {args.output}")


def cmd_compare_checkpoints(args):
    """对比不同训练阶段 checkpoint 的效果。"""
    from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
    from lora_diffusion import patch_pipe, tune_lora_scale

    # 查找所有 checkpoint
    ckpt_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "step_*.safetensors")))
    final = os.path.join(args.checkpoint_dir, "final_lora.safetensors")
    if os.path.exists(final):
        ckpt_files.append(final)

    if not ckpt_files:
        print(f"在 {args.checkpoint_dir} 中未找到 checkpoint")
        return

    print(f"找到 {len(ckpt_files)} 个 checkpoint")
    model_id = args.model_id

    images = []
    for ckpt_path in ckpt_files:
        ckpt_name = os.path.basename(ckpt_path).replace(".safetensors", "")
        print(f"  生成 {ckpt_name}...")

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(args.device)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        patch_pipe(pipe, ckpt_path, patch_text=True, patch_ti=True, patch_unet=True)
        tune_lora_scale(pipe.unet, 0.8)
        tune_lora_scale(pipe.text_encoder, 0.8)

        torch.manual_seed(args.seed)
        img = pipe(
            args.prompt,
            negative_prompt="blurry, bad quality",
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images[0]

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        draw.text((10, 10), ckpt_name, fill=(255, 255, 255), font=font)
        images.append(img)

        del pipe
        torch.cuda.empty_cache()

    grid = make_grid(images, cols=min(len(images), 5))
    grid = add_title(grid, f"Training Progress: {args.prompt}")
    grid.save(args.output)
    print(f"训练过程对比图已保存: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="可视化工具")
    subparsers = parser.add_subparsers(dest="command")

    # grid 子命令
    p_grid = subparsers.add_parser("grid", help="将图片拼成网格")
    p_grid.add_argument("--image_dir", type=str, required=True)
    p_grid.add_argument("--output", type=str, default="outputs/grid.png")
    p_grid.add_argument("--cols", type=int, default=4)
    p_grid.add_argument("--title", type=str, default=None)

    # compare_scales 子命令
    p_scales = subparsers.add_parser("compare_scales", help="对比不同 LoRA scale")
    p_scales.add_argument("--lora_path", type=str, required=True)
    p_scales.add_argument("--prompt", type=str, required=True)
    p_scales.add_argument("--output", type=str, default="outputs/scale_comparison.png")
    p_scales.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p_scales.add_argument("--seed", type=int, default=42)
    p_scales.add_argument("--device", type=str, default="cuda")

    # compare_checkpoints 子命令
    p_ckpt = subparsers.add_parser("compare_checkpoints", help="对比不同 checkpoint")
    p_ckpt.add_argument("--checkpoint_dir", type=str, required=True)
    p_ckpt.add_argument("--prompt", type=str, required=True)
    p_ckpt.add_argument("--output", type=str, default="outputs/checkpoint_comparison.png")
    p_ckpt.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p_ckpt.add_argument("--seed", type=int, default=42)
    p_ckpt.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.command == "grid":
        cmd_grid(args)
    elif args.command == "compare_scales":
        cmd_compare_scales(args)
    elif args.command == "compare_checkpoints":
        cmd_compare_checkpoints(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
