"""
推理脚本: 使用训练好的 LoRA 模型生成图片

用法:
    # 使用 lora_pti 训练的模型 (safetensors 格式)
    python scripts/inference.py \
        --lora_path outputs/lora_output/final_lora.safetensors \
        --prompt "a cat in style of <s1><s2>" \
        --output_dir outputs/generated \
        --num_images 8

    # 批量生成多个 prompt
    python scripts/inference.py \
        --lora_path outputs/lora_output/final_lora.safetensors \
        --prompt_file scripts/prompts.txt \
        --output_dir outputs/generated \
        --num_images 4
"""

import argparse
import os

import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

from lora_diffusion import patch_pipe, tune_lora_scale


def load_pipeline(model_id: str, lora_path: str, device: str = "cuda"):
    """加载基础模型并应用 LoRA 权重。"""
    print(f"加载基础模型: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )

    print(f"加载 LoRA 权重: {lora_path}")
    patch_pipe(
        pipe,
        lora_path,
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )

    return pipe


def generate_images(
    pipe,
    prompt: str,
    num_images: int = 4,
    lora_scale_unet: float = 0.8,
    lora_scale_text: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 42,
    negative_prompt: str = "blurry, bad quality, worst quality, low quality",
):
    """使用给定 prompt 生成多张图片。"""
    tune_lora_scale(pipe.unet, lora_scale_unet)
    tune_lora_scale(pipe.text_encoder, lora_scale_text)

    images = []
    for i in range(num_images):
        torch.manual_seed(seed + i)
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        images.append(image)

    return images


def main():
    parser = argparse.ArgumentParser(description="使用 LoRA 模型生成图片")
    parser.add_argument("--model_id", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="基础模型 ID")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="LoRA 权重文件路径 (.safetensors)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="生成图片的 prompt")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="包含多个 prompt 的文本文件 (每行一个)")
    parser.add_argument("--negative_prompt", type=str,
                        default="blurry, bad quality, worst quality, low quality",
                        help="负面 prompt")
    parser.add_argument("--output_dir", type=str, default="outputs/generated",
                        help="输出目录")
    parser.add_argument("--num_images", type=int, default=4,
                        help="每个 prompt 生成的图片数量")
    parser.add_argument("--lora_scale_unet", type=float, default=0.8,
                        help="UNet LoRA 缩放系数 (0.0-1.5)")
    parser.add_argument("--lora_scale_text", type=float, default=0.8,
                        help="Text Encoder LoRA 缩放系数 (0.0-1.5)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="推理步数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 收集所有 prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts.extend(line.strip() for line in f if line.strip())

    if not prompts:
        # 默认 prompt 列表 (适用于风格 LoRA)
        prompts = [
            "a cat in style of <s1><s2>",
            "a beautiful landscape in style of <s1><s2>",
            "a portrait of a woman in style of <s1><s2>",
            "a castle on a mountain in style of <s1><s2>",
            "a city street at night in style of <s1><s2>",
            "a dragon flying over the ocean in style of <s1><s2>",
            "a forest with sunlight in style of <s1><s2>",
            "a robot in a futuristic city in style of <s1><s2>",
        ]
        print("未指定 prompt, 使用默认风格 prompt 列表")

    # 加载模型
    pipe = load_pipeline(args.model_id, args.lora_path, args.device)

    # 生成图片
    total_count = 0
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[{prompt_idx + 1}/{len(prompts)}] Prompt: {prompt}")
        images = generate_images(
            pipe,
            prompt=prompt,
            num_images=args.num_images,
            lora_scale_unet=args.lora_scale_unet,
            lora_scale_text=args.lora_scale_text,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
        )

        for img_idx, img in enumerate(images):
            filename = f"prompt{prompt_idx:02d}_seed{args.seed + img_idx}.png"
            save_path = os.path.join(args.output_dir, filename)
            img.save(save_path)
            total_count += 1
            print(f"  保存: {save_path}")

    print(f"\n生成完成! 共 {total_count} 张图片保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
