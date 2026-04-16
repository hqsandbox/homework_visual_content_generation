"""
数据预处理脚本
将原始图片统一处理为 512x512 的 RGB 图片，用于 LoRA 训练。

用法:
    python scripts/prepare_data.py --input_dir data/raw --output_dir data/processed --size 512
"""

import argparse
import os
from pathlib import Path

from PIL import Image


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def resize_and_crop(image: Image.Image, size: int) -> Image.Image:
    """将图片等比例缩放后中心裁剪为 size x size。"""
    w, h = image.size
    # 等比缩放，使短边等于 size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    # 中心裁剪
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    image = image.crop((left, top, left + size, top + size))
    return image


def process_images(input_dir: str, output_dir: str, size: int = 512):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        return

    image_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in SUPPORTED_FORMATS
    )

    if len(image_files) == 0:
        print(f"[ERROR] 在 {input_dir} 中未找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    success_count = 0
    for i, img_file in enumerate(image_files):
        try:
            img = Image.open(img_file).convert("RGB")
            img = resize_and_crop(img, size)
            output_file = output_path / f"{i:04d}.jpg"
            img.save(output_file, "JPEG", quality=95)
            success_count += 1
        except Exception as e:
            print(f"  [WARN] 跳过 {img_file.name}: {e}")

    print(f"\n处理完成! 成功: {success_count}/{len(image_files)}")
    print(f"输出目录: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理训练图片")
    parser.add_argument("--input_dir", type=str, default="data/raw",
                        help="原始图片目录")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="处理后图片输出目录")
    parser.add_argument("--size", type=int, default=512,
                        help="输出图片尺寸 (默认 512)")
    args = parser.parse_args()
    process_images(args.input_dir, args.output_dir, args.size)
