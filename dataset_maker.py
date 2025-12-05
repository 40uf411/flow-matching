import random
from pathlib import Path
from PIL import Image
import argparse
import tqdm

def extract_random_crops(
    tiff_dir: Path,
    output_dir: Path,
    crop_size: int = 500,
    crops_per_image: int = 50,
    image_size: int = None,  # optional resize after crop
):
    tiff_dir = Path(tiff_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted([p for p in tiff_dir.iterdir() if p.suffix.lower() == ".tiff"])
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {tiff_dir}")

    idx = 0
    for path in tqdm.tqdm(tiff_files, desc="Processing TIFF files"):
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size

            if w < crop_size or h < crop_size:
                print(f"Skipping {path} (smaller than crop size)")
                continue

            for _ in range(crops_per_image):
                x0 = random.randint(0, w - crop_size)
                y0 = random.randint(0, h - crop_size)
                crop = img.crop((x0, y0, x0 + crop_size, y0 + crop_size))

                if image_size is not None:
                    crop = crop.resize((image_size, image_size), Image.BICUBIC)

                crop.save(output_dir / f"crop_{idx:06d}.png")
                idx += 1

    print(f"Saved {idx} cropped images in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff_dir", type=str, required=True, help="Folder containing .tiff files")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save cropped PNGs")
    parser.add_argument("--crop_size", type=int, default=500)
    parser.add_argument("--crops_per_image", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=None)

    args = parser.parse_args()

    extract_random_crops(
        tiff_dir=args.tiff_dir,
        output_dir=args.output_dir,
        crop_size=args.crop_size,
        crops_per_image=args.crops_per_image,
        image_size=args.image_size,
    )
