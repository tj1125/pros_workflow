import argparse
from pathlib import Path

import torch
from PIL import Image


DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "checkpoints" / "siglip2-base-patch16-224"
DEFAULT_LABELS = ["wine bottle", "doll", "apple"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal SigLIP2 image-text matching demo.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model id.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Local directory containing the downloaded SigLIP2 model.",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Candidate label. Repeat this flag to add multiple labels.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu or cuda:0.",
    )
    return parser.parse_args()


def load_siglip2(model_ref: str, cache_dir: Path, device: torch.device):
    import transformers
    from packaging import version

    min_version = version.parse("4.57.0")
    current_version = version.parse(transformers.__version__)
    if current_version < min_version:
        raise RuntimeError(
            f"SigLIP2 requires transformers>={min_version}, but found {transformers.__version__}. "
            "Upgrade transformers before running this demo."
        )

    from transformers import AutoModel, AutoProcessor

    source = str(cache_dir.expanduser().resolve()) if cache_dir.exists() else model_ref
    processor = AutoProcessor.from_pretrained(source)
    model = AutoModel.from_pretrained(source).to(device).eval()
    return processor, model, source


def main() -> None:
    args = parse_args()
    labels = args.labels or DEFAULT_LABELS
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    device = torch.device(args.device)
    processor, model, source = load_siglip2(args.model_id, args.cache_dir, device)

    image = Image.open(image_path).convert("RGB")
    texts = [f"This is a photo of {label}." for label in labels]
    images = [image] * len(texts)

    inputs = processor(
        text=texts,
        images=images,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = torch.sigmoid(logits)

    ranked = sorted(zip(labels, probs.tolist()), key=lambda item: item[1], reverse=True)

    print(f"device: {device}")
    print(f"source: {source}")
    print(f"image: {image_path}")
    print("scores:")
    for label, score in ranked:
        print(f"  {label}: {score:.4f}")


if __name__ == "__main__":
    main()
