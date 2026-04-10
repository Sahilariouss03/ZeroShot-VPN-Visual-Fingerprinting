from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .generate import build_histograms_from_pcap
from .model import FlowPicEmbeddingNet


def load_model_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
) -> tuple[FlowPicEmbeddingNet, torch.device]:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    embedding_dim = int(checkpoint.get("embedding_dim", 128)) if isinstance(checkpoint, dict) else 128
    model = FlowPicEmbeddingNet(embedding_dim=embedding_dim)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()
    return model, resolved_device


def flowpic_image_to_tensor(image_path: str | Path) -> torch.Tensor:
    image = Image.open(image_path).convert("L")
    # Resize to strictly match the 224x224 spatial dimensions expected by the CNN.
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def histogram_to_tensor(histogram: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(histogram.astype(np.float32)).unsqueeze(0)


def embed_tensors(model: FlowPicEmbeddingNet, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not tensors:
        raise ValueError("At least one tensor is required for embedding.")
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        embeddings = model(batch)
    return embeddings.cpu()


def embed_images(
    model: FlowPicEmbeddingNet,
    image_paths: list[str | Path],
    device: torch.device,
) -> torch.Tensor:
    return embed_tensors(model, [flowpic_image_to_tensor(path) for path in image_paths], device)


def embed_pcap(
    model: FlowPicEmbeddingNet,
    pcap_path: str | Path,
    *,
    device: torch.device,
    window_seconds: float = 5.0,
    bins_time: int = 224,
    bins_size: int = 224,
    time_feature: str = "arrival_time",
    max_packet_size: int = 1600,
) -> torch.Tensor:
    histograms = [
        histogram_to_tensor(histogram)
        for _, histogram in build_histograms_from_pcap(
            pcap_path,
            window_seconds=window_seconds,
            bins_time=bins_time,
            bins_size=bins_size,
            time_feature=time_feature,
            max_packet_size=max_packet_size,
        )
    ]
    return embed_tensors(model, histograms, device)


def mean_embedding(embeddings: torch.Tensor) -> torch.Tensor:
    centroid = embeddings.mean(dim=0)
    return F.normalize(centroid, p=2, dim=0)


def load_manifest_image_paths(manifest_path: str | Path, *, label: str | None = None) -> list[str]:
    paths: list[str] = []
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if label is None or row.get("label") == label:
                image_path = row.get("image_path")
                if image_path:
                    paths.append(image_path)
    return paths


def load_template_library(library_path: str | Path) -> dict:
    file_path = Path(library_path)
    if not file_path.exists():
        return {"embedding_dim": 128, "metric": "cosine", "templates": {}}
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_template_library(library_path: str | Path, library: dict) -> None:
    file_path = Path(library_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(library, handle, indent=2)


def register_application(
    *,
    label: str,
    checkpoint_path: str | Path,
    library_path: str | Path,
    image_paths: list[str | Path] | None = None,
    manifest_path: str | Path | None = None,
    device: str | None = None,
) -> dict:
    if not image_paths and not manifest_path:
        raise ValueError("Provide either image_paths or manifest_path when registering an application.")

    selected_images = [str(path) for path in image_paths] if image_paths else load_manifest_image_paths(manifest_path, label=label)
    if not selected_images:
        raise ValueError(f"No FlowPic images found for label '{label}'.")

    model, resolved_device = load_model_checkpoint(checkpoint_path, device=device)
    centroid = mean_embedding(embed_images(model, selected_images, resolved_device))

    library = load_template_library(library_path)
    library.setdefault("templates", {})
    library["embedding_dim"] = int(centroid.numel())
    library["templates"][label] = {
        "embedding": centroid.tolist(),
        "num_samples": len(selected_images),
        "source_images": selected_images,
    }
    save_template_library(library_path, library)
    return library["templates"][label]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage FlowPic embeddings and template registration.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser("embed", help="Generate an embedding from an image or a PCAP.")
    embed_parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint.")
    embed_group = embed_parser.add_mutually_exclusive_group(required=True)
    embed_group.add_argument("--image", help="Path to a FlowPic image.")
    embed_group.add_argument("--pcap", help="Path to a PCAP file.")
    embed_parser.add_argument("--window-seconds", type=float, default=5.0)
    embed_parser.add_argument("--bins-time", type=int, default=224)
    embed_parser.add_argument("--bins-size", type=int, default=224)
    embed_parser.add_argument("--time-feature", choices=["arrival_time", "inter_arrival"], default="arrival_time")
    embed_parser.add_argument("--max-packet-size", type=int, default=1600)

    register_parser = subparsers.add_parser("register", help="Register an application template.")
    register_parser.add_argument("--label", required=True, help="Application label to register.")
    register_parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint.")
    register_parser.add_argument("--library", required=True, help="Path to templates/library.json.")
    register_group = register_parser.add_mutually_exclusive_group(required=True)
    register_group.add_argument("--manifest", help="Manifest CSV containing image paths for the label.")
    register_group.add_argument("--images", nargs="+", help="Explicit list of FlowPic images.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "embed":
        model, device = load_model_checkpoint(args.checkpoint)
        if args.image:
            embedding = mean_embedding(embed_images(model, [args.image], device))
        else:
            embedding = mean_embedding(
                embed_pcap(
                    model,
                    args.pcap,
                    device=device,
                    window_seconds=args.window_seconds,
                    bins_time=args.bins_time,
                    bins_size=args.bins_size,
                    time_feature=args.time_feature,
                    max_packet_size=args.max_packet_size,
                )
            )
        print(json.dumps({"embedding": embedding.tolist()}, indent=2))
        return

    if args.command == "register":
        template = register_application(
            label=args.label,
            checkpoint_path=args.checkpoint,
            library_path=args.library,
            image_paths=args.images,
            manifest_path=args.manifest,
        )
        print(json.dumps(template, indent=2))


if __name__ == "__main__":
    main()
