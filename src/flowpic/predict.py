from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .library import embed_images, embed_pcap, load_model_checkpoint, load_template_library, mean_embedding
from .matching import match_embedding


def load_library_tensors(library_path: str | Path) -> dict[str, torch.Tensor]:
    library = load_template_library(library_path)
    templates = library.get("templates", {})
    return {
        label: torch.tensor(template["embedding"], dtype=torch.float32)
        for label, template in templates.items()
    }


def predict_input(
    *,
    checkpoint_path: str | Path,
    library_path: str | Path,
    image_path: str | Path | None = None,
    pcap_path: str | Path | None = None,
    threshold: float = 0.35,
    metric: str = "cosine",
    window_seconds: float = 5.0,
    bins_time: int = 224,
    bins_size: int = 224,
    time_feature: str = "arrival_time",
    max_packet_size: int = 1600,
) -> dict[str, object]:
    if bool(image_path) == bool(pcap_path):
        raise ValueError("Provide exactly one of image_path or pcap_path.")

    model, device = load_model_checkpoint(checkpoint_path)
    if image_path:
        embedding = mean_embedding(embed_images(model, [image_path], device))
        source = str(image_path)
    else:
        embedding = mean_embedding(
            embed_pcap(
                model,
                pcap_path,
                device=device,
                window_seconds=window_seconds,
                bins_time=bins_time,
                bins_size=bins_size,
                time_feature=time_feature,
                max_packet_size=max_packet_size,
            )
        )
        source = str(pcap_path)

    result = match_embedding(embedding, load_library_tensors(library_path), threshold=threshold, metric=metric)
    return {
        "input": source,
        "label": result.label,
        "distance": result.distance,
        "is_unknown": result.is_unknown,
        "metric": metric,
        "threshold": threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict application label from an unknown FlowPic or PCAP.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint.")
    parser.add_argument("--library", required=True, help="Path to templates/library.json.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", help="Path to a FlowPic image.")
    source_group.add_argument("--pcap", help="Path to a PCAP file.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Unknown decision threshold.")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine", help="Distance metric.")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--bins-time", type=int, default=224)
    parser.add_argument("--bins-size", type=int, default=224)
    parser.add_argument("--time-feature", choices=["arrival_time", "inter_arrival"], default="arrival_time")
    parser.add_argument("--max-packet-size", type=int, default=1600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_input(
        checkpoint_path=args.checkpoint,
        library_path=args.library,
        image_path=args.image,
        pcap_path=args.pcap,
        threshold=args.threshold,
        metric=args.metric,
        window_seconds=args.window_seconds,
        bins_time=args.bins_time,
        bins_size=args.bins_size,
        time_feature=args.time_feature,
        max_packet_size=args.max_packet_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
