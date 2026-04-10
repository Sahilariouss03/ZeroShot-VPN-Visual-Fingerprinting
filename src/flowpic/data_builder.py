from __future__ import annotations

import argparse
from pathlib import Path

from .generate import FlowPicManifestRow, process_pcap_to_flowpics, write_manifest

PCAP_SUFFIXES = {".pcap", ".pcapng"}


def discover_labeled_pcaps(raw_root: str | Path) -> list[tuple[str, Path]]:
    root = Path(raw_root)
    items: list[tuple[str, Path]] = []
    if not root.exists():
        return items

    for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for pcap_path in sorted(path for path in label_dir.rglob("*") if path.is_file() and path.suffix.lower() in PCAP_SUFFIXES):
            items.append((label_dir.name, pcap_path))
    return items


def build_dataset(
    raw_root: str | Path = "data/raw",
    processed_root: str | Path = "data/processed",
    *,
    manifest_path: str | Path | None = None,
    window_seconds: float = 5.0,
    bins_time: int = 224,
    bins_size: int = 224,
    time_feature: str = "arrival_time",
    max_packet_size: int = 1600,
) -> list[FlowPicManifestRow]:
    all_rows: list[FlowPicManifestRow] = []
    raw_path = Path(raw_root)
    processed_path = Path(processed_root)
    processed_path.mkdir(parents=True, exist_ok=True)

    for label, pcap_path in discover_labeled_pcaps(raw_path):
        app_output_dir = processed_path / label
        rows = process_pcap_to_flowpics(
            pcap_path,
            app_output_dir,
            label=label,
            window_seconds=window_seconds,
            bins_time=bins_time,
            bins_size=bins_size,
            time_feature=time_feature,
            max_packet_size=max_packet_size,
            # Keep training images crisp and consistent; preview can blur, training should not.
            blur=0,
            prefix=pcap_path.stem,
        )
        all_rows.extend(rows)

    final_manifest = Path(manifest_path) if manifest_path else processed_path / "manifest.csv"
    write_manifest(all_rows, final_manifest)
    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FlowPic dataset from labeled raw PCAP directories.")
    parser.add_argument("--raw-root", default="data/raw", help="Root directory containing data/raw/<app_name>/ PCAP files.")
    parser.add_argument("--processed-root", default="data/processed", help="Output directory for generated FlowPics.")
    parser.add_argument("--manifest", help="Optional manifest CSV path.")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Window size in seconds.")
    parser.add_argument("--bins-time", type=int, default=224, help="Number of histogram bins on the x-axis.")
    parser.add_argument("--bins-size", type=int, default=224, help="Number of histogram bins on the y-axis.")
    parser.add_argument(
        "--time-feature",
        choices=["arrival_time", "inter_arrival"],
        default="arrival_time",
        help="Time feature to map on the x-axis.",
    )
    parser.add_argument("--max-packet-size", type=int, default=1600, help="Maximum packet size used for clipping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_dataset(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        manifest_path=args.manifest,
        window_seconds=args.window_seconds,
        bins_time=args.bins_time,
        bins_size=args.bins_size,
        time_feature=args.time_feature,
        max_packet_size=args.max_packet_size,
    )
    print(f"Generated {len(rows)} FlowPics.")


if __name__ == "__main__":
    main()
