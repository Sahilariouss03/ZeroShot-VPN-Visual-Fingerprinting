from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .preprocess import FlowWindow, iter_packet_records, window_packet_records

TIME_FEATURES = {"arrival_time", "inter_arrival"}
CONTRAST_MODES = {"linear", "sqrt", "log"}


@dataclass(frozen=True)
class FlowPicManifestRow:
    label: str
    source_pcap: str
    window_index: int
    window_start: float
    window_end: float
    packet_count: int
    image_path: str


def _extract_time_values(flow_window: FlowWindow, time_feature: str) -> np.ndarray:
    if time_feature == "arrival_time":
        if len(flow_window.packets) == 1:
            return np.array([0.0], dtype=np.float32)

        duration = max(flow_window.packets[-1].relative_time - flow_window.packets[0].relative_time, 1e-9)
        return np.array(
            [(packet.relative_time - flow_window.packets[0].relative_time) / duration for packet in flow_window.packets],
            dtype=np.float32,
        )

    if time_feature == "inter_arrival":
        iats = np.array([packet.inter_arrival_time for packet in flow_window.packets], dtype=np.float32)
        max_iat = max(float(iats.max(initial=0.0)), 1e-9)
        return np.clip(iats / max_iat, 0.0, 1.0)

    raise ValueError(f"Unsupported time feature: {time_feature}")


def build_flowpic_histogram(
    flow_window: FlowWindow,
    bins_time: int = 224,
    bins_size: int = 224,
    time_feature: str = "arrival_time",
    max_packet_size: int = 1600,
    contrast: str = "log",
    blur: int = 0,
    contrast_percentile: float = 0.995,
) -> np.ndarray:
    if bins_time <= 0 or bins_size <= 0:
        raise ValueError("bins_time and bins_size must be positive.")
    if time_feature not in TIME_FEATURES:
        raise ValueError(f"time_feature must be one of {sorted(TIME_FEATURES)}")
    if contrast not in CONTRAST_MODES:
        raise ValueError(f"contrast must be one of {sorted(CONTRAST_MODES)}")
    if blur < 0:
        raise ValueError("blur must be >= 0")
    if not (0.5 <= contrast_percentile <= 1.0):
        raise ValueError("contrast_percentile must be between 0.5 and 1.0")
    if not flow_window.packets:
        raise ValueError("Cannot build a FlowPic histogram from an empty window.")

    time_values = _extract_time_values(flow_window, time_feature=time_feature)
    packet_sizes = np.array(
        [min(max(packet.packet_size, 0), max_packet_size) for packet in flow_window.packets],
        dtype=np.float32,
    )

    histogram, _, _ = np.histogram2d(
        x=time_values,
        y=packet_sizes,
        bins=[bins_time, bins_size],
        range=[[0.0, 1.0], [0.0, max_packet_size]],
    )
    histogram = histogram.astype(np.float32)
    if contrast == "sqrt":
        histogram = np.sqrt(histogram)
    elif contrast == "log":
        histogram = np.log1p(histogram)

    if blur > 0:
        # Lightweight separable box blur to make point-like histograms more image-like without extra deps.
        kernel = np.ones((2 * blur + 1,), dtype=np.float32)
        kernel /= float(kernel.sum())
        histogram = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, histogram)
        histogram = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 1, histogram)

    # Auto-contrast based on a high percentile of non-zero bins. This makes sparse-but-real structure visible.
    nonzero = histogram[histogram > 0]
    if nonzero.size > 0:
        scale = float(np.quantile(nonzero, contrast_percentile))
        if scale <= 0:
            scale = float(nonzero.max(initial=0.0))
        if scale > 0:
            histogram = np.clip(histogram / scale, 0.0, 1.0)

    return histogram.T


def load_flow_windows_from_pcap(
    pcap_path: str | Path,
    *,
    window_seconds: float = 5.0,
) -> list[FlowWindow]:
    packets = list(iter_packet_records(pcap_path))
    return window_packet_records(packets, window_seconds=window_seconds, allow_single_short_window=True)


def build_histograms_from_pcap(
    pcap_path: str | Path,
    *,
    window_seconds: float = 5.0,
    bins_time: int = 224,
    bins_size: int = 224,
    time_feature: str = "arrival_time",
    max_packet_size: int = 1600,
    contrast: str = "log",
    blur: int = 0,
    contrast_percentile: float = 0.995,
) -> list[tuple[FlowWindow, np.ndarray]]:
    windows = load_flow_windows_from_pcap(pcap_path, window_seconds=window_seconds)
    return [
        (
            window,
            build_flowpic_histogram(
                window,
                bins_time=bins_time,
                bins_size=bins_size,
                time_feature=time_feature,
                max_packet_size=max_packet_size,
                contrast=contrast,
                blur=blur,
                contrast_percentile=contrast_percentile,
            ),
        )
        for window in windows
    ]


def save_flowpic_image(histogram: np.ndarray, output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.clip(histogram * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    image.save(output_file)


def show_flowpic_preview(histogram: np.ndarray, *, title: str | None = None) -> None:
    figure, axis = plt.subplots(figsize=(6, 6), dpi=160)
    axis.imshow(histogram, cmap="gray", origin="lower", aspect="auto")
    axis.set_xlabel("Normalized Time")
    axis.set_ylabel("Packet Size Bin")
    if title:
        axis.set_title(title)
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def write_manifest(rows: list[FlowPicManifestRow | Mapping[str, object]], manifest_path: str | Path) -> None:
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    with manifest_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "source_pcap",
                "window_index",
                "window_start",
                "window_end",
                "packet_count",
                "image_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row.__dict__) if isinstance(row, FlowPicManifestRow) else dict(row))


def process_pcap_to_flowpics(
    pcap_path: str | Path,
    output_dir: str | Path,
    *,
    label: str,
    window_seconds: float = 5.0,
    bins_time: int = 224,
    bins_size: int = 224,
    time_feature: str = "arrival_time",
    max_packet_size: int = 1600,
    contrast: str = "log",
    blur: int = 0,
    contrast_percentile: float = 0.995,
    prefix: str | None = None,
) -> list[FlowPicManifestRow]:
    pcap_file = Path(pcap_path)
    processed_dir = Path(output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    base_name = prefix or pcap_file.stem

    rows: list[FlowPicManifestRow] = []
    for window, histogram in build_histograms_from_pcap(
        pcap_file,
        window_seconds=window_seconds,
        bins_time=bins_time,
        bins_size=bins_size,
        time_feature=time_feature,
        max_packet_size=max_packet_size,
        contrast=contrast,
        blur=blur,
        contrast_percentile=contrast_percentile,
    ):
        image_path = processed_dir / f"{base_name}_w{window.index:04d}.png"
        save_flowpic_image(histogram, image_path)
        rows.append(
            FlowPicManifestRow(
                label=label,
                source_pcap=str(pcap_file),
                window_index=window.index,
                window_start=window.start_time,
                window_end=window.end_time,
                packet_count=len(window.packets),
                image_path=str(image_path),
            )
        )
    return rows


def generate_flowpic_preview(
    pcap_path: str | Path,
    output_path: str | Path,
    *,
    label: str = "unknown",
    window_seconds: float = 5.0,
    bins_time: int = 96,
    bins_size: int = 96,
    time_feature: str = "arrival_time",
    max_packet_size: int = 1600,
    contrast: str = "log",
    blur: int = 2,
    contrast_percentile: float = 0.995,
    show: bool = False,
    manifest_path: str | Path | None = None,
    window_index: int = -1,
) -> FlowPicManifestRow:
    histograms = build_histograms_from_pcap(
        pcap_path,
        window_seconds=window_seconds,
        bins_time=bins_time,
        bins_size=bins_size,
        time_feature=time_feature,
        max_packet_size=max_packet_size,
        contrast=contrast,
        blur=blur,
        contrast_percentile=contrast_percentile,
    )
    if not histograms:
        raise ValueError("No IP packets were found in the provided PCAP.")
    if window_index == -1:
        window_index = int(np.argmax([len(window.packets) for window, _ in histograms]))
    if window_index < 0 or window_index >= len(histograms):
        raise IndexError(f"window_index {window_index} is out of range for {len(histograms)} windows.")

    target_window, histogram = histograms[window_index]
    save_flowpic_image(histogram, output_path)
    if show:
        show_flowpic_preview(histogram, title=f"FlowPic Window {target_window.index} ({time_feature})")

    row = FlowPicManifestRow(
        label=label,
        source_pcap=str(Path(pcap_path)),
        window_index=target_window.index,
        window_start=target_window.start_time,
        window_end=target_window.end_time,
        packet_count=len(target_window.packets),
        image_path=str(Path(output_path)),
    )

    final_manifest = Path(manifest_path) if manifest_path else Path(output_path).with_suffix(".csv")
    write_manifest([row], final_manifest)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a FlowPic image from a PCAP.")
    parser.add_argument("--pcap", required=True, help="Path to the source PCAP file.")
    parser.add_argument("--output", required=True, help="Path to the output PNG file.")
    parser.add_argument("--label", default="unknown", help="Label to store in the manifest row.")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Window size in seconds.")
    parser.add_argument(
        "--window-index",
        type=int,
        default=-1,
        help="Which non-empty window to render. Use -1 to automatically pick the window with most packets.",
    )
    parser.add_argument("--bins-time", type=int, default=96, help="Number of histogram bins on the x-axis.")
    parser.add_argument("--bins-size", type=int, default=96, help="Number of histogram bins on the y-axis.")
    parser.add_argument(
        "--time-feature",
        choices=sorted(TIME_FEATURES),
        default="arrival_time",
        help="Time feature to map on the x-axis.",
    )
    parser.add_argument("--max-packet-size", type=int, default=1600, help="Maximum packet size used for clipping.")
    parser.add_argument(
        "--contrast",
        choices=sorted(CONTRAST_MODES),
        default="log",
        help="Histogram contrast transform to make sparse windows visible.",
    )
    parser.add_argument("--blur", type=int, default=1, help="Box-blur radius applied after normalization (0 disables).")
    parser.add_argument(
        "--contrast-percentile",
        type=float,
        default=0.995,
        help="Percentile used for auto-contrast scaling (higher emphasizes sparse structure).",
    )
    parser.add_argument("--manifest", help="Optional manifest CSV output path.")
    parser.add_argument("--show", action="store_true", help="Display the FlowPic interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_flowpic_preview(
        pcap_path=args.pcap,
        output_path=args.output,
        label=args.label,
        window_seconds=args.window_seconds,
        bins_time=args.bins_time,
        bins_size=args.bins_size,
        time_feature=args.time_feature,
        max_packet_size=args.max_packet_size,
        contrast=args.contrast,
        blur=args.blur,
        contrast_percentile=args.contrast_percentile,
        show=args.show,
        manifest_path=args.manifest,
        window_index=args.window_index,
    )


if __name__ == "__main__":
    main()
