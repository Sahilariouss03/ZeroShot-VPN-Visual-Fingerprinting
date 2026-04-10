from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .generate import FlowPicManifestRow, save_flowpic_image, write_manifest
from .library import register_application
from .predict import predict_input
from .train_pipeline import train_model

IMAGE_SIZE = 96
KNOWN_APPS = ("YouTube", "Zoom")
ZERO_DAY_APP = "Signal"


@dataclass(frozen=True)
class DemoPaths:
    root: Path
    images_root: Path
    manifests_root: Path
    checkpoints_root: Path
    templates_root: Path
    train_manifest: Path
    known_manifest: Path
    zero_day_manifest: Path
    checkpoint: Path
    library: Path
    report: Path


def _paths(root: str | Path) -> DemoPaths:
    base = Path(root)
    images_root = base / "images"
    manifests_root = base / "manifests"
    checkpoints_root = base / "checkpoints"
    templates_root = base / "templates"
    return DemoPaths(
        root=base,
        images_root=images_root,
        manifests_root=manifests_root,
        checkpoints_root=checkpoints_root,
        templates_root=templates_root,
        train_manifest=manifests_root / "train_manifest.csv",
        known_manifest=manifests_root / "known_manifest.csv",
        zero_day_manifest=manifests_root / "zero_day_manifest.csv",
        checkpoint=checkpoints_root / "demo_backbone.pth",
        library=templates_root / "library.json",
        report=base / "demo_report.json",
    )


def _make_pattern(label: str, sample_index: int, size: int = IMAGE_SIZE) -> np.ndarray:
    seed = int(hashlib.sha256(f"{label}:{sample_index}".encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed=seed)
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    noise = rng.normal(0.0, 0.015, size=(size, size)).astype(np.float32)

    if label == "YouTube":
        base = np.zeros((size, size), dtype=np.float32)
        base[int(size * 0.55) : int(size * 0.88), int(size * 0.2) : int(size * 0.9)] = 0.8
        base[int(size * 0.25) : int(size * 0.35), int(size * 0.55) : int(size * 0.85)] = 0.5
        accents = 0.08 * ((grid_y % 10) < 2).astype(np.float32)
    elif label == "Zoom":
        base = np.zeros((size, size), dtype=np.float32)
        base[int(size * 0.12) : int(size * 0.92), int(size * 0.12) : int(size * 0.34)] = 0.78
        diagonal = np.abs(grid_x - (size - grid_y)) < 5
        base[diagonal] = 0.62
        accents = 0.07 * ((grid_x % 12) < 3).astype(np.float32)
    elif label == ZERO_DAY_APP:
        base = np.zeros((size, size), dtype=np.float32)
        ring = np.sqrt((grid_x - size * 0.5) ** 2 + (grid_y - size * 0.5) ** 2)
        base[np.logical_and(ring > size * 0.18, ring < size * 0.28)] = 0.84
        base[int(size * 0.08) : int(size * 0.18), int(size * 0.4) : int(size * 0.6)] = 0.7
        base[int(size * 0.82) : int(size * 0.92), int(size * 0.38) : int(size * 0.62)] = 0.7
        accents = 0.06 * (((grid_x + grid_y) % 18) < 3).astype(np.float32)
    else:
        base = rng.random((size, size), dtype=np.float32)
        accents = np.zeros((size, size), dtype=np.float32)

    histogram = np.clip(base + accents + noise, 0.0, 1.0).astype(np.float32)
    histogram[:6, :] *= 0.35
    histogram[:, :4] *= 0.6
    return histogram


def _build_rows_for_label(label: str, output_dir: Path, count: int, source_tag: str) -> list[FlowPicManifestRow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[FlowPicManifestRow] = []
    for index in range(count):
        histogram = _make_pattern(label, index)
        image_path = output_dir / f"{label.lower()}_{index:02d}.png"
        save_flowpic_image(histogram, image_path)
        rows.append(
            FlowPicManifestRow(
                label=label,
                source_pcap=f"synthetic://{source_tag}/{label.lower()}_{index:02d}",
                window_index=index,
                window_start=float(index * 5),
                window_end=float((index + 1) * 5),
                packet_count=256,
                image_path=str(image_path),
            )
        )
    return rows


def create_demo_assets(root: str | Path = "demo_assets") -> dict[str, object]:
    paths = _paths(root)
    for directory in [paths.images_root, paths.manifests_root, paths.checkpoints_root, paths.templates_root]:
        directory.mkdir(parents=True, exist_ok=True)

    known_rows: list[FlowPicManifestRow] = []
    train_rows: list[FlowPicManifestRow] = []
    zero_day_rows: list[FlowPicManifestRow] = []

    for label in KNOWN_APPS:
        rows = _build_rows_for_label(label, paths.images_root / label, count=6, source_tag="known")
        known_rows.extend(rows)
        train_rows.extend(rows[:4])

    zero_day_rows.extend(_build_rows_for_label(ZERO_DAY_APP, paths.images_root / ZERO_DAY_APP, count=4, source_tag="zero-day"))

    write_manifest(train_rows, paths.train_manifest)
    write_manifest(known_rows, paths.known_manifest)
    write_manifest(zero_day_rows, paths.zero_day_manifest)

    return {
        "root": str(paths.root),
        "train_manifest": str(paths.train_manifest),
        "known_manifest": str(paths.known_manifest),
        "zero_day_manifest": str(paths.zero_day_manifest),
        "known_apps": list(KNOWN_APPS),
        "zero_day_app": ZERO_DAY_APP,
    }


def run_demo(root: str | Path = "demo_assets", *, epochs: int = 6, threshold: float = 0.08) -> dict[str, object]:
    paths = _paths(root)
    assets = create_demo_assets(paths.root)

    training_result = train_model(
        paths.train_manifest,
        paths.checkpoint,
        epochs=epochs,
        batch_size=4,
        samples_per_class=2,
    )

    for label in KNOWN_APPS:
        register_application(
            label=label,
            checkpoint_path=paths.checkpoint,
            library_path=paths.library,
            manifest_path=paths.known_manifest,
        )

    known_prediction = predict_input(
        checkpoint_path=paths.checkpoint,
        library_path=paths.library,
        image_path=paths.images_root / KNOWN_APPS[0] / f"{KNOWN_APPS[0].lower()}_04.png",
        threshold=threshold,
    )
    unknown_before = predict_input(
        checkpoint_path=paths.checkpoint,
        library_path=paths.library,
        image_path=paths.images_root / ZERO_DAY_APP / f"{ZERO_DAY_APP.lower()}_00.png",
        threshold=threshold,
    )

    register_application(
        label=ZERO_DAY_APP,
        checkpoint_path=paths.checkpoint,
        library_path=paths.library,
        manifest_path=paths.zero_day_manifest,
    )
    unknown_after = predict_input(
        checkpoint_path=paths.checkpoint,
        library_path=paths.library,
        image_path=paths.images_root / ZERO_DAY_APP / f"{ZERO_DAY_APP.lower()}_00.png",
        threshold=threshold,
    )

    report = {
        "assets": assets,
        "training": training_result,
        "known_prediction": known_prediction,
        "zero_day_before_registration": unknown_before,
        "zero_day_after_registration": unknown_after,
        "demo_claim": {
            "known_match": known_prediction["label"] == KNOWN_APPS[0],
            "zero_day_unknown_before": unknown_before["label"] == "Unknown",
            "zero_day_known_after": unknown_after["label"] == ZERO_DAY_APP,
        },
    }
    paths.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and run a synthetic FlowPic demo pipeline.")
    parser.add_argument("--root", default="demo_assets", help="Output directory for demo assets.")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs for the demo checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.08, help="Prediction threshold for the demo.")
    parser.add_argument("--assets-only", action="store_true", help="Only create demo images/manifests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.assets_only:
        print(json.dumps(create_demo_assets(args.root), indent=2))
        return
    print(json.dumps(run_demo(args.root, epochs=args.epochs, threshold=args.threshold), indent=2))


if __name__ == "__main__":
    main()
