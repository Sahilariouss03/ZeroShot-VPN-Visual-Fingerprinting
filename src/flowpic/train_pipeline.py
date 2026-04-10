from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import random

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .model import FlowPicEmbeddingNet
from .train import build_triplet_loss, train_triplet_step


class FlowPicDataset(Dataset):
    def __init__(self, manifest_path: str | Path) -> None:
        self.samples: list[tuple[str, str]] = []
        with Path(manifest_path).open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = row.get("label")
                image_path = row.get("image_path")
                if label and image_path:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError("Manifest did not contain any FlowPic samples.")

        labels = sorted({label for _, label in self.samples})
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.class_indices: dict[int, list[int]] = defaultdict(list)
        for sample_index, (_, label) in enumerate(self.samples):
            self.class_indices[self.label_to_index[label]].append(sample_index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0), torch.tensor(self.label_to_index[label], dtype=torch.long)


class LabelBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: FlowPicDataset,
        *,
        batch_size: int = 8,
        samples_per_class: int = 2,
        steps_per_epoch: int | None = None,
    ) -> None:
        if samples_per_class < 2:
            raise ValueError("samples_per_class must be at least 2 for triplet learning.")

        self.dataset = dataset
        self.samples_per_class = samples_per_class
        self.classes_per_batch = batch_size // samples_per_class
        if self.classes_per_batch < 2:
            raise ValueError("batch_size must include at least two classes per batch.")

        self.eligible_classes = [
            class_index
            for class_index, indices in dataset.class_indices.items()
            if len(indices) >= samples_per_class
        ]
        if len(self.eligible_classes) < 2:
            raise ValueError("Need at least two labels with multiple samples to train with triplet loss.")

        self.steps_per_epoch = steps_per_epoch or max(1, len(dataset) // batch_size)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random()
        for _ in range(self.steps_per_epoch):
            chosen_classes = rng.sample(self.eligible_classes, k=min(self.classes_per_batch, len(self.eligible_classes)))
            batch_indices: list[int] = []
            for class_index in chosen_classes:
                batch_indices.extend(rng.sample(self.dataset.class_indices[class_index], k=self.samples_per_class))
            yield batch_indices


def train_model(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 8,
    samples_per_class: int = 2,
    learning_rate: float = 1e-3,
    margin: float = 0.2,
    device: str | None = None,
) -> dict[str, object]:
    dataset = FlowPicDataset(manifest_path)
    dataloader = DataLoader(
        dataset,
        batch_sampler=LabelBalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            samples_per_class=samples_per_class,
        ),
    )

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FlowPicEmbeddingNet()
    model.to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = build_triplet_loss(margin=margin)

    best_loss = float("inf")
    history: list[float] = []
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        epoch_losses: list[float] = []
        for images, labels in dataloader:
            epoch_loss = train_triplet_step(
                model,
                optimizer,
                loss_fn,
                images.to(resolved_device),
                labels.to(resolved_device),
            )
            epoch_losses.append(epoch_loss)

        average_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        history.append(average_loss)
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": 128,
                    "best_loss": best_loss,
                    "epochs": epoch + 1,
                    "manifest_path": str(manifest_path),
                    "labels": dataset.label_to_index,
                },
                checkpoint_file,
            )

    return {
        "checkpoint_path": str(checkpoint_file),
        "best_loss": best_loss,
        "epochs": epochs,
        "num_samples": len(dataset),
        "num_classes": len(dataset.label_to_index),
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FlowPic CNN with triplet margin loss.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv", help="Path to dataset manifest.")
    parser.add_argument("--checkpoint", default="checkpoints/backbone_v1.pth", help="Path to save best checkpoint.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--samples-per-class", type=int, default=2, help="Samples per class in each batch.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet margin.")
    parser.add_argument("--device", help="Optional device override, for example cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_model(
        args.manifest,
        args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_class=args.samples_per_class,
        learning_rate=args.learning_rate,
        margin=args.margin,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
