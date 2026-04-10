from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class MatchResult:
    label: str
    distance: float
    is_unknown: bool


def build_template_library(embeddings_by_label: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    library: dict[str, torch.Tensor] = {}
    for label, embeddings in embeddings_by_label.items():
        if embeddings.ndim != 2:
            raise ValueError("Each template entry must have shape [num_examples, embedding_dim].")
        centroid = embeddings.mean(dim=0, keepdim=True)
        library[label] = F.normalize(centroid, p=2, dim=1).squeeze(0)
    return library


def _cosine_distance(unknown: torch.Tensor, template: torch.Tensor) -> float:
    return float(1.0 - F.cosine_similarity(unknown.unsqueeze(0), template.unsqueeze(0)).item())


def _euclidean_distance(unknown: torch.Tensor, template: torch.Tensor) -> float:
    return float(torch.norm(unknown - template, p=2).item())


def match_embedding(
    unknown_embedding: torch.Tensor,
    template_library: dict[str, torch.Tensor],
    *,
    threshold: float = 0.35,
    metric: str = "cosine",
) -> MatchResult:
    if not template_library:
        raise ValueError("template_library must not be empty.")

    unknown = F.normalize(unknown_embedding.view(-1), p=2, dim=0)
    best_label = "Unknown"
    best_distance = float("inf")

    for label, template_embedding in template_library.items():
        template = F.normalize(template_embedding.view(-1), p=2, dim=0)
        if metric == "cosine":
            distance = _cosine_distance(unknown, template)
        elif metric == "euclidean":
            distance = _euclidean_distance(unknown, template)
        else:
            raise ValueError("metric must be either 'cosine' or 'euclidean'.")

        if distance < best_distance:
            best_distance = distance
            best_label = label

    is_unknown = best_distance > threshold
    return MatchResult(
        label="Unknown" if is_unknown else best_label,
        distance=best_distance,
        is_unknown=is_unknown,
    )

