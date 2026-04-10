from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TripletBatch:
    anchors: torch.Tensor
    positives: torch.Tensor
    negatives: torch.Tensor


def sample_triplets(inputs: torch.Tensor, labels: torch.Tensor) -> TripletBatch:
    if inputs.size(0) != labels.size(0):
        raise ValueError("inputs and labels must contain the same number of samples.")

    anchors: list[torch.Tensor] = []
    positives: list[torch.Tensor] = []
    negatives: list[torch.Tensor] = []

    for index, label in enumerate(labels):
        positive_indices = torch.where(labels == label)[0]
        negative_indices = torch.where(labels != label)[0]
        positive_indices = positive_indices[positive_indices != index]

        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue

        anchors.append(inputs[index])
        positives.append(inputs[int(positive_indices[0])])
        negatives.append(inputs[int(negative_indices[0])])

    if not anchors:
        raise ValueError("Unable to sample triplets from the provided batch.")

    return TripletBatch(
        anchors=torch.stack(anchors),
        positives=torch.stack(positives),
        negatives=torch.stack(negatives),
    )


def train_triplet_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.TripletMarginLoss,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad()

    triplets = sample_triplets(inputs, labels)
    anchor_embeddings = model(triplets.anchors)
    positive_embeddings = model(triplets.positives)
    negative_embeddings = model(triplets.negatives)

    loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def build_triplet_loss(margin: float = 0.2) -> nn.TripletMarginLoss:
    return nn.TripletMarginLoss(margin=margin, p=2)

