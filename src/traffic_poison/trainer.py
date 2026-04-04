from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .metrics import compute_regression_metrics
from .utils import resolve_device


@dataclass
class TrainHistory:
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float


@dataclass
class TrainArtifacts:
    history: TrainHistory
    device: str


def _build_loss(name: str) -> nn.Module:
    if name.lower() == "mse":
        return nn.MSELoss()
    return nn.L1Loss()


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        if training:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else 0.0


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    training_cfg: dict[str, Any],
) -> TrainArtifacts:
    device = resolve_device(training_cfg.get("device"))
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    loss_fn = _build_loss(str(training_cfg.get("loss", "mae")))
    epochs = int(training_cfg.get("epochs", 20))
    patience = int(training_cfg.get("patience", 5))
    grad_clip_norm = training_cfg.get("grad_clip_norm", None)
    grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_state = None
    best_epoch = 0
    best_val = float("inf")
    patience_left = patience

    for epoch in range(epochs):
        train_loss = _run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss = _run_epoch(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_left = patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = TrainHistory(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val,
    )
    return TrainArtifacts(history=history, device=device)


@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.eval()

    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for inputs, batch_targets in loader:
        inputs = inputs.to(resolved_device)
        batch_predictions = model(inputs).detach().cpu().numpy()
        predictions.append(batch_predictions)
        targets.append(batch_targets.detach().cpu().numpy())

    if not targets:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(targets, axis=0), np.concatenate(predictions, axis=0)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    y_true, y_pred = predict_model(model, loader, device=device)
    metrics = compute_regression_metrics(y_true, y_pred)
    return metrics, y_true, y_pred
