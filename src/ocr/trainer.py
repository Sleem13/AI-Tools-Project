"""CRNN training loop for license plate OCR."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ocr.data import OCRPlateDataset, build_vocab
from src.ocr.model import CRNNModel


def train_crnn(
    config: dict,
    train_dataset: OCRPlateDataset,
    val_dataset: OCRPlateDataset | None = None,
    output_dir: str | Path = "models/ocr",
) -> dict:
    """Train a CRNN OCR model.

    Args:
        config: OCR configuration dictionary (from ocr.yaml).
        train_dataset: Training dataset of plate crops.
        val_dataset: Optional validation dataset.
        output_dir: Where to save model checkpoints.

    Returns:
        Dictionary of training results.
    """
    raw_device = config.get("training", {}).get("device", "cpu")
    if isinstance(raw_device, int) or (isinstance(raw_device, str) and raw_device.isdigit()):
        device = torch.device(f"cuda:{raw_device}")
    else:
        device = torch.device(raw_device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hp = config.get("hyperparameters", {})
    mc = config.get("model_config", {})

    vocab = build_vocab()
    num_classes = len(vocab)

    model = CRNNModel(
        num_classes=num_classes,
        input_height=config.get("model", {}).get("input_height", 32),
        cnn_channels=mc.get("cnn_channels", [1, 64, 128, 256, 512, 512]),
        bilstm_hidden=mc.get("bilstm_hidden", 256),
        bilstm_layers=mc.get("bilstm_layers", 2),
        dropout=mc.get("dropout", 0.3),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp.get("lr", 0.0003),
        weight_decay=hp.get("weight_decay", 0.0001),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hp.get("scheduler_step", 10),
        gamma=hp.get("scheduler_gamma", 0.5),
    )
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.get("batch", 64),
        shuffle=True,
        num_workers=config.get("training", {}).get("workers", 4),
        collate_fn=_ctc_collate,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=hp.get("batch", 64),
            shuffle=False,
            num_workers=config.get("training", {}).get("workers", 4),
            collate_fn=_ctc_collate,
        )

    epochs = hp.get("epochs", 50)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, targets, target_lengths in loop:
            images = images.to(device)
            log_probs = model.forward_for_ctc(images)  # (T, B, C)

            input_lengths = torch.full(
                (images.size(0),), log_probs.size(0), dtype=torch.long
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets, target_lengths in val_loader:
                    images = images.to(device)
                    log_probs = model.forward_for_ctc(images)
                    input_lengths = torch.full(
                        (images.size(0),), log_probs.size(0), dtype=torch.long
                    )
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
                    val_loss += loss.item()
            print(f"  Val loss: {val_loss / len(val_loader):.4f}")

        if epoch % config.get("training", {}).get("save_period", 5) == 0:
            ckpt_path = output_dir / f"ocr_epoch_{epoch:03d}.pth"
            torch.save(model.state_dict(), ckpt_path)

    best_path = output_dir / "ocr_best.pth"
    torch.save(model.state_dict(), best_path)
    return {"best_path": str(best_path), "epochs": epochs}


def _ctc_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for CTC training.

    Returns:
        (images, targets, target_lengths)
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)

    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets, dim=0)

    return images, targets_concat, target_lengths
