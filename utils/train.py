from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import random
import matplotlib.pyplot as plt
import torch

from utils.losses import BCEDiceLoss
from utils.metrics import compute_metrics


@dataclass
class PolypTrainConfig:
    model_name: str = "Model"
    save_path: Path = Path("checkpoints/model.pth")
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"          # "cosine" | "plateau" | "onecycle"
    warmup_epochs: int = 0
    cosine_min_lr: float = 1e-6
    checkpoint_policy: str = "last"       # "last" | "best"
    early_stopping_patience: Optional[int] = None
    eval_each_epoch: bool = True
    max_grad_norm: Optional[float] = None
    grad_accum_steps: int = 1
    use_amp: bool = False


def load_checkpoint(model, path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


def evaluate_model(model, loader, criterion, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    total_loss = 0
    acc_metrics = {k: 0.0 for k in ["Accuracy", "Dice", "IoU", "Precision", "Recall", "F1"]}
    n = 0

    with torch.no_grad():
        for img, mask in loader:
            img  = img.to(device)
            mask = mask.unsqueeze(1).to(device)
            pred = model(img)

            total_loss += criterion(pred, mask).item()
            m = compute_metrics(pred, mask)
            for k in acc_metrics:
                acc_metrics[k] += m[k]
            n += 1

    result = {"loss": total_loss / n}
    result.update({k: v / n for k, v in acc_metrics.items()})
    result["iou"] = result["IoU"]   # alias for backward compat
    return result


def train_polyp_segmentation(model, loaders, data_cfg, train_cfg, split_summary, data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    criterion = BCEDiceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # Build scheduler
    if train_cfg.lr_scheduler == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, train_cfg.epochs - train_cfg.warmup_epochs),
            eta_min=train_cfg.cosine_min_lr,
        )
    elif train_cfg.lr_scheduler == "plateau":
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    else:  # onecycle
        main_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_cfg.learning_rate,
            steps_per_epoch=len(loaders["train"]),
            epochs=train_cfg.epochs,
        )

    scaler = torch.cuda.amp.GradScaler() if train_cfg.use_amp else None

    history = {"train_loss": [], "test_loss": [], "train_iou": [], "test_iou": []}
    best_iou = 0.0
    patience_counter = 0
    save_path = Path(train_cfg.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"model={train_cfg.model_name} | device={device} | data_root={data_root} | "
        f"image_size={data_cfg.image_size} | batch_size={data_cfg.batch_size} | "
        f"epochs={train_cfg.epochs} | lr={train_cfg.learning_rate} | seed={data_cfg.seed} | "
        f"num_workers={data_cfg.num_workers} | "
        f"split=train:{split_summary['train']},test:{split_summary['test']} | "
        f"checkpoint={save_path}"
    )

    for epoch in range(1, train_cfg.epochs + 1):
        # ---- warmup linear LR ----
        if train_cfg.warmup_epochs > 0 and epoch <= train_cfg.warmup_epochs:
            warmup_lr = train_cfg.learning_rate * epoch / train_cfg.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # ---- training ----
        model.train()
        train_loss = 0.0
        train_m    = {k: 0.0 for k in ["Accuracy", "Dice", "IoU", "Precision", "Recall", "F1"]}
        n_steps    = 0

        optimizer.zero_grad()
        for step, (img, mask) in enumerate(loaders["train"]):
            img  = img.to(device)
            mask = mask.unsqueeze(1).to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    pred = model(img)
                    loss = criterion(pred, mask) / train_cfg.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                pred = model(img)
                loss = criterion(pred, mask) / train_cfg.grad_accum_steps
                loss.backward()

            if (step + 1) % train_cfg.grad_accum_steps == 0:
                if train_cfg.max_grad_norm:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if train_cfg.lr_scheduler == "onecycle":
                    main_scheduler.step()

            train_loss += loss.item() * train_cfg.grad_accum_steps
            m = compute_metrics(pred.detach(), mask)
            for k in train_m:
                train_m[k] += m[k]
            n_steps += 1

        train_loss /= n_steps
        for k in train_m:
            train_m[k] /= n_steps

        # ---- validation ----
        test_result = evaluate_model(model, loaders["test"], criterion, device)

        # ---- scheduler step ----
        if epoch > train_cfg.warmup_epochs:
            if train_cfg.lr_scheduler == "cosine":
                main_scheduler.step()
            elif train_cfg.lr_scheduler == "plateau":
                main_scheduler.step(test_result["loss"])

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_result["loss"])
        history["train_iou"].append(train_m["IoU"])
        history["test_iou"].append(test_result["IoU"])

        # ---- checkpoint ----
        saved = False
        if train_cfg.checkpoint_policy == "best" and test_result["IoU"] > best_iou:
            best_iou = test_result["IoU"]
            torch.save(model.state_dict(), save_path)
            saved = True
        elif train_cfg.checkpoint_policy == "last":
            torch.save(model.state_dict(), save_path)
            saved = True

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch}/{train_cfg.epochs} | lr={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} | train_iou={train_m['IoU']:.4f} | "
            f"test_loss={test_result['loss']:.4f} | test_iou={test_result['IoU']:.4f} | "
            f"checkpoint_saved={'yes' if saved else 'no'}"
        )

        # ---- early stopping ----
        if train_cfg.early_stopping_patience is not None:
            if test_result["IoU"] > best_iou:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= train_cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    return history, save_path


def visualize_predictions(model, dataset, num_samples=3, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    with torch.no_grad():
        for row, idx in enumerate(indices):
            img, mask = dataset[idx]
            pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))
            pred_bin = (pred > 0.5).float().squeeze().cpu()

            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * std + mean).clip(0, 1)

            axes[row][0].imshow(img_np);        axes[row][0].set_title("Image");        axes[row][0].axis("off")
            axes[row][1].imshow(mask.squeeze().cpu().numpy(), cmap="gray"); axes[row][1].set_title("Ground Truth"); axes[row][1].axis("off")
            axes[row][2].imshow(pred_bin.numpy(), cmap="gray"); axes[row][2].set_title("Prediction"); axes[row][2].axis("off")

    plt.tight_layout()
    plt.show()
