import numpy as np
import torch

from utils.losses import BCEDiceLoss
from utils.metrics import compute_metrics


def report_tuned_metrics(model, loaders, use_tta=False):
    """
    Tìm ngưỡng tối ưu trên tập train, sau đó báo cáo toàn bộ 6 độ đo trên tập test.
    Trả về dict kết quả (có trường 'threshold' để lưu cho app).
    """
    device = next(model.parameters()).device
    criterion = BCEDiceLoss()
    model.eval()

    def _collect(loader, tta):
        preds, masks, total_loss, n = [], [], 0.0, 0
        with torch.no_grad():
            for img, mask in loader:
                img  = img.to(device)
                mask = mask.unsqueeze(1).to(device)
                pred = model(img)

                if tta:
                    pred_h = model(torch.flip(img, [3]));  pred_h = torch.flip(pred_h, [3])
                    pred_v = model(torch.flip(img, [2]));  pred_v = torch.flip(pred_v, [2])
                    pred   = (pred + pred_h + pred_v) / 3.0

                total_loss += criterion(pred, mask).item()
                n += 1
                preds.append(torch.sigmoid(pred).cpu())
                masks.append(mask.cpu())
        return torch.cat(preds), torch.cat(masks), total_loss / n

    # Tìm ngưỡng tối ưu trên tập train (không dùng TTA để nhanh)
    train_preds, train_masks, _ = _collect(loaders["train"], tta=False)
    best_threshold, best_iou = 0.5, 0.0
    for t in np.arange(0.1, 0.90, 0.05):
        pred_bin = (train_preds > t).float()
        tp = (pred_bin * train_masks).sum()
        fp = (pred_bin * (1 - train_masks)).sum()
        fn = ((1 - pred_bin) * train_masks).sum()
        iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
        if iou.item() > best_iou:
            best_iou = iou.item()
            best_threshold = float(t)

    # Đánh giá trên tập test với ngưỡng tối ưu
    test_preds, test_masks, avg_loss = _collect(loaders["test"], tta=use_tta)
    pred_logits = torch.logit(test_preds.clamp(1e-6, 1 - 1e-6))
    metrics = compute_metrics(pred_logits, test_masks, threshold=best_threshold)

    metrics["loss"]      = avg_loss
    metrics["threshold"] = best_threshold
    metrics["tta"]       = use_tta

    # In bảng kết quả
    tta_label = " (TTA)" if use_tta else ""
    print("\n" + "=" * 52)
    print(f"  Kết quả đánh giá — tập test{tta_label}")
    print("=" * 52)
    print(f"  {'Độ đo':<15} {'Giá trị':>10}")
    print("-" * 52)
    for key in ["Accuracy", "Dice", "IoU", "Precision", "Recall", "F1"]:
        print(f"  {key:<15} {metrics[key]:>10.4f}")
    print("-" * 52)
    print(f"  {'Loss':<15} {avg_loss:>10.4f}")
    print(f"  {'Threshold':<15} {best_threshold:>10.4f}")
    print("=" * 52 + "\n")

    return metrics
