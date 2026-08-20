import torch
from utils.losses import BCEDiceLoss  # re-export — notebooks import BCEDiceLoss from here


def _get_stats(pred, target, threshold=0.5, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    target = target.float()

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()

    return tp, fp, fn, tn, smooth


def accuracy_score(pred, target, threshold=0.5, smooth=1e-6):
    tp, fp, fn, tn, s = _get_stats(pred, target, threshold, smooth)
    return (tp + tn + s) / (tp + tn + fp + fn + s)


def precision_score(pred, target, threshold=0.5, smooth=1e-6):
    tp, fp, fn, tn, s = _get_stats(pred, target, threshold, smooth)
    return (tp + s) / (tp + fp + s)


def recall_score(pred, target, threshold=0.5, smooth=1e-6):
    tp, fp, fn, tn, s = _get_stats(pred, target, threshold, smooth)
    return (tp + s) / (tp + fn + s)


def f1_score(pred, target, threshold=0.5, smooth=1e-6):
    tp, fp, fn, tn, s = _get_stats(pred, target, threshold, smooth)
    return (2 * tp + s) / (2 * tp + fp + fn + s)


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    return f1_score(pred, target, threshold, smooth)


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    tp, fp, fn, tn, s = _get_stats(pred, target, threshold, smooth)
    return (tp + s) / (tp + fp + fn + s)


def compute_metrics(pred, target, threshold=0.5, smooth=1e-6):
    tp, fp, fn, tn, s = _get_stats(pred, target, threshold, smooth)

    acc  = (tp + tn + s) / (tp + tn + fp + fn + s)
    prec = (tp + s) / (tp + fp + s)
    rec  = (tp + s) / (tp + fn + s)
    f1   = (2 * tp + s) / (2 * tp + fp + fn + s)
    iou  = (tp + s) / (tp + fp + fn + s)

    return {
        "Accuracy":  acc.item(),
        "Dice":      f1.item(),
        "IoU":       iou.item(),
        "Precision": prec.item(),
        "Recall":    rec.item(),
        "F1":        f1.item(),
    }
