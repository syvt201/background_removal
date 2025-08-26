import torch

def dice_coef(logits, targets, threshold=0.5, smooth=1e-6):
    """Compute Dice Coefficient"""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    """Compute IoU Score"""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def precision_score(logits, targets, threshold=0.5, smooth=1e-6):
    """Compute Precision"""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    true_positive = (preds * targets).sum(dim=(1, 2, 3))
    predicted_positive = preds.sum(dim=(1, 2, 3))

    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision.mean().item()


def recall_score(logits, targets, threshold=0.5, smooth=1e-6):
    """Compute Recall"""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    true_positive = (preds * targets).sum(dim=(1, 2, 3))
    actual_positive = targets.sum(dim=(1, 2, 3))

    recall = (true_positive + smooth) / (actual_positive + smooth)
    return recall.mean().item()
