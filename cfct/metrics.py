import os
import numpy as np
from PIL import Image


VALID_EXTENSIONS = ('.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif')


def dice_score(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    total = gt.sum() + pred.sum()
    if total == 0:
        return 1.0
    return 2 * intersection / total


def load_binary_mask(path, size=(352, 352)):
    img = Image.open(path).convert('L')
    img = img.resize(size, Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)


def compute_metrics(gt_folder, pred_folder, size=(352, 352)):
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.lower().endswith(VALID_EXTENSIONS)])
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.lower().endswith(VALID_EXTENSIONS)])
    if len(gt_files) != len(pred_files):
        raise ValueError(f"Mismatch in file count: {len(gt_files)} ground-truth masks, {len(pred_files)} predictions")
    total_iou = 0.0
    total_acc = 0.0
    total_dice = 0.0
    n = len(gt_files)
    for gt_file, pred_file in zip(gt_files, pred_files):
        gt = load_binary_mask(os.path.join(gt_folder, gt_file), size)
        pred = load_binary_mask(os.path.join(pred_folder, pred_file), size)
        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        correct = (gt == pred).sum()
        total = gt.size
        total_iou += intersection / union if union != 0 else 1.0
        total_acc += correct / total
        total_dice += dice_score(gt, pred)
    return {
        'iou': total_iou / n,
        'pixel_accuracy': total_acc / n,
        'dice': total_dice / n
    }
