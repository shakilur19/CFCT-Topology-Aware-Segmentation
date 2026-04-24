import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gudhi as gd


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        preds = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (preds * targets).sum(dim=(2, 3))
        dice = 1 - (2 * intersection + smooth) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
        return bce + dice.mean()


def edge_loss(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target)


def min_pool2d(x, kernel_size, stride, padding):
    return -F.max_pool2d(-x, kernel_size, stride, padding)


def make_edge_targets(masks, edge_map_shape):
    edge_targets = F.max_pool2d(masks, kernel_size=3, stride=1, padding=1) - min_pool2d(masks, kernel_size=3, stride=1, padding=1)
    edge_targets = (edge_targets > 0.1).float()
    return F.interpolate(edge_targets, size=edge_map_shape, mode='nearest')


def topology_loss(pred_mask, gt_mask):
    pred_np = pred_mask.detach().cpu().numpy().squeeze()
    gt_np = gt_mask.detach().cpu().numpy().squeeze()
    pred_np = (pred_np > 0.5).astype(np.float32)
    gt_np = (gt_np > 0.5).astype(np.float32)
    pred_cc = gd.CubicalComplex(top_dimensional_cells=pred_np)
    gt_cc = gd.CubicalComplex(top_dimensional_cells=gt_np)
    pred_cc.compute_persistence()
    gt_cc.compute_persistence()
    pred_dgm = pred_cc.persistence()
    gt_dgm = gt_cc.persistence()
    pred_betti_1 = sum(1 for b in pred_dgm if b[0] == 1)
    gt_betti_1 = sum(1 for b in gt_dgm if b[0] == 1)
    loss = abs(pred_betti_1 - gt_betti_1)
    return torch.tensor(loss, dtype=pred_mask.dtype, device=pred_mask.device)
