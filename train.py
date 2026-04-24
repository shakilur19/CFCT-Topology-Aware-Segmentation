import os
from datetime import datetime
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cfct.dataset import SegDataset
from cfct.losses import DiceBCELoss, edge_loss, make_edge_targets, topology_loss
from cfct.model import HybridNet
from cfct.utils import AvgMeter, set_seed


def load_config(path='configs/default.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_paths(cfg):
    dataset = cfg['dataset']
    root = cfg['root']
    snapshot_path = os.path.join(cfg['snapshot_root'], dataset)
    return {
        'train_images': f'../{root}/{dataset}/train/images',
        'train_masks': f'../{root}/{dataset}/train/masks',
        'val_images': f'../{root}/{dataset}/val/images',
        'val_masks': f'../{root}/{dataset}/val/masks',
        'snapshot_path': snapshot_path
    }


def validate(model, val_loader, device):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out1, *_ = model(imgs)
            out1 = F.interpolate(out1, size=masks.shape[2:], mode='bilinear', align_corners=True)
            pred = torch.sigmoid(out1)
            intersection = (pred * masks).sum(dim=(2, 3))
            dice = (2 * intersection + 1e-6) / (pred.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + 1e-6)
            dice_scores.append(dice.mean().item())
    return sum(dice_scores) / len(dice_scores)


def train(train_loader, val_loader, model, optimizer, cfg, device):
    criterion = DiceBCELoss()
    print_freq = 20
    total_step = len(train_loader)
    best_dice = 0.0
    snapshot_path = cfg['snapshot_path']
    os.makedirs(snapshot_path, exist_ok=True)

    for epoch in range(1, cfg['num_epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        loss_record2 = AvgMeter()
        loss_record3 = AvgMeter()
        loss_record4 = AvgMeter()
        loss_record5 = AvgMeter()

        for i, (imgs, masks, _) in enumerate(train_loader, 1):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out1, out2, out3, out4, edge_map = model(imgs)
            gt_size = masks.shape[2:]
            out1_up = F.interpolate(out1, size=gt_size, mode='bilinear', align_corners=True)
            out2_up = F.interpolate(out2, size=gt_size, mode='bilinear', align_corners=True)
            out3_up = F.interpolate(out3, size=gt_size, mode='bilinear', align_corners=True)
            out4_up = F.interpolate(out4, size=gt_size, mode='bilinear', align_corners=True)
            edge_targets_down = make_edge_targets(masks, edge_map.shape[2:])
            loss_e = edge_loss(edge_map, edge_targets_down)
            loss1 = criterion(out1_up, masks)
            loss2 = criterion(out2_up, masks)
            loss3 = criterion(out3_up, masks)
            loss4 = criterion(out4_up, masks)
            topo = topology_loss(torch.sigmoid(out1_up), masks)
            loss = (loss1 + loss2 + loss3 + loss4) / 4 + cfg['edge_weight'] * loss_e + cfg['topology_weight'] * topo
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loss_record2.update(loss4.item(), imgs.size(0))
            loss_record3.update(loss3.item(), imgs.size(0))
            loss_record4.update(loss2.item(), imgs.size(0))
            loss_record5.update(loss1.item(), imgs.size(0))

            if i % print_freq == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-2: {:.4f}, lateral-3: {:.4f}, lateral-4: {:.4f}, lateral-5: {:.4f}]'.format(
                    datetime.now(), epoch, cfg['num_epochs'], i, total_step,
                    loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()
                ))

        avg_loss = epoch_loss / len(train_loader)
        avg_dice = validate(model, val_loader, device) if val_loader is not None else None

        if avg_dice is not None and avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), os.path.join(snapshot_path, f"{cfg['model_name']}.pth"))
            print(f"Model saved with improved Dice: {best_dice:.4f}")

        if epoch == cfg['num_epochs']:
            torch.save(model.state_dict(), os.path.join(snapshot_path, f"{cfg['model_name']}_last.pth"))
            print(f"Final epoch model saved as {cfg['model_name']}_last.pth")

        now = datetime.now()
        if avg_dice is not None:
            print(f"[{now}] Epoch {epoch:03d}/{cfg['num_epochs']} | TrainLoss={avg_loss:.4f} | Val Dice={avg_dice:.4f}")
        else:
            print(f"[{now}] Epoch {epoch:03d}/{cfg['num_epochs']} | TrainLoss={avg_loss:.4f}")


def main():
    cfg = load_config()
    paths = build_paths(cfg)
    cfg.update(paths)
    set_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = SegDataset(cfg['train_images'], cfg['train_masks'], cfg['input_size'])
    val_ds = SegDataset(cfg['val_images'], cfg['val_masks'], cfg['input_size'])
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers_train'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg['num_workers_val'])
    model = HybridNet(cfg['backbone_name'], cfg['out_channels'], cfg['pretrained']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    resume_path = os.path.join(cfg['snapshot_path'], f"{cfg['model_name']}_last.pth")

    if os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device))

    train(train_loader, val_loader, model, optimizer, cfg, device)


if __name__ == '__main__':
    main()
