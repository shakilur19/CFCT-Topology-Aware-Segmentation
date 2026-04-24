import os
import yaml
import torch
import numpy as np
from skimage import img_as_ubyte, io

from cfct.dataset import TestDataset
from cfct.metrics import compute_metrics
from cfct.model import HybridNet


def load_config(path='configs/default.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def predict(model, model_path, test_image_root, output_path, device, test_size=352):
    os.makedirs(output_path, exist_ok=True)
    test_dataset = TestDataset(test_image_root, test_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, name, _ = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            outputs = model(image)
            res = outputs[3] if isinstance(outputs, (list, tuple)) else outputs
            res = res.sigmoid().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res > 0.5).astype(np.uint8) * 255
            io.imsave(os.path.join(output_path, name), img_as_ubyte(res))


def main():
    cfg = load_config()
    dataset = cfg['dataset']
    root = cfg['root']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    snapshot_path = os.path.join(cfg['snapshot_root'], dataset)
    test_image_root = f'../{root}/{dataset}/test/images'
    test_mask_root = f'../{root}/{dataset}/test/masks'
    model_path = os.path.join(snapshot_path, f"{cfg['model_name']}.pth")
    output_path = f'../{root}/{dataset}/outputs/{dataset}/{cfg["model_name"]}'
    model = HybridNet(cfg['backbone_name'], cfg['out_channels'], pretrained=False).to(device)
    predict(model, model_path, test_image_root, output_path, device, cfg['input_size'])
    metrics = compute_metrics(test_mask_root, output_path, size=(cfg['input_size'], cfg['input_size']))
    print(f"Average IoU: {metrics['iou']:.4f}")
    print(f"Average Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Average Dice: {metrics['dice']:.4f}")


if __name__ == '__main__':
    main()
