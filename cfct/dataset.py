import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


VALID_EXTENSIONS = ('.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif')


class SegDataset(Dataset):
    def __init__(self, image_root, mask_root, size, return_name=True):
        self.images = sorted([
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.lower().endswith(VALID_EXTENSIONS)
        ])
        self.masks = sorted([
            os.path.join(mask_root, f)
            for f in os.listdir(mask_root)
            if f.lower().endswith(VALID_EXTENSIONS)
        ])
        if len(self.images) != len(self.masks):
            raise ValueError(f"Image/mask count mismatch: {len(self.images)} images, {len(self.masks)} masks")
        self.size = size
        self.return_name = return_name
        self.transform_img = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        img = self.transform_img(img)
        mask = self.transform_mask(mask)
        if self.return_name:
            return img, mask, os.path.basename(img_path)
        return img, mask


class TestDataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = sorted([
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.lower().endswith(VALID_EXTENSIONS)
        ])
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        name = os.path.splitext(os.path.basename(img_path))[0] + '.png'
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        image = self.transform(image)
        return image, name, original_size
