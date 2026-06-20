from operator import index
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch


import cv2
def generate_bpanno(gt_np, kernel_size=15, epsilon=2.0):
    """
    From a binary GT mask (H,W, values 0/1 float),
    produce inscribed mask, envelope mask, uncertain ring,
    and 3-class CCG label map.

    Returns all as float32 numpy arrays (H, W).
    y_c is int64.
    """
    # Work in uint8 [0,255]
    gt_u8 = (gt_np * 255).astype(np.uint8)

    # Adaptive kernel: at least 5, scales with lesion size
    area = gt_u8.sum() / 255
    if area > 0:
        radius = np.sqrt(area / np.pi)
        kernel_size = max(5, int(radius * 0.12))
        # ensure odd
        if kernel_size % 2 == 0:
            kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    dilated = cv2.dilate(gt_u8, kernel, iterations=1)   # envelope
    eroded  = cv2.erode(gt_u8,  kernel, iterations=1)   # inscribed

    def to_poly_mask(binary_u8):
        contours, _ = cv2.findContours(
            binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        out = np.zeros_like(binary_u8)
        if not contours:
            return out
        c = max(contours, key=cv2.contourArea)
        simplified = cv2.approxPolyDP(c, epsilon, closed=True)
        cv2.fillPoly(out, [simplified], 255)
        return out

    y_en_u8 = to_poly_mask(dilated)
    y_in_u8 = to_poly_mask(eroded)

    # Edge case: erosion collapsed to empty → use GT itself
    if y_in_u8.sum() == 0:
        y_in_u8 = gt_u8.copy()

    y_en = (y_en_u8 > 127).astype(np.float32)
    y_in = (y_in_u8 > 127).astype(np.float32)

    # Uncertain ring: inside envelope but outside inscribed
    omega_delta = np.clip(y_en - y_in, 0, 1).astype(np.float32)

    # 3-class label map for CCG
    # 0=certain bg, 1=uncertain ring, 2=certain fg
    y_c = np.zeros_like(y_in, dtype=np.int64)
    y_c[y_en == 1] = 1       # start: everything inside envelope = uncertain
    y_c[y_in == 1] = 2       # overwrite: inside inscribed = certain fg
    # outside envelope stays 0 = certain bg

    return y_in, y_en, omega_delta, y_c


class PolypDataset(data.Dataset):

    def __init__(self, image_root, gt_root, trainsize, augmentations, split='train', color_image=True):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.split = split
        self.color_image = color_image

        print(self.augmentations)

        self.images = [
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ]
        self.gts = [
            gt_root + f for f in os.listdir(gt_root)
            if f.endswith('.png') or f.endswith('.jpg')
        ]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

        if self.augmentations == 'True' or self.augmentations is True:
            print('Using RandomRotation, RandomFlip')

            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])

        else:
            print('no augmentation')

            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt    = self.binary_loader(self.gts[index])

        seed = np.random.randint(2147483647)
        random.seed(seed);  torch.manual_seed(seed)
        if self.img_transform is not None:
            image = self.img_transform(image)

        random.seed(seed);  torch.manual_seed(seed)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)          # tensor (1,H,W) float

        # ── BPAnno generation ─────────────────────────────────────────
        gt_np = gt.squeeze().cpu().numpy()      # (H,W) float [0,1]
        y_in, y_en, omega_delta, y_c = generate_bpanno(gt_np)

        y_in_t = torch.from_numpy(y_in).float()        # (H,W)
        y_en_t = torch.from_numpy(y_en).float()        # (H,W)
        omega_delta_t = torch.from_numpy(omega_delta).float() # (H,W)
        y_c_t = torch.from_numpy(y_c).long()          # (H,W)
        return image, y_in_t, y_en_t, omega_delta_t, y_c_t, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)

        images = []
        gts = []

        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)

            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)

        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size

        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return (
                img.resize((w, h), Image.BILINEAR),
                gt.resize((w, h), Image.NEAREST)
            )
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize,
               shuffle=False, num_workers=4,
               pin_memory=True, augmentation=False,
               split='train', color_image=True):

    dataset = PolypDataset(
        image_root, gt_root, trainsize,
        augmentation, split, color_image
    )

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize

        self.images = [
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ]

        self.gts = [
            gt_root + f for f in os.listdir(gt_root)
            if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')
        ]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.replace('.jpg', '.png')

        self.index += 1

        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')