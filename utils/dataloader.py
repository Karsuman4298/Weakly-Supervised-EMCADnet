# dataloader.py — MRPAnno version with PRC support
# Ground truth is NEVER used for training supervision.
# GT is returned only for validation/test metric computation.

from operator import index
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2
from scipy.ndimage import distance_transform_edt

# ─────────────────────────────────────────────────────────────
# MRPAnno Configuration
# ─────────────────────────────────────────────────────────────
MRPANNO_DIL_SCALE    = 0.40
MRPANNO_ERO_SCALE    = 0.30
MRPANNO_MID_SCALE    = 0.08
MRPANNO_DP_RATIO     = 0.02
MRPANNO_MIN_VERTICES = 5
MRPANNO_MAX_VERTICES = 15

# Fixed pseudo-label values (replaces distance-sigmoid)
ALPHA    = 0.40    # Ω_Δ1 zone weight in loss
BETA     = 0.15    # Ω_Δ2 zone weight in loss
TGT_D1   = 0.85   # Ω_Δ1 fixed pseudo-label target
TGT_D2   = 0.15   # Ω_Δ2 fixed pseudo-label target


# ─────────────────────────────────────────────────────────────
# Polygon utilities
# ─────────────────────────────────────────────────────────────

def _morph(binary_u8, ksize, op):
    if ksize < 1:
        ksize = 1
    if ksize % 2 == 0:
        ksize += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if op == 'dilate':
        return cv2.dilate(binary_u8, k, iterations=1)
    return cv2.erode(binary_u8, k, iterations=1)


def _to_poly_mask(binary_u8, H, W,
                  dp_ratio=MRPANNO_DP_RATIO,
                  min_v=MRPANNO_MIN_VERTICES,
                  max_v=MRPANNO_MAX_VERTICES):
    contours, _ = cv2.findContours(
        binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    out = np.zeros((H, W), dtype=np.uint8)
    if not contours:
        return out

    contour   = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter < 10:
        return out

    epsilon = perimeter * dp_ratio
    polygon = cv2.approxPolyDP(contour, epsilon, closed=True)

    for _ in range(15):
        if len(polygon) <= max_v:
            break
        epsilon *= 1.25
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)

    for _ in range(15):
        if len(polygon) >= min_v:
            break
        epsilon *= 0.75
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)

    if len(polygon) >= 3:
        cv2.fillPoly(out, [polygon], 255)
    return out


def generate_mrpanno(gt_np, dp_epsilon_ratio=MRPANNO_DP_RATIO):
    """
    Generate MRPAnno polygon masks from binary GT mask.

    Changes vs previous version:
    - softlabel removed (no longer needed — PRC handles targets in train loop)
    - y_c5 replaced by y_c3 (3-class CCG: BG=0 / uncertain=1 / FG=2)
    - pmid_strip kept for contrastive anchor selection

    Returns
    ───────
    y_in        (H,W) float32   P_in fill   — certain FG mask
    y_mid       (H,W) float32   P_mid fill  — boundary polygon fill
    y_out       (H,W) float32   P_out fill  — envelope mask
    omega_d1    (H,W) float32   Ω_Δ1 mask  (inner uncertain band)
    omega_d2    (H,W) float32   Ω_Δ2 mask  (outer uncertain band)
    pmid_strip  (H,W) float32   thin strip ±3px around P_mid boundary
    y_c3        (H,W) int64     3-class CCG label map
                                  0 = Ω_O   (certain BG)
                                  1 = Ω_Δ   (uncertain — both bands)
                                  2 = Ω_I   (certain FG)
    """
    gt_u8 = (gt_np * 255).astype(np.uint8)
    H, W  = gt_np.shape

    # ── Adaptive kernel sizes ─────────────────────────────────
    area   = float((gt_u8 > 127).sum())
    radius = np.sqrt(max(area, 1.0) / np.pi)

    dil_ks = max(7, int(radius * MRPANNO_DIL_SCALE))
    ero_ks = max(5, int(radius * MRPANNO_ERO_SCALE))
    mid_ks = max(3, int(radius * MRPANNO_MID_SCALE))

    dil_ks = dil_ks + (dil_ks % 2 == 0)
    ero_ks = ero_ks + (ero_ks % 2 == 0)
    mid_ks = mid_ks + (mid_ks % 2 == 0)

    # ── Small lesion fallback check ───────────────────────────
    lesion_diameter = 2.0 * radius

    # ── Three morphological bases ─────────────────────────────
    dilated = _morph(gt_u8, dil_ks, 'dilate')
    eroded  = _morph(gt_u8, ero_ks, 'erode')
    midbase = _morph(gt_u8, mid_ks, 'dilate')

    # ── Polygon fills ─────────────────────────────────────────
    p_out_u8 = _to_poly_mask(dilated, H, W, dp_epsilon_ratio)
    p_in_u8  = _to_poly_mask(eroded,  H, W, dp_epsilon_ratio)
    p_mid_u8 = _to_poly_mask(midbase, H, W, dp_epsilon_ratio)

    # Fallback: erosion collapsed
    if p_in_u8.sum() == 0:
        if gt_u8.sum() > 0:
            p_in_u8 = gt_u8.copy()

    # Containment enforcement
    p_in_u8  = np.where(
        (p_out_u8 > 0) & (p_mid_u8 > 0), p_in_u8, 0
    ).astype(np.uint8)
    p_mid_u8 = np.where(p_out_u8 > 0, p_mid_u8, 0).astype(np.uint8)

    # Float masks
    y_out = (p_out_u8 > 127).astype(np.float32)
    y_mid = (p_mid_u8 > 127).astype(np.float32)
    y_in  = (p_in_u8  > 127).astype(np.float32)

    # ── Zone masks ────────────────────────────────────────────
    omega_I  = y_in.astype(bool)
    omega_d1 = y_mid.astype(bool) & ~omega_I
    omega_d2 = y_out.astype(bool) & ~y_mid.astype(bool)

    # ── P_mid boundary strip (±3px) ───────────────────────────
    p_mid_u8_bool = y_mid.astype(np.uint8)
    strip_dil     = _morph(p_mid_u8_bool, 7, 'dilate')
    strip_ero     = _morph(p_mid_u8_bool, 7, 'erode')
    pmid_strip    = ((strip_dil > 0) & (strip_ero == 0)).astype(np.float32)

    # ── Small lesion fallback ─────────────────────────────────
    if lesion_diameter < 12.0:
        # Collapse to BPAnno: merge bands, wipe strip
        omega_d2   = (omega_d1 | omega_d2).astype(np.float32)
        omega_d1   = np.zeros_like(omega_d1, dtype=np.float32)
        pmid_strip = np.zeros_like(pmid_strip, dtype=np.float32)
    else:
        omega_d1 = omega_d1.astype(np.float32)
        omega_d2 = omega_d2.astype(np.float32)

    # ── 3-class CCG label map ─────────────────────────────────
    # Class 0: Ω_O  (certain background — outside P_out)
    # Class 1: Ω_Δ  (uncertain — inside P_out, outside P_in)
    # Class 2: Ω_I  (certain foreground — inside P_in)
    y_c3 = np.zeros((H, W), dtype=np.int64)
    y_c3[y_out.astype(bool)] = 1   # uncertain (overridden below)
    y_c3[y_in.astype(bool)]  = 2   # certain FG

    return (
        y_in.astype(np.float32),       # certain FG
        y_mid.astype(np.float32),      # P_mid fill
        y_out.astype(np.float32),      # envelope
        omega_d1.astype(np.float32),   # Ω_Δ1
        omega_d2.astype(np.float32),   # Ω_Δ2
        pmid_strip.astype(np.float32), # P_mid strip
        y_c3,                          # 3-class CCG
    )


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class PolypDataset(data.Dataset):

    def __init__(self, image_root, gt_root, trainsize,
                 augmentations, split='train', color_image=True):
        self.trainsize     = trainsize
        self.augmentations = augmentations
        self.split         = split
        self.color_image   = color_image

        print(self.augmentations)

        self.images = sorted([
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        self.gts = sorted([
            gt_root + f for f in os.listdir(gt_root)
            if f.endswith('.png') or f.endswith('.jpg')
        ])

        self.filter_files()
        self.size = len(self.images)

        if self.augmentations == 'True' or self.augmentations is True:
            print('Using RandomRotation, RandomFlip, ColorJitter')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(          # NEW
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.1
                ),
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
        image  = self.rgb_loader(self.images[index])
        gt_pil = self.binary_loader(self.gts[index])

        # Synchronised augmentation
        seed = np.random.randint(2147483647)
        random.seed(seed);  torch.manual_seed(seed)
        image = self.img_transform(image)
        random.seed(seed);  torch.manual_seed(seed)
        gt_t  = self.gt_transform(gt_pil)

        gt_np = gt_t.squeeze().cpu().numpy()
        (y_in, y_mid, y_out,
         omega_d1, omega_d2,
         pmid_strip, y_c3) = generate_mrpanno(gt_np)

        return (
            image,                                         # (3,H,W)
            torch.from_numpy(y_in).float(),                # certain FG
            torch.from_numpy(y_mid).float(),               # P_mid fill
            torch.from_numpy(y_out).float(),               # envelope
            torch.from_numpy(omega_d1).float(),            # Ω_Δ1
            torch.from_numpy(omega_d2).float(),            # Ω_Δ2
            torch.from_numpy(pmid_strip).float(),          # P_mid strip
            torch.from_numpy(y_c3).long(),                 # 3-class CCG
            gt_t,                                          # GT — val only
        )

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt  = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts    = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

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
    return data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


# ─────────────────────────────────────────────────────────────
# Test dataset
# ─────────────────────────────────────────────────────────────

class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = sorted([
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        self.gts = sorted([
            gt_root + f for f in os.listdir(gt_root)
            if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')
        ])
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size  = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt    = self.binary_loader(self.gts[self.index])
        name  = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.replace('.jpg', '.png')
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')