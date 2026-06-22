# dataloader.py  — MRPAnno version
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
# P_out  : GT dilated  → loose envelope  (same as BPAnno outer)
# P_mid  : GT itself   → boundary approximation (NEW)
# P_in   : GT eroded   → conservative interior  (same as BPAnno inner)
#
# Dilation scale kept same as BPAnno for P_out.
# Erosion scale increased slightly vs BPAnno so that P_in sits
# clearly inside P_mid, giving a non-degenerate Ω_Δ1 band.
# ─────────────────────────────────────────────────────────────
MRPANNO_DIL_SCALE    = 0.40    # P_out dilation   (same as BPAnno)
MRPANNO_ERO_SCALE    = 0.30    # P_in  erosion    (larger than BPAnno 0.22)
MRPANNO_MID_SCALE    = 0.08    # P_mid tiny smoothing dilation before contour
MRPANNO_DP_RATIO     = 0.02    # Douglas-Peucker: 2 % of perimeter
MRPANNO_MIN_VERTICES = 5
MRPANNO_MAX_VERTICES = 15

# Soft-label weights for the two uncertain bands
ALPHA = 0.70   # Ω_Δ1 (inner band, likely FG) pseudo-label
BETA  = 0.30   # Ω_Δ2 (outer band, likely BG) pseudo-label

# Bandwidth for distance-sigmoid soft label (pixels)
SIGMA_SOFT = 8.0


# ─────────────────────────────────────────────────────────────
# Polygon utilities
# ─────────────────────────────────────────────────────────────

def _morph(binary_u8, ksize, op):
    """Apply morphological operation with elliptic kernel."""
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
    """
    Largest-contour → Douglas-Peucker polygon (5-15 vertices) → filled mask.
    Returns uint8 mask (H, W).
    """
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

    # Too many vertices → increase epsilon
    for _ in range(15):
        if len(polygon) <= max_v:
            break
        epsilon *= 1.25
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)

    # Too few vertices → decrease epsilon
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
    GT is used ONLY to derive the three polygon shapes here —
    it is never passed to the training loss directly.

    Zone definitions
    ────────────────
    Ω_I   : inside P_in           → certain foreground
    Ω_Δ1  : P_in  → P_mid        → inner uncertain (likely FG, weight α)
    Ω_Δ2  : P_mid → P_out        → outer uncertain (likely BG, weight β)
    Ω_O   : outside P_out         → certain background

    Returns
    ───────
    y_in        (H,W) float32   P_in fill   — certain FG mask
    y_mid       (H,W) float32   P_mid fill  — boundary polygon fill (NEW)
    y_out       (H,W) float32   P_out fill  — envelope mask
    omega_d1    (H,W) float32   Ω_Δ1 mask  (inner uncertain band)
    omega_d2    (H,W) float32   Ω_Δ2 mask  (outer uncertain band)
    softlabel   (H,W) float32   distance-sigmoid soft label ∈ [0,1]
    pmid_strip  (H,W) float32   thin strip ±3px around P_mid boundary
    y_c5        (H,W) int64     5-class CCG label map
                                  0 = Ω_O   (certain BG)
                                  1 = Ω_Δ2  (outer uncertain)
                                  2 = P_mid strip (boundary zone)
                                  3 = Ω_Δ1  (inner uncertain)
                                  4 = Ω_I   (certain FG)
    """
    gt_u8 = (gt_np * 255).astype(np.uint8)
    H, W  = gt_np.shape

    # ── Adaptive kernel sizes from lesion radius ──────────────
    area   = float((gt_u8 > 127).sum())
    radius = np.sqrt(max(area, 1.0) / np.pi)

    dil_ks = max(7,  int(radius * MRPANNO_DIL_SCALE))
    ero_ks = max(5,  int(radius * MRPANNO_ERO_SCALE))
    mid_ks = max(3,  int(radius * MRPANNO_MID_SCALE))  # tiny smooth for P_mid

    # force odd
    dil_ks = dil_ks + (dil_ks % 2 == 0)
    ero_ks = ero_ks + (ero_ks % 2 == 0)
    mid_ks = mid_ks + (mid_ks % 2 == 0)

    # ── Three morphological bases ─────────────────────────────
    dilated = _morph(gt_u8, dil_ks, 'dilate')   # P_out base
    eroded  = _morph(gt_u8, ero_ks, 'erode')    # P_in  base
    midbase = _morph(gt_u8, mid_ks, 'dilate')   # P_mid base (tiny smooth)

    # ── Polygon fills ─────────────────────────────────────────
    p_out_u8 = _to_poly_mask(dilated, H, W, dp_epsilon_ratio)
    p_in_u8  = _to_poly_mask(eroded,  H, W, dp_epsilon_ratio)
    p_mid_u8 = _to_poly_mask(midbase, H, W, dp_epsilon_ratio)

    # Fallback: erosion collapsed → use GT as P_in
    if p_in_u8.sum() == 0:
        p_in_u8 = gt_u8.copy()

    # Containment enforcement
    # P_in must be inside P_mid must be inside P_out
    p_in_u8  = np.where((p_out_u8 > 0) & (p_mid_u8 > 0), p_in_u8, 0).astype(np.uint8)
    p_mid_u8 = np.where(p_out_u8 > 0, p_mid_u8, 0).astype(np.uint8)

    # Float masks
    y_out = (p_out_u8 > 127).astype(np.float32)
    y_mid = (p_mid_u8 > 127).astype(np.float32)
    y_in  = (p_in_u8  > 127).astype(np.float32)

    # ── Zone masks ────────────────────────────────────────────
    # Ω_I   = inside P_in
    omega_I  = y_in.astype(bool)
    # Ω_Δ1  = inside P_mid but outside P_in
    omega_d1 = (y_mid.astype(bool)) & (~omega_I)
    # Ω_Δ2  = inside P_out but outside P_mid
    omega_d2 = (y_out.astype(bool)) & (~y_mid.astype(bool))
    # Ω_O   = outside P_out  (complement of y_out)

    # ── P_mid boundary strip (±3 px) ─────────────────────────
    p_mid_bool = y_mid.astype(np.uint8)
    strip_dil  = _morph(p_mid_bool, 7, 'dilate')   # 3px outward
    strip_ero  = _morph(p_mid_bool, 7, 'erode')    # 3px inward
    pmid_strip = ((strip_dil > 0) & (strip_ero == 0)).astype(np.float32)

    # ── Distance-sigmoid soft label ───────────────────────────
    # Signed distance from P_mid polygon boundary
    dist_in  =  distance_transform_edt(y_mid).astype(np.float32)
    dist_out = -distance_transform_edt(1.0 - y_mid).astype(np.float32)
    signed   = np.where(y_mid > 0.5, dist_in, dist_out)
    softlabel = (1.0 / (1.0 + np.exp(-signed / SIGMA_SOFT))).astype(np.float32)

    # ── 5-class CCG label map ─────────────────────────────────
    y_c5 = np.zeros((H, W), dtype=np.int64)
    y_c5[y_out.astype(bool)]   = 1   # Ω_Δ2 (overridden below)
    y_c5[y_mid.astype(bool)]   = 3   # Ω_Δ1 (overridden below)
    y_c5[omega_I]              = 4   # Ω_I
    y_c5[pmid_strip.astype(bool) & y_out.astype(bool)] = 2  # P_mid strip

    return (
        y_in.astype(np.float32),          # certain FG
        y_mid.astype(np.float32),         # P_mid fill
        y_out.astype(np.float32),         # envelope
        omega_d1.astype(np.float32),      # Ω_Δ1
        omega_d2.astype(np.float32),      # Ω_Δ2
        softlabel,                         # soft label
        pmid_strip.astype(np.float32),    # P_mid boundary strip
        y_c5,                              # 5-class CCG labels
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

        # GT loaded only to derive polygon masks — never used as training label
        gt_pil = self.binary_loader(self.gts[index])

        # Synchronised augmentation
        seed = np.random.randint(2147483647)
        random.seed(seed);  torch.manual_seed(seed)
        image = self.img_transform(image)       # (3, H, W)
        random.seed(seed);  torch.manual_seed(seed)
        gt_t  = self.gt_transform(gt_pil)      # (1, H, W) float [0,1]

        # MRPAnno generation from the augmented, resized GT
        gt_np = gt_t.squeeze().cpu().numpy()   # (H, W)
        (y_in, y_mid, y_out,
         omega_d1, omega_d2,
         softlabel, pmid_strip,
         y_c5) = generate_mrpanno(gt_np)

        return (
            image,                                        # (3,H,W)
            torch.from_numpy(y_in).float(),               # certain FG
            torch.from_numpy(y_mid).float(),              # P_mid fill
            torch.from_numpy(y_out).float(),              # envelope
            torch.from_numpy(omega_d1).float(),           # Ω_Δ1
            torch.from_numpy(omega_d2).float(),           # Ω_Δ2
            torch.from_numpy(softlabel).float(),          # soft label
            torch.from_numpy(pmid_strip).float(),         # P_mid strip
            torch.from_numpy(y_c5).long(),                # 5-class CCG
            gt_t,                                         # GT — val/test only
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
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

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


# ─────────────────────────────────────────────────────────────
# Test dataset  (unchanged — GT used for metrics only)
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