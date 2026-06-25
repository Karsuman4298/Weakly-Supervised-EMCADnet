# dataloader.py  —  HUPAnno dataloader
# Returns per sample:
#   image         (3, H, W)
#   y_in          (H, W) float  — global P_in mask  (certain FG = Omega_I)
#   y_out         (H, W) float  — global P_out mask (outer envelope)
#   omega_delta   (H, W) float  — global uncertain ring (P_out - P_in)
#   lrp_fg        (H, W) float  — LRP resolved foreground  (Omega_RF)
#   lrp_bg        (H, W) float  — LRP resolved background  (Omega_RB)
#   lrp_uncertain (H, W) float  — LRP uncertain strip
#   lrp_mask      (H, W) float  — binary: 1 where any LRP patch exists
#   y_c           (H, W) int64  — 4-class CCG label map
#                                  0 = certain BG  (outside P_out or Omega_RB)
#                                  1 = uncertain   (global ring, non-LRP)
#                                  2 = LRP uncertain (inside patch, between tight rings)
#                                  3 = certain FG  (inside P_in or Omega_RF)
#   gt            (1, H, W) float — original GT for validation

import os
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import random
import cv2
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Global polygon scales (same as BPAnno — fast, coarse)
HUP_DIL_SCALE = 0.40
HUP_ERO_SCALE = 0.22
HUP_DP_RATIO  = 0.02
HUP_MIN_VTX   = 5
HUP_MAX_VTX   = 15

# LRP settings
HUP_K_PATCHES = 2      # number of hard segments per image
HUP_PATCH_ARC = 0.12   # fraction of contour perimeter per patch
HUP_LRP_DIL   = 0.08   # tight dilation scale for LRP outer ring
HUP_LRP_ERO   = 0.06   # tight erosion scale for LRP inner ring
HUP_MIN_AREA  = 50     # skip lesions smaller than this (px²)


# ---------------------------------------------------------------------------
# Polygon helper
# ---------------------------------------------------------------------------

def to_poly_mask(binary_u8, H, W,
                 dp_ratio=HUP_DP_RATIO,
                 min_vtx=HUP_MIN_VTX,
                 max_vtx=HUP_MAX_VTX):
    """Convert a binary mask to a filled polygon mask using Douglas-Peucker."""
    contours, _ = cv2.findContours(
        binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    out = np.zeros((H, W), dtype=np.uint8)
    if not contours:
        return out
    cnt       = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter < 10:
        return out
    epsilon = perimeter * dp_ratio
    polygon = cv2.approxPolyDP(cnt, epsilon, True)
    for _ in range(12):
        if len(polygon) > max_vtx:
            epsilon *= 1.25
            polygon  = cv2.approxPolyDP(cnt, epsilon, True)
        else:
            break
    for _ in range(12):
        if len(polygon) < min_vtx:
            epsilon *= 0.75
            polygon  = cv2.approxPolyDP(cnt, epsilon, True)
        else:
            break
    if len(polygon) >= 3:
        cv2.fillPoly(out, [polygon], 255)
    return out


# ---------------------------------------------------------------------------
# Hard segment detection
# ---------------------------------------------------------------------------

def find_hard_segments(gt_u8, K=HUP_K_PATCHES, patch_arc=HUP_PATCH_ARC):
    """
    Find K hardest boundary segments on the GT contour using curvature.

    High curvature = sharp corner / narrow protrusion = genuinely hard.

    Returns:
        segments : list of (center_idx, start_idx, end_idx) — non-overlapping
        cnt      : the contour array  shape (N,1,2)
    Returns ([], None) if contour cannot be found.
    """
    contours, _ = cv2.findContours(
        gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return [], None
    cnt = max(contours, key=cv2.contourArea)
    N   = len(cnt)
    if N < 20:
        return [], cnt

    pts = cnt.reshape(-1, 2).astype(np.float32)   # (N, 2)

    # --- curvature: 1 - cos(angle between successive tangent vectors) -------
    scores = np.zeros(N)
    for i in range(N):
        p0 = pts[(i - 2) % N]
        p1 = pts[i]
        p2 = pts[(i + 2) % N]
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cos_a    = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
        scores[i] = 1.0 - cos_a   # 0 = straight, 2 = U-turn

    # smooth to avoid adjacent pixels on the same corner
    kernel = np.ones(5) / 5.0
    scores = np.convolve(scores, kernel, mode='same')

    # --- greedy non-overlapping peak selection ------------------------------
    half_patch = max(1, int(N * patch_arc / 2))
    selected   = []
    suppressed = np.zeros(N, dtype=bool)

    for _ in range(K):
        masked = scores.copy()
        masked[suppressed] = -1.0
        if masked.max() < 0:
            break
        center = int(np.argmax(masked))
        start  = (center - half_patch) % N
        end    = (center + half_patch) % N
        selected.append((center, start, end))
        for offset in range(-half_patch * 2, half_patch * 2 + 1):
            suppressed[(center + offset) % N] = True

    return selected, cnt


# ---------------------------------------------------------------------------
# LRP mask generation
# ---------------------------------------------------------------------------

def generate_lrp_masks(gt_u8, H, W, K=HUP_K_PATCHES):
    """
    Generate Local Refinement Patch masks at the K hardest boundary segments.

    For each hard segment:
      - crop a bounding box from the GT
      - apply TIGHT erosion  (much smaller than global P_in kernel)
      - apply TIGHT dilation (much smaller than global P_out kernel)
      - convert to polygon masks within the patch
      - paste back into full-image maps

    Returns:
      lrp_fg        float32 (H,W) — resolved FG inside patches    (Omega_RF)
      lrp_bg        float32 (H,W) — resolved BG inside patches    (Omega_RB)
      lrp_uncertain float32 (H,W) — uncertain strip inside patches
      lrp_mask      float32 (H,W) — 1 where any LRP patch exists
    """
    _empty = np.zeros((H, W), dtype=np.float32)

    area = float(gt_u8.sum()) / 255.0
    if area < HUP_MIN_AREA:
        return _empty.copy(), _empty.copy(), _empty.copy(), _empty.copy()

    segments, cnt = find_hard_segments(gt_u8, K=K)
    if not segments:
        return _empty.copy(), _empty.copy(), _empty.copy(), _empty.copy()

    pts = cnt.reshape(-1, 2)
    N   = len(pts)

    # tight morph kernel sizes (much smaller than global rings)
    radius  = np.sqrt(area / np.pi)
    lrp_ero = max(3, int(radius * HUP_LRP_ERO))
    lrp_dil = max(3, int(radius * HUP_LRP_DIL))
    if lrp_ero % 2 == 0:
        lrp_ero += 1
    if lrp_dil % 2 == 0:
        lrp_dil += 1

    ero_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (lrp_ero, lrp_ero)
    )
    dil_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (lrp_dil, lrp_dil)
    )

    lrp_inner_all = np.zeros((H, W), dtype=np.uint8)
    lrp_outer_all = np.zeros((H, W), dtype=np.uint8)
    lrp_mask_all  = np.zeros((H, W), dtype=np.uint8)

    for (center, start, end) in segments:
        # collect contour point indices for this segment (wrap-around safe)
        if start <= end:
            seg_idx = list(range(start, end + 1))
        else:
            seg_idx = list(range(start, N)) + list(range(0, end + 1))

        seg_pts = pts[seg_idx]   # (n_seg, 2)

        # bounding box with padding
        pad   = lrp_dil * 2
        x_min = max(0, int(seg_pts[:, 0].min()) - pad)
        x_max = min(W, int(seg_pts[:, 0].max()) + pad)
        y_min = max(0, int(seg_pts[:, 1].min()) - pad)
        y_max = min(H, int(seg_pts[:, 1].max()) + pad)

        patch_gt = gt_u8[y_min:y_max, x_min:x_max]
        if patch_gt.size == 0 or patch_gt.sum() == 0:
            continue

        # tight morphological operations on patch
        patch_ero = cv2.erode(patch_gt,  ero_kernel, iterations=1)
        patch_dil = cv2.dilate(patch_gt, dil_kernel, iterations=1)

        # convert to polygon masks at patch resolution
        ph, pw = patch_gt.shape
        pm_ero = to_poly_mask(patch_ero, ph, pw)
        pm_dil = to_poly_mask(patch_dil, ph, pw)

        # paste back (union across patches)
        lrp_inner_all[y_min:y_max, x_min:x_max] = np.maximum(
            lrp_inner_all[y_min:y_max, x_min:x_max], pm_ero
        )
        lrp_outer_all[y_min:y_max, x_min:x_max] = np.maximum(
            lrp_outer_all[y_min:y_max, x_min:x_max], pm_dil
        )
        lrp_mask_all[y_min:y_max, x_min:x_max] = 1

    # --- derive zone maps ---------------------------------------------------
    inner_f = (lrp_inner_all > 127).astype(np.float32)
    outer_f = (lrp_outer_all > 127).astype(np.float32)
    mask_f  = lrp_mask_all.astype(np.float32)

    lrp_fg        = inner_f                              # Omega_RF
    lrp_bg        = mask_f * (1.0 - outer_f)            # Omega_RB
    lrp_uncertain = np.clip(outer_f - inner_f, 0.0, 1.0)
    lrp_mask      = mask_f

    return lrp_fg, lrp_bg, lrp_uncertain, lrp_mask


# ---------------------------------------------------------------------------
# Main annotation generator
# ---------------------------------------------------------------------------

def generate_hupanno(gt_np, K=HUP_K_PATCHES):
    """
    Generate HUPAnno masks from a binary GT mask.

    Args:
        gt_np : (H, W) float32 in [0, 1]
        K     : number of LRP patches

    Returns (all numpy arrays):
        y_in          float32 (H,W) — global P_in  mask  (certain FG)
        y_out         float32 (H,W) — global P_out mask  (outer envelope)
        omega_delta   float32 (H,W) — global uncertain ring
        lrp_fg        float32 (H,W) — Omega_RF
        lrp_bg        float32 (H,W) — Omega_RB
        lrp_uncertain float32 (H,W) — narrow uncertain strip at hard segments
        lrp_mask      float32 (H,W) — 1 where LRP patches exist
        y_c           int64   (H,W) — 4-class CCG label map
    """
    gt_u8 = (gt_np * 255).astype(np.uint8)
    H, W  = gt_np.shape

    # --- global rings (identical to BPAnno) ---------------------------------
    area = float(gt_u8.sum()) / 255.0
    if area > 10:
        radius = np.sqrt(area / np.pi)
        dil_ks = max(7, int(radius * HUP_DIL_SCALE))
        ero_ks = max(5, int(radius * HUP_ERO_SCALE))
    else:
        dil_ks, ero_ks = 9, 7

    if dil_ks % 2 == 0:
        dil_ks += 1
    if ero_ks % 2 == 0:
        ero_ks += 1

    dilated = cv2.dilate(
        gt_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ks, dil_ks))
    )
    eroded  = cv2.erode(
        gt_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero_ks, ero_ks))
    )

    m_out_u8 = to_poly_mask(dilated, H, W)
    m_in_u8  = to_poly_mask(eroded,  H, W)
    if m_in_u8.sum() == 0:
        m_in_u8 = gt_u8.copy()

    # enforce P_in ⊆ P_out
    m_in_u8 = cv2.bitwise_and(m_in_u8, m_out_u8)

    y_in        = (m_in_u8  > 127).astype(np.float32)
    y_out       = (m_out_u8 > 127).astype(np.float32)
    omega_delta = np.clip(y_out - y_in, 0.0, 1.0).astype(np.float32)

    # --- LRP patches --------------------------------------------------------
    lrp_fg, lrp_bg, lrp_uncertain, lrp_mask = generate_lrp_masks(
        gt_u8, H, W, K=K
    )

    # --- 4-class CCG label map ----------------------------------------------
    # Priority (highest overwrites lower):
    #   0 = certain BG  (default / outside P_out / Omega_RB)
    #   1 = global uncertain  (Omega_Delta non-LRP)
    #   2 = LRP uncertain     (inside patch, between tight rings)
    #   3 = certain FG        (inside P_in / Omega_RF)
    y_c = np.zeros((H, W), dtype=np.int64)

    y_c[omega_delta == 1]  = 1   # global uncertain ring
    y_c[lrp_uncertain == 1] = 2  # LRP uncertain strip (overrides 1)
    y_c[lrp_bg == 1]        = 0  # LRP resolved BG    (overrides 2)
    y_c[y_in == 1]          = 3  # global certain FG   (highest priority)
    y_c[lrp_fg == 1]        = 3  # LRP resolved FG     (highest priority)

    return (y_in, y_out, omega_delta,
            lrp_fg, lrp_bg, lrp_uncertain, lrp_mask,
            y_c)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PolypDataset(data.Dataset):

    def __init__(self, image_root, gt_root, trainsize,
                 augmentations, split='train', K=HUP_K_PATCHES):
        self.trainsize     = trainsize
        self.augmentations = augmentations
        self.split         = split
        self.K             = K

        self.images = sorted([
            image_root + f for f in os.listdir(image_root)
            if f.lower().endswith('.jpg') or f.lower().endswith('.png')
        ])
        self.gts = sorted([
            gt_root + f for f in os.listdir(gt_root)
            if f.lower().endswith('.png') or f.lower().endswith('.jpg')
        ])
        self.filter_files()
        self.size = len(self.images)

        if self.augmentations in ('True', True):
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

        # same random seed for image and mask transforms
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)
        random.seed(seed)
        torch.manual_seed(seed)
        gt    = self.gt_transform(gt)

        gt_np = gt.squeeze().cpu().numpy()

        (y_in, y_out, omega_delta,
         lrp_fg, lrp_bg, lrp_uncertain, lrp_mask,
         y_c) = generate_hupanno(gt_np, K=self.K)

        return (
            image,
            torch.from_numpy(y_in).float(),
            torch.from_numpy(y_out).float(),
            torch.from_numpy(omega_delta).float(),
            torch.from_numpy(lrp_fg).float(),
            torch.from_numpy(lrp_bg).float(),
            torch.from_numpy(lrp_uncertain).float(),
            torch.from_numpy(lrp_mask).float(),
            torch.from_numpy(y_c).long(),
            gt
        )

    def filter_files(self):
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img_size = Image.open(img_path).size
            gt_size  = Image.open(gt_path).size
            if img_size == gt_size:
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
               shuffle=False, num_workers=4, pin_memory=True,
               augmentation=False, split='train', K=HUP_K_PATCHES):
    dataset = PolypDataset(
        image_root, gt_root, trainsize,
        augmentation, split, K=K
    )
    return data.DataLoader(
        dataset, batch_size=batchsize,
        shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory
    )


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = sorted([
            image_root + f for f in os.listdir(image_root)
            if f.lower().endswith('.jpg') or f.lower().endswith('.png')
        ])
        self.gts = sorted([
            gt_root + f for f in os.listdir(gt_root)
            if (f.lower().endswith('.tif') or
                f.lower().endswith('.png') or
                f.lower().endswith('.jpg'))
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
        name  = name.replace('.jpg', '.png')
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')