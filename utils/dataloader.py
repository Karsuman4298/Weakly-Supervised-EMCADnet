from operator import index
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2


# BPAnno Configuration 
# dilation_scale = 0.40  → Ring mean ~9.7% 
# erosion_scale  = 0.22  → Zero collapses  
BPANNO_DIL_SCALE    = 0.40
BPANNO_ERO_SCALE    = 0.22
BPANNO_DP_RATIO     = 0.02   # Douglas-Peucker: 2% of perimeter
BPANNO_MIN_VERTICES = 5
BPANNO_MAX_VERTICES = 15



def generate_bpanno(gt_np, dp_epsilon_ratio=BPANNO_DP_RATIO):
    """
    Generate BPAnno polygon masks from binary GT mask.
    Creates polygon shapes with straight edges (5-15 vertices).
    Uses SEPARATE dilation and erosion scales to compensate for
    polygon corner-cutting which would otherwise shrink the ring.

    Args:
        gt_np : (H,W) float [0,1] binary ground truth mask
        dp_epsilon_ratio: Douglas-Peucker epsilon as fraction of perimeter 0.02 = real polygon with ~6-15 straight-edge vertices

    Returns:
        y_in : (H,W) float32 - inscribed polygon mask(certain fg)
        y_en : (H,W) float32 - envelope polygon mask(boundary enclosure)
        omega_delta : (H,W) float32 - uncertain ring(y_en - y_in)
        y_c : (H,W) int64   - 3-class CCG labels
                                      0 = certain background
                                      1 = uncertain ring
                                      2 = certain foreground
    """
    gt_u8 = (gt_np * 255).astype(np.uint8)
    H, W  = gt_np.shape

    # Adaptive kernel sizes 
    area = float(gt_u8.sum()) / 255.0

    if area > 10:
        radius = np.sqrt(area / np.pi)
        dil_ks = max(7, int(radius * BPANNO_DIL_SCALE))
        ero_ks = max(5, int(radius * BPANNO_ERO_SCALE))
    else:
        # Tiny lesion fallback
        dil_ks = 9
        ero_ks = 7

    # Kernel must be odd
    if dil_ks % 2 == 0: dil_ks += 1
    if ero_ks % 2 == 0: ero_ks += 1

    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ks, dil_ks))
    ero_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero_ks, ero_ks))

    dilated = cv2.dilate(gt_u8, dil_kernel, iterations=1)   # envelope base
    eroded  = cv2.erode(gt_u8,  ero_kernel, iterations=1)   # inscribed base

    # Convert contour to polygon mask 
    def to_poly_mask(binary_u8):
        """
        Finds the largest contour in binary_u8, applies Douglas-Peucker
        simplification to get a real polygon, then fills it.
        """
        contours, _ = cv2.findContours(
            binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        out = np.zeros((H, W), dtype=np.uint8)

        if not contours:
            return out

        # Take largest contour only
        contour   = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, closed=True)

        if perimeter < 10:
            return out

        # Start epsilon as ratio of perimeter
        epsilon = perimeter * dp_epsilon_ratio
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)

        # Too many vertices → increase epsilon(more aggressive simplification)
        attempts = 0
        while len(polygon) > BPANNO_MAX_VERTICES and attempts < 12:
            epsilon *= 1.25
            polygon  = cv2.approxPolyDP(contour, epsilon, closed=True)
            attempts += 1

        # Too few vertices → decrease epsilon (less aggressive)
        attempts = 0
        while len(polygon) < BPANNO_MIN_VERTICES and attempts < 12:
            epsilon *= 0.75
            polygon  = cv2.approxPolyDP(contour, epsilon, closed=True)
            attempts += 1

        # Need at least 3 points for a valid polygon
        if len(polygon) >= 3:
            cv2.fillPoly(out, [polygon], 255)
        return out

    # Build polygon masks
    y_en_u8 = to_poly_mask(dilated)
    y_in_u8 = to_poly_mask(eroded)

    # if erosion collapsed to empty use GT as inscribed
    if y_in_u8.sum() == 0:
        y_in_u8 = gt_u8.copy()

    y_en = (y_en_u8 > 127).astype(np.float32)
    y_in = (y_in_u8 > 127).astype(np.float32)

    #  inscribed must be strictly inside envelope
    # (polygon straight edges can occasionally push inscribed outside)
    y_in = y_in * y_en

    # Uncertain ring: inside envelope but outside inscribed
    omega_delta = np.clip(y_en - y_in, 0, 1).astype(np.float32)

    # 3-class CCG label map
    y_c = np.zeros_like(y_in, dtype=np.int64)
    y_c[y_en == 1] = 1    # inside envelope → uncertain (overwritten below)
    y_c[y_in == 1] = 2    # inside inscribed → certain fg (overrides ring)
    # pixels outside y_en stay 0 → certain bg

    return y_in, y_en, omega_delta, y_c


class PolypDataset(data.Dataset):

    def __init__(self, image_root, gt_root, trainsize,
                 augmentations, split='train', color_image=True):
        self.trainsize    = trainsize
        self.augmentations= augmentations
        self.split        = split
        self.color_image  = color_image

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
        gt    = self.binary_loader(self.gts[index])

        # Synchronized augmentation for image and GT
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image) # (3, H, W)
        random.seed(seed)
        torch.manual_seed(seed)
        gt = self.gt_transform(gt)  # (1, H, W) float [0,1]
        #BPAnno generation(from already-transformed GT)
        # gt is already resized and augmented, so BPAnno masks
        # are always spatially aligned with the image
        gt_np = gt.squeeze().cpu().numpy()  # (H, W) float [0,1]
        y_in, y_en, omega_delta, y_c = generate_bpanno(gt_np)
        y_in_t  = torch.from_numpy(y_in).float()   # (H, W)
        y_en_t = torch.from_numpy(y_en).float()    # (H, W)
        omega_delta_t = torch.from_numpy(omega_delta).float()  # (H, W)
        y_c_t = torch.from_numpy(y_c).long()   # (H, W)
        return image, y_in_t, y_en_t, omega_delta_t, y_c_t, gt

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

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return (img.resize((w, h), Image.BILINEAR),
                    gt.resize((w, h), Image.NEAREST))
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
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')