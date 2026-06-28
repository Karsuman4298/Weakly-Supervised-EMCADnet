import os
import time
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib.networks_fs import EMCADNet
from utils.dataloader_fs import get_loader
from utils.utils import clip_gradient, AvgMeter


# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------

def structure_loss_full(pred, gt):
    """
    pred : logits  (B, 1, H, W)
    gt   : binary  (B, 1, H, W)
    """
    eps = 1e-6
    prob = torch.sigmoid(pred)

    # Weighted BCE (focus on hard pixels)
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt
    )
    wbce = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    wbce = wbce.mean()

    # Weighted IoU
    inter = ((prob * gt) * weit).sum(dim=(2, 3))
    union = ((prob + gt) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + eps) / (union - inter + eps)
    wiou = wiou.mean()

    # Edge loss (Laplacian)
    laplacian_kernel = torch.tensor(
        [[1,  1, 1],
         [1, -8, 1],
         [1,  1, 1]],
        dtype=torch.float32,
        device=pred.device
    ).view(1, 1, 3, 3)

    pred_edge = F.conv2d(prob, laplacian_kernel, padding=1)
    gt_edge   = F.conv2d(gt,   laplacian_kernel, padding=1)
    edge_loss = F.l1_loss(pred_edge, gt_edge)

    return wbce + wiou + 0.3 * edge_loss


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def dice_coefficient(predicted, labels):
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat    = labels.contiguous().view(-1)
    intersection   = (predicted_flat * labels_flat).sum()
    total          = predicted_flat.sum() + labels_flat.sum()
    return (2.0 * intersection + smooth) / (total + smooth)


def iou_score(predicted, labels):
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat    = labels.contiguous().view(-1)
    intersection   = (predicted_flat * labels_flat).sum()
    union          = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def test(model, path, dataset, opt):
    data_path  = os.path.join(path, dataset)
    image_root = os.path.join(data_path, 'images/')
    gt_root    = os.path.join(data_path, 'masks/')

    model.eval()

    test_loader = get_loader(
        image_root=image_root,
        gt_root=gt_root,
        batchsize=opt.test_batchsize,
        trainsize=opt.img_size,
        shuffle=False,
        augmentation=False
    )

    DSC = 0.0
    IOU = 0.0
    total_images = 0

    with torch.no_grad():
        for images, gts in test_loader:
            images = images.cuda()
            gts    = gts.cuda().float()

            # mode='test' returns [finest, ..., coarsest]
            preds = model(images, mode='test')

            # Use finest prediction (index 0 in test mode)
            pred = preds[0]

            for i in range(images.shape[0]):
                p = torch.sigmoid(pred[i]).squeeze()
                g = gts[i].squeeze()

                binary_pred   = (p >= 0.5).float()
                binary_target = (g >= 0.5).float()

                DSC += dice_coefficient(binary_pred, binary_target).item()
                IOU += iou_score(binary_pred, binary_target).item()
                total_images += 1

    n = max(total_images, 1)
    return DSC / n, IOU / n, total_images


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train(train_loader, model, optimizer, epoch, opt, model_name):
    global best, total_train_time

    model.train()

    epoch_start  = time.time()
    loss_record  = AvgMeter()
    size_rates   = [0.75, 1.0, 1.25]
    total_step   = len(train_loader)

    for i, (images, gts) in enumerate(train_loader, start=1):

        images_ori = images.cuda()
        gts_ori    = gts.cuda().float()

        for rate in size_rates:
            optimizer.zero_grad()

            # Multi-scale training
            if rate != 1.0:
                trainsize = int(round(opt.img_size * rate / 32) * 32)
                images_r  = F.interpolate(
                    images_ori,
                    size=(trainsize, trainsize),
                    mode='bilinear',
                    align_corners=True
                )
            else:
                images_r = images_ori

            # Forward — train mode returns [coarse→fine]: [p4, p3, p2, p1]
            preds = model(images_r, mode='train')

            # Resize GT to match each prediction (all same size here,
            # but kept general in case decoder outputs vary)
            gts_r = F.interpolate(
                gts_ori,
                size=preds[0].shape[2:],
                mode='nearest'
            )

            # Deep supervision: loss on every decoder output
            # preds[0]=coarsest(p4), preds[3]=finest(p1)
            loss4 = structure_loss_full(preds[0], gts_r)   # p4 coarsest
            loss3 = structure_loss_full(preds[1], gts_r)   # p3
            loss2 = structure_loss_full(preds[2], gts_r)   # p2
            loss1 = structure_loss_full(preds[3], gts_r)   # p1 finest

            # Ensemble (summed logits) loss
            loss_sum = structure_loss_full(
                preds[0] + preds[1] + preds[2] + preds[3],
                gts_r
            )

            loss = loss1 + loss2 + loss3 + loss4 + loss_sum

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1.0:
                loss_record.update(loss.item(), opt.batchsize)

        # Logging
        if i % 50 == 0 or i == total_step:
            print(
                f'{datetime.now()}  '
                f'Epoch [{epoch:03d}/{opt.epoch}]  '
                f'Step [{i:04d}/{total_step}]  '
                f'Loss: {loss_record.show():.4f}'
            )

    total_train_time += time.time() - epoch_start

    # ---- Save last checkpoint ----
    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(save_path, f'{model_name}-last.pth')
    )

    # ---- Validate ----
    val_dice, val_iou, n_val = test(model, opt.test_path, 'val', opt)
    print(
        f'Epoch {epoch:03d} | Val Dice: {val_dice:.4f} | '
        f'Val IoU: {val_iou:.4f} | Images: {n_val}'
    )

    # ---- Save best checkpoint ----
    if val_dice > best:
        best = val_dice
        torch.save(
            model.state_dict(),
            os.path.join(save_path, f'{model_name}-best.pth')
        )
        print(f'  >> Best model updated  (Dice={best:.4f})')


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',          type=int,   default=200)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--batchsize',      type=int,   default=8)
    parser.add_argument('--test_batchsize', type=int,   default=8)
    parser.add_argument('--img_size',       type=int,   default=352)
    parser.add_argument('--clip',           type=float, default=0.5)
    parser.add_argument('--train_path',     type=str,   required=True)
    parser.add_argument('--test_path',      type=str,   required=True)
    parser.add_argument('--train_save',     type=str,   default='./model_pth_full/')
    opt = parser.parse_args()

    # Global trackers
    best             = 0.0
    total_train_time = 0.0

    # ---- Model ----
    model = EMCADNet(
        num_classes=1,
        kernel_sizes=[1, 3, 5],
        expansion_factor=2,
        dw_parallel=True,
        add=True,
        lgag_ks=3,
        activation='relu6',
        encoder='pvt_v2_b2',
        pretrain=True
    ).cuda()

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=1e-4
    )

    # ---- Scheduler ----
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=opt.epoch,
        eta_min=1e-6
    )

    # ---- Data ----
    train_loader = get_loader(
        image_root=f'{opt.train_path}/images/',
        gt_root=f'{opt.train_path}/masks/',
        batchsize=opt.batchsize,
        trainsize=opt.img_size,
        shuffle=True,
        augmentation=True
    )

    print("Starting full-supervised training...")
    print(f"  Epochs      : {opt.epoch}")
    print(f"  Batch size  : {opt.batchsize}")
    print(f"  Image size  : {opt.img_size}")
    print(f"  LR          : {opt.lr}")

    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch, opt, 'full_supervision')
        scheduler.step()

    hrs  = total_train_time // 3600
    mins = (total_train_time % 3600) // 60
    print(f"\nTraining Complete in {int(hrs)}h {int(mins)}m")
    print(f"Best Val Dice : {best:.4f}")