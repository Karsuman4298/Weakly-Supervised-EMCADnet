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


def structure_loss_full(pred, gt):
    """
    pred: logits, shape (B,1,H,W)
    gt:   binary mask, shape (B,1,H,W)
    """
    eps = 1e-6
    prob = torch.sigmoid(pred)

    # BCE
    bce = F.binary_cross_entropy_with_logits(pred, gt)

    # Dice, batch-wise stable
    inter = (prob * gt).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + eps) / (union + eps)
    dice = dice.mean()

    # Edge loss
    laplacian_kernel = torch.tensor(
        [[1, 1, 1],
         [1, -8, 1],
         [1, 1, 1]],
        dtype=torch.float32,
        device=pred.device
    ).view(1, 1, 3, 3)

    pred_edge = F.conv2d(prob, laplacian_kernel, padding=1)
    gt_edge = F.conv2d(gt, laplacian_kernel, padding=1)
    edge_loss = F.l1_loss(pred_edge, gt_edge)

    return 1.0 * bce + 1.2 * dice + 0.3 * edge_loss


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def dice_coefficient(predicted, labels):
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2.0 * intersection + smooth) / (total + smooth)


def iou(predicted, labels):
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def test(model, path, dataset, opt):
    data_path = os.path.join(path, dataset)
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'

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
            gts = gts.cuda().float()

            # networks_hup default mode is test, but explicit is cleaner
            preds = model(images, mode='test')
            if isinstance(preds, list):
                preds = preds[-1]

            for i in range(images.shape[0]):
                p = torch.sigmoid(preds[i]).squeeze()
                g = gts[i].squeeze()

                input_binary = (p >= 0.5).float()
                target_binary = (g >= 0.5).float()

                DSC += dice_coefficient(input_binary, target_binary).item()
                IOU += iou(input_binary, target_binary).item()
                total_images += 1

    return DSC / max(total_images, 1), IOU / max(total_images, 1), total_images


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train(train_loader, model, optimizer, epoch, opt, model_name):
    global best, test_dice_at_best_val, total_train_time

    model.train()

    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates = [0.75, 1.0, 1.25]
    total_step = len(train_loader)

    for i, (images, gts) in enumerate(train_loader, start=1):

        images_ori = images.cuda()
        gts_ori = gts.cuda().float()

        for rate in size_rates:
            optimizer.zero_grad()

            if rate != 1.0:
                trainsize = int(round(opt.img_size * rate / 32) * 32)
                images_r = F.interpolate(
                    images_ori,
                    size=(trainsize, trainsize),
                    mode='bilinear',
                    align_corners=True
                )
            else:
                images_r = images_ori

            preds = model(images_r, mode='test')
            if not isinstance(preds, list):
                preds = [preds]

            gts_r = F.interpolate(
                gts_ori,
                size=preds[0].shape[2:],
                mode='nearest'
            )

            loss1 = structure_loss_full(preds[0], gts_r)
            loss2 = structure_loss_full(preds[1], gts_r)
            loss3 = structure_loss_full(preds[2], gts_r)
            loss4 = structure_loss_full(preds[3], gts_r)

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

        if i % 50 == 0 or i == total_step:
            print(
                f'{datetime.now()} Epoch [{epoch}/{opt.epoch}], '
                f'Step [{i}/{total_step}], '
                f'Loss: {loss_record.show():.4f}'
            )

    total_train_time += time.time() - epoch_start

    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(save_path, f"{model_name}-last.pth")
    )

    d_dice, d_iou, _ = test(model, opt.test_path, 'val', opt)

    print(f'Epoch {epoch} | Val Dice: {d_dice:.4f} | IoU: {d_iou:.4f}')

    if d_dice > best:
        best = d_dice
        test_dice_at_best_val = d_dice

        torch.save(
            model.state_dict(),
            os.path.join(save_path, f"{model_name}-best.pth")
        )

        print("Best model updated.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--test_batchsize', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--train_save', type=str, default='./model_pth_full/')

    opt = parser.parse_args()

    best = 0.0
    test_dice_at_best_val = 0.0
    total_train_time = 0.0

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=1e-4
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=opt.epoch,
        eta_min=1e-6
    )

    train_loader = get_loader(
        image_root=f'{opt.train_path}/images/',
        gt_root=f'{opt.train_path}/masks/',
        batchsize=opt.batchsize,
        trainsize=opt.img_size,
        shuffle=True,
        augmentation=True
    )

    print("Starting full-supervised training...")

    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch, opt, "full_supervision")
        scheduler.step()

    print("\nTraining Complete")
    print(f"Best Val Dice: {best:.4f}")