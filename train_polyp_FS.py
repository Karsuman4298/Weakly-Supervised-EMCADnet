import os
import time
import logging
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from lib.networks import EMCADNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter, cal_params_flops


def structure_loss_full(pred, gt):
    eps = 1e-6
    prob = torch.sigmoid(pred)
    # BCE
    bce = F.binary_cross_entropy_with_logits(pred, gt)
    # Dice
    inter = (prob * gt).sum()
    dice = 1 - (2 * inter + eps) / (prob.sum() + gt.sum() + eps)
    # Edge loss
    laplacian_kernel = torch.tensor(
        [[1,1,1],
         [1,-8,1],
         [1,1,1]],
        dtype=torch.float32,
        device=pred.device
    ).view(1,1,3,3)

    pred_edge = F.conv2d(prob, laplacian_kernel, padding=1)
    gt_edge   = F.conv2d(gt, laplacian_kernel, padding=1)
    edge_loss = F.l1_loss(pred_edge, gt_edge)
    return 1.0 * bce + 1.2 * dice + 0.3 * edge_loss


def dice_coefficient(predicted, labels):
    smooth = 1e-6
    predicted_flat = predicted.view(-1)
    labels_flat = labels.view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2. * intersection + smooth) / (total + smooth)

def iou(predicted, labels):
    smooth = 1e-6
    predicted_flat = predicted.view(-1)
    labels_flat = labels.view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


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
        for images, _,gts in test_loader:

            images = images.cuda()
            gts = gts.cuda().float()

            preds = model(images)
            if isinstance(preds, list):
                preds = preds[-1]

            for i in range(len(images)):
                p = torch.sigmoid(preds[i]).squeeze()

                input_binary = (p >= 0.5).float()
                target_binary = (gts[i].squeeze() >= 0.5).float()

                DSC += dice_coefficient(input_binary, target_binary).item()
                IOU += iou(input_binary, target_binary).item()

                total_images += 1

    return DSC / total_images, IOU / total_images, total_images


def train(train_loader, model, optimizer, epoch, opt, model_name):
    model.train()

    global best, test_dice_at_best_val, total_train_time, dict_plot

    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates = [0.75, 1, 1.25]
    total_step = len(train_loader)

    for i, (images, _, gts) in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda().float()
            if rate != 1:
                trainsize = int(round(opt.img_size * rate / 32) * 32)
                images = F.interpolate(
                    images,
                    size=(trainsize, trainsize),
                    mode='bilinear',
                    align_corners=True
                )

            P = model(images)
            if not isinstance(P, list):
                P = [P]

            gts_resized = F.interpolate(
                gts,
                size=P[0].shape[2:],
                mode='nearest'
            )

            loss1 = structure_loss_full(P[0], gts_resized)
            loss2 = structure_loss_full(P[1], gts_resized)
            loss3 = structure_loss_full(P[2], gts_resized)
            loss4 = structure_loss_full(P[3], gts_resized)

            loss_sum = structure_loss_full(
                P[0] + P[1] + P[2] + P[3],
                gts_resized
            )

            loss = loss1 + loss2 + loss3 + loss4 + loss_sum
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            if rate == 1:
                loss_record.update(loss, opt.batchsize)

        if i % 50 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{epoch}/{opt.epoch}], '
                  f'Step [{i}/{total_step}], '
                  f'Loss: {loss_record.show():.4f}')

    total_train_time += (time.time() - epoch_start)

    # Save Last
    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-last.pth"))

    # Validation
    for ds in ['val']:
        d_dice, d_iou, _ = test(model, opt.test_path, ds, opt)
        print(f'Epoch {epoch} | Val Dice: {d_dice:.4f} | IoU: {d_iou:.4f}')

        if d_dice > best:
            best = d_dice
            test_dice_at_best_val = d_dice
            torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-best.pth"))
            print("Best model updated.")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--test_batchsize', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--train_save', type=str, default='./model_pth_full/')

    opt = parser.parse_args()

    best = 0.0
    test_dice_at_best_val = 0.0
    total_train_time = 0

    model = EMCADNet(
        num_classes=1,
        kernel_sizes=[1,3,5],
        expansion_factor=2,
        dw_parallel=True,
        add=True,
        lgag_ks=3,
        activation='relu6',
        encoder='pvt_v2_b2',
        pretrain=True
    )

    model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

    train_loader = get_loader(
        image_root=f'{opt.train_path}/images/',
        gt_root=f'{opt.train_path}/masks/',
        batchsize=opt.batchsize,
        trainsize=opt.img_size,
        shuffle=True,
        augmentation=True
    )

    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 300)
        train(train_loader, model, optimizer, epoch, opt, "full_supervision")
        scheduler.step()

    print("\nTraining Complete")