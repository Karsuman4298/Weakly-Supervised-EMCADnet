import os
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from lib.networks import EMCADNet
from utils.dataloader import get_loader as get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter, cal_params_flops


# ═══════════════════════════════════════════════════════════════════
#  BPAnno Loss Functions
# ═══════════════════════════════════════════════════════════════════

def dice_loss_masked(pred_prob, target, valid_mask=None, smooth=1e-6):
    if valid_mask is not None:
        pred_prob = pred_prob * valid_mask
        target    = target    * valid_mask
    p = pred_prob.reshape(pred_prob.shape[0], -1)
    t = target.reshape(target.shape[0], -1).float()
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    return (1 - (2 * inter + smooth) / (denom + smooth)).mean()


def dual_mask_loss(seg_logit, y_in, y_en, use_edge=True):
    pred = torch.sigmoid(seg_logit)
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    L_in = dice_loss_masked(pred, y_in)
    L_en = dice_loss_masked(pred, y_en)
    Lc   = L_in + L_en
    if use_edge:
        laplacian = torch.tensor(
            [[1,  1, 1],
             [1, -8, 1],
             [1,  1, 1]],
            dtype=torch.float32,
            device=pred.device
        ).view(1, 1, 3, 3)
        pred_4d   = pred.unsqueeze(1)
        y_en_4d   = y_en.unsqueeze(1).float()
        pred_edge = F.conv2d(pred_4d,  laplacian, padding=1).squeeze(1)
        gt_edge   = F.conv2d(y_en_4d, laplacian, padding=1).squeeze(1)
        edge_loss = F.l1_loss(pred_edge * y_en, gt_edge * y_en)
        Lc = Lc + 0.3 * edge_loss
    return Lc


def classification_loss_ccg(cls_logits, y_c):
    B, C, H, W = cls_logits.shape
    logits_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = y_c.reshape(-1)
    return F.cross_entropy(logits_flat, labels_flat)


@torch.no_grad()
def generate_pseudo_labels(seg_logit, cls_logits, omega_delta, entropy_thresh=0.5):
    seg_prob = torch.sigmoid(seg_logit)
    if seg_prob.dim() == 4:
        seg_prob = seg_prob.squeeze(1)
    cls_prob = F.softmax(cls_logits, dim=1)
    cls_pred = cls_prob.argmax(dim=1)
    p_bg = cls_prob[:, 0]
    p_fg = cls_prob[:, 2]
    U_c  = (p_fg >= p_bg).long()
    U_c[cls_pred == 0] = 0
    U_c[cls_pred == 2] = 1
    seg_fg_in_ring = (seg_prob >= 0.5) & (omega_delta == 1)
    U_c[seg_fg_in_ring] = 1
    eps = 1e-6
    entropy = -(
        seg_prob       * torch.log(seg_prob       + eps) +
        (1 - seg_prob) * torch.log(1 - seg_prob   + eps)
    )
    U_e = torch.zeros_like(U_c)
    U_e[entropy >= entropy_thresh] = -1
    U = torch.clamp(U_c + 2 * U_e, min=-1)
    U = U * omega_delta.long()
    U[omega_delta == 0] = -1
    return U


def contrastive_loss_ccl(embeddings, seg_logit, pseudo_labels,
                          omega_delta, y_in, y_en, neg_queue,
                          temperature=0.1, hard_ratio=0.7,
                          num_anchors=100, pixel_pool=512):
    """
    LPCL: pixel-wise contrastive loss with hard-sample mining.
    neg_queue must be passed as .detach().clone() by the caller.
    """
    B, D, H, W = embeddings.shape
    device = embeddings.device

    # Always detach queue inside function as safety guarantee
    neg_queue = neg_queue.detach()

    seg_pred = (torch.sigmoid(seg_logit.detach()).squeeze(1) >= 0.5).long()

    y_hat = seg_pred.clone()
    for b in range(B):
        valid = (omega_delta[b] == 1) & (pseudo_labels[b] != -1)
        y_hat[b][valid] = pseudo_labels[b][valid]

    certain_fg   = (y_in  == 1)
    certain_bg   = (y_en  == 0) & (omega_delta == 0)
    certain_gt   = torch.zeros_like(seg_pred)
    certain_gt[certain_fg] = 1
    certain_gt[certain_bg] = 0
    certain_mask = (omega_delta == 0)

    all_anchor_embs = []
    all_anchor_lbls = []
    all_pixel_embs  = []
    all_pixel_lbls  = []

    for b in range(B):
        hard = (
            (certain_mask[b] & (seg_pred[b] != certain_gt[b])) |
            ((omega_delta[b] == 1) & (pseudo_labels[b] != -1))
        )
        easy = certain_mask[b] & (seg_pred[b] == certain_gt[b])

        h_idx = hard.nonzero(as_tuple=False)
        e_idx = easy.nonzero(as_tuple=False)

        if len(h_idx) == 0 or len(e_idx) == 0:
            continue

        n_h = min(int(num_anchors * hard_ratio), len(h_idx))
        n_e = min(num_anchors - n_h,             len(e_idx))

        h_sel = h_idx[torch.randperm(len(h_idx), device=device)[:n_h]]
        e_sel = e_idx[torch.randperm(len(e_idx), device=device)[:n_e]]
        anc   = torch.cat([h_sel, e_sel], dim=0)

        ah, aw = anc[:, 0], anc[:, 1]
        all_anchor_embs.append(embeddings[b, :, ah, aw].T)
        all_anchor_lbls.append(y_hat[b, ah, aw])

        flat_emb = embeddings[b].reshape(D, -1).T
        flat_lbl = y_hat[b].reshape(-1)
        pool_n   = min(pixel_pool, H * W)
        pool_idx = torch.randperm(H * W, device=device)[:pool_n]
        all_pixel_embs.append(flat_emb[pool_idx])
        all_pixel_lbls.append(flat_lbl[pool_idx])

    if len(all_anchor_embs) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    anc_emb  = torch.cat(all_anchor_embs, dim=0)
    anc_lbl  = torch.cat(all_anchor_lbls, dim=0)
    pix_emb  = torch.cat(all_pixel_embs,  dim=0)
    pix_lbl  = torch.cat(all_pixel_lbls,  dim=0)

    sim_pos_pool  = torch.mm(anc_emb, pix_emb.T) / temperature
    sim_neg_queue = torch.mm(anc_emb, neg_queue)  / temperature
    neg_denom     = torch.exp(sim_neg_queue).sum(dim=1)

    total_loss  = torch.tensor(0.0, device=device)
    valid_count = 0

    for i in range(len(anc_emb)):
        li = anc_lbl[i].item()
        if li == -1:
            continue
        pos_mask = (pix_lbl == li)
        if pos_mask.sum() == 0:
            continue
        pos_sim = torch.exp(sim_pos_pool[i][pos_mask]).mean()
        denom   = pos_sim + neg_denom[i]
        loss_i  = -torch.log(pos_sim / (denom + 1e-6))
        total_loss  = total_loss + loss_i
        valid_count += 1

    return total_loss / max(valid_count, 1)


# ═══════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════

def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat    = labels.contiguous().view(-1)
    intersection   = (predicted_flat * labels_flat).sum()
    total          = predicted_flat.sum() + labels_flat.sum()
    return (2. * intersection + smooth) / (total + smooth)


def iou(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat    = labels.contiguous().view(-1)
    intersection   = (predicted_flat * labels_flat).sum()
    union          = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ═══════════════════════════════════════════════════════════════════
#  Test function
# ═══════════════════════════════════════════════════════════════════

def test(model, path, dataset, opt):
    data_path  = os.path.join(path, dataset)
    image_root = f'{data_path}/images/'
    gt_root    = f'{data_path}/masks/'
    model.eval()

    test_loader = get_loader(
        image_root=image_root,
        gt_root=gt_root,
        batchsize=opt.test_batchsize,
        trainsize=opt.img_size,
        shuffle=False,
        augmentation=False
    )

    DSC          = 0.0
    IOU          = 0.0
    total_images = 0

    with torch.no_grad():
        for pack in test_loader:
            # pack = (images, y_in, y_en, omega_delta, y_c, gts)
            images = pack[0].cuda()
            gts    = pack[5].cuda().float()

            ress = model(images, mode='test')
            if not isinstance(ress, list):
                ress = [ress]
            predictions = ress[-1]

            for idx in range(len(images)):
                p             = predictions[idx].unsqueeze(0)
                pred_resized  = torch.sigmoid(p).squeeze()
                gt_resized    = gts[idx].squeeze()
                input_binary  = (pred_resized >= 0.5).float()
                target_binary = (gt_resized   >= 0.5).float()
                DSC          += dice_coefficient(input_binary, target_binary).item()
                IOU          += iou(input_binary, target_binary).item()
                total_images += 1

    return DSC / total_images, IOU / total_images, total_images


# ═══════════════════════════════════════════════════════════════════
#  Train function
# ═══════════════════════════════════════════════════════════════════

def train(train_loader, model, optimizer, epoch, opt, model_name):
    model.train()
    global best, test_dice_at_best_val, total_train_time, dict_plot

    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates  = [0.75, 1, 1.25]
    total_step  = len(train_loader)

    # BPAnno staged training: CCG+CCL activate after 30% of epochs
    use_ccg_ccl = (epoch > int(0.3 * opt.epoch))

    for i, (images, y_in, y_en, omega_delta, y_c, gts) in enumerate(train_loader, start=1):

        for rate in size_rates:
            optimizer.zero_grad()

            # ── Move to GPU ───────────────────────────────────────────
            images      = Variable(images).cuda()
            y_in        = y_in.float().cuda()
            y_en        = y_en.float().cuda()
            omega_delta = omega_delta.float().cuda()
            y_c         = y_c.long().cuda()
            gts         = Variable(gts).float().cuda()

            # ── Multi-scale resize ────────────────────────────────────
            if rate != 1:
                trainsize = int(round(opt.img_size * rate / 32) * 32)
                images    = F.interpolate(
                    images,
                    size=(trainsize, trainsize),
                    mode='bilinear',
                    align_corners=True
                )

            # ── Forward pass ──────────────────────────────────────────
            P, cls_logits, embeddings = model(images, mode='train')

            if not isinstance(P, list):
                P = [P]

            # ── Helper: resize BPAnno masks ───────────────────────────
            def resize_mask(m, size):
                return F.interpolate(
                    m.unsqueeze(1).float(), size=size, mode='nearest'
                ).squeeze(1)

            target_size = P[0].shape[2:]
            y_in_r      = resize_mask(y_in,  target_size)
            y_en_r      = resize_mask(y_en,  target_size)

            # ── Lc: dual-mask Dice on all 4 heads + ensemble ──────────
            seg_ensemble = P[0] + P[1] + P[2] + P[3]
            loss = (
                dual_mask_loss(P[0],         y_in_r, y_en_r) +
                dual_mask_loss(P[1],         y_in_r, y_en_r) +
                dual_mask_loss(P[2],         y_in_r, y_en_r) +
                dual_mask_loss(P[3],         y_in_r, y_en_r) +
                dual_mask_loss(seg_ensemble, y_in_r, y_en_r)
            )

            # ── CCG + CCL (after warmup) ──────────────────────────────
            # Store embeddings snapshot BEFORE backward for queue update
            emb_for_queue = None

            if use_ccg_ccl:
                H_img = images.shape[2]
                W_img = images.shape[3]

                y_in_full = resize_mask(y_in,       (H_img, W_img))
                y_en_full = resize_mask(y_en,       (H_img, W_img))
                od_full   = resize_mask(omega_delta, (H_img, W_img))
                y_c_full  = F.interpolate(
                    y_c.unsqueeze(1).float(),
                    size=(H_img, W_img),
                    mode='nearest'
                ).squeeze(1).long()

                # Lce: classification loss
                Lce = classification_loss_ccg(cls_logits, y_c_full)

                # Pseudo labels
                pseudo = generate_pseudo_labels(
                    P[-1].detach(), cls_logits.detach(), od_full
                )

                # LPCL: contrastive loss
                # Pass detached clone of queue — avoids inplace grad conflict
                LPCL = contrastive_loss_ccl(
                    embeddings,
                    P[-1].detach(),
                    pseudo,
                    od_full,
                    y_in_full,
                    y_en_full,
                    model.neg_queue.detach().clone(),  # KEY FIX
                    temperature=0.1,
                    hard_ratio=0.7,
                    num_anchors=100,
                    pixel_pool=512
                )

                loss = loss + opt.lambda1 * LPCL + opt.lambda2 * Lce

                # Snapshot embeddings for queue update AFTER backward
                # Must be detached so it does not hold graph references
                emb_for_queue = embeddings.detach().permute(0, 2, 3, 1).reshape(
                    -1, model.embed_dim
                )
                
            loss.backward()

            # ── Update memory queue AFTER backward ───────────────────
            # This is the correct order — queue update must NOT happen
            # before backward() because it modifies neg_queue inplace
            if emb_for_queue is not None:
                with torch.no_grad():
                    idx = torch.randperm(
                        emb_for_queue.shape[0],
                        device=emb_for_queue.device
                    )[:64]
                    model.update_queue(emb_for_queue[idx])

            # ── Optimizer step ────────────────────────────────────────
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 100 == 0 or i == total_step:
            phase = "CCG+CCL" if use_ccg_ccl else "warmup-Lc"
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], '
                  f'Step [{i:04d}/{total_step:04d}], '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
                  f'Loss: {loss_record.show():.4f}  [{phase}]')

    total_train_time += (time.time() - epoch_start)

    # Save last checkpoint
    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(save_path, f"{model_name}-last.pth"))

    # Validation + test evaluation
    epoch_results = {}
    for ds in ['test', 'val']:
        d_dice, d_iou, _ = test(model, opt.test_path, ds, opt)
        epoch_results[ds] = d_dice
        logging.info(f'Epoch: {epoch}, Dataset: {ds}, '
                     f'Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
        print(f'Epoch: {epoch}, Dataset: {ds}, '
              f'Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
        dict_plot[ds].append(d_dice)

    # Save best checkpoint
    if epoch_results['val'] > best:
        logging.info(
            f"### Best Model Saved "
            f"(Dice {best:.4f} -> {epoch_results['val']:.4f}) ###"
        )
        print(
            f"### Best Model Saved "
            f"(Dice {best:.4f} -> {epoch_results['val']:.4f}) ###"
        )
        best                  = epoch_results['val']
        test_dice_at_best_val = epoch_results['test']
        torch.save(model.state_dict(),
                   os.path.join(save_path, f"{model_name}-best.pth"))


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    dataset_name = 'ClinicDB'

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='pvt_v2_b2')
    parser.add_argument('--expansion_factor', type=int, default=2)
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--lgag_ks', type=int, default=3)
    parser.add_argument('--activation_mscb', type=str, default='relu6')
    parser.add_argument('--no_dw_parallel', action='store_true', default=False)
    parser.add_argument('--concatenation', action='store_true', default=False)
    parser.add_argument('--no_pretrain', action='store_true', default=False)
    parser.add_argument('--pretrained_dir', type=str, default='./pretrained_pth/pvt/')
    parser.add_argument('--supervision', type=str, default='mutation')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--lambda1', type=float, default=0.3,
                        help='weight for contrastive loss LPCL')
    parser.add_argument('--lambda2', type=float, default=0.5,
                        help='weight for classification loss Lce')
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--test_batchsize', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=300)
    parser.add_argument('--color_image', default=True)
    parser.add_argument('--augmentation', default=True)
    parser.add_argument('--train_path', type=str,
                        default=f'./data/polyp/target/{dataset_name}/train/')
    parser.add_argument('--test_path', type=str,
                        default=f'./data/polyp/target/{dataset_name}/')
    parser.add_argument('--train_save', type=str, default='')
    parser.add_argument('--resume', type=str, default='',
                        help='path to checkpoint .pth to resume from')

    opt = parser.parse_args()

    for run in [1, 2, 3, 4, 5]:
        dict_plot             = {'val': [], 'test': []}
        best                  = 0.0
        test_dice_at_best_val = 0.0
        total_train_time      = 0

        aggregation = 'concat' if opt.concatenation  else 'add'
        dw_mode     = 'series' if opt.no_dw_parallel else 'parallel'

        timestamp = time.strftime('%H%M%S')
        run_id = (
            f"{dataset_name}_{opt.encoder}_EMCAD"
            f"_kernel_sizes_{opt.kernel_sizes}"
            f"_dw_{dw_mode}_{aggregation}"
            f"_lgag_ks_{opt.lgag_ks}"
            f"_ef{opt.expansion_factor}"
            f"_act_mscb_{opt.activation_mscb}"
            f"_bs{opt.batchsize}"
            f"_lr{opt.lr}"
            f"_e{opt.epoch}"
            f"_aug{opt.augmentation}"
            f"_run{run}_t{timestamp}"
        )
        run_id = run_id.replace('[', '').replace(']', '').replace(', ', '_')
        opt.train_save = f'./model_pth/{run_id}/'

        os.makedirs('logs', exist_ok=True)
        os.makedirs(opt.train_save, exist_ok=True)

        logging.basicConfig(
            filename=f'logs/train_log_{run_id}.log',
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            force=True
        )

        # Build model
        model = EMCADNet(
            num_classes      = 1,
            kernel_sizes     = opt.kernel_sizes,
            expansion_factor = opt.expansion_factor,
            dw_parallel      = not opt.no_dw_parallel,
            add              = not opt.concatenation,
            lgag_ks          = opt.lgag_ks,
            activation       = opt.activation_mscb,
            encoder          = opt.encoder,
            pretrain         = not opt.no_pretrain,
            pretrained_dir   = opt.pretrained_dir
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Resume from checkpoint if specified
        if opt.resume and os.path.isfile(opt.resume):
            print(f"Resuming from: {opt.resume}")
            ckpt = torch.load(opt.resume, map_location=device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            else:
                model.load_state_dict(ckpt)
            print("Checkpoint loaded successfully")
        elif opt.resume:
            print(f"WARNING: checkpoint not found at {opt.resume}")

        print(f"Encoder: {opt.encoder} | Decoder: EMCAD")
        cal_params_flops(model, opt.img_size, logging)

        optimizer = torch.optim.AdamW(
            model.parameters(), opt.lr, weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=opt.epoch, eta_min=1e-6
        )

        train_loader = get_loader(
            image_root   = f'{opt.train_path}/images/',
            gt_root      = f'{opt.train_path}/masks/',
            batchsize    = opt.batchsize,
            trainsize    = opt.img_size,
            shuffle      = True,
            augmentation = opt.augmentation,
            split        = 'train',
            color_image  = opt.color_image
        )

        for epoch in range(1, opt.epoch + 1):
            adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            train(train_loader, model, optimizer, epoch, opt, run_id)
            scheduler.step()

        summary = (
            f"\n{'='*40}\n"
            f"FINAL RESULTS: {run_id}\n"
            f"Best Val Dice:          {best:.4f}\n"
            f"Test Dice at Best Val:  {test_dice_at_best_val:.4f}\n"
            f"Total Train Time:       {total_train_time:.2f}s\n"
            f"{'='*40}"
        )
        print(summary)
        logging.info(summary)