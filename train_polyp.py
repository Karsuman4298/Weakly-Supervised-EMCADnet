# train_polyp.py  —  WS-EMCADNet training with HUPAnno + 3-phase curriculum
#
# Faithful to annotation_ideas_detailed.md:
#
#   Phase 1  (0  → 30%):  L = L_c
#   Phase 2  (30 → 70%):  L = L_c + λ1·L_PCL(easy) + λ2·L_ce
#   Phase 3  (70 → 100%): L = L_c + λ1·L_PCL(all)  + λ2·L_ce + λ3·L_patch
#
# Key correctness fixes vs first draft:
#   - loss_certain : all four Dice terms equal weight (no 3× on LRP)
#   - proto_fg/bg  : computed from Ω_I / Ω_O, NOT from lrp_fg / lrp_bg
#   - MU_HARD=0.3  : LOWER (stricter) inside LRP  — correct direction
#   - MU_EASY=0.5  : HIGHER (tolerant) outside    — correct direction
#   - Phase 3 loss : single combined expression, no double-lambda on PCL
#   - loss_patch B : defined explicitly, no walrus-operator scoping issue
#   - l_conf       : included in Phase 2 AND Phase 3 (supporting auxiliary)
#
# Run:
#   python train_polyp.py \
#     --train_path /path/to/dataset/train \
#     --test_path  /path/to/dataset \
#     --epoch 200  --batchsize 8  --K 2

import os
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib.networks import EMCADNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter


# =========================================================================== #
# Hyperparameters
# =========================================================================== #

LAMBDA_PCL   = 0.1    # λ1  — contrastive loss weight
LAMBDA_CE    = 0.3    # λ2  — CCG cross-entropy weight
LAMBDA_PATCH = 0.2    # λ3  — boundary consistency (patch) loss weight
LAMBDA_CONF  = 0.05   # auxiliary weight for confidence-head supervision

# Spatially-varying confidence threshold (README §3)
#   MU_HARD (0.3) inside LRP patches  → LOWER / stricter
#   MU_EASY (0.5) outside             → HIGHER / more tolerant
MU_HARD = 0.3
MU_EASY = 0.5

PCL_TEMP = 0.07   # InfoNCE temperature

# Curriculum phase boundaries (fraction of total epochs)
PHASE2_START = 0.30
PHASE3_START = 0.70

EMA_DECAY = 0.99   # prototype EMA decay rate

# global state
best             = 0.0
total_train_time = 0.0


# =========================================================================== #
# Loss 1 — L_c : patch-aware certain loss  (README §1)
#
#   L_c = L_in(p, y_in) + L_out(p, y_out) + L_RF(p, y_RF) + L_RB(p, y_RB)
#
#   All four terms have EQUAL weight = 1.0.
#   LRP-resolved regions carry the same confidence as Ω_I / Ω_O.
# =========================================================================== #

def soft_dice(pred_logit, target, mask=None, eps=1e-6):
    """
    Soft Dice loss between sigmoid(pred_logit) and target,
    computed only over pixels where mask is True (if provided).
    Returns 0 (no gradient) if no pixels are selected.
    """
    prob = torch.sigmoid(pred_logit)
    if mask is not None:
        if not mask.any():
            return torch.tensor(0.0, device=pred_logit.device,
                                requires_grad=True)
        prob   = prob[mask]
        target = target[mask]
    inter = (prob * target).sum()
    union = prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (union + eps)


def loss_certain(pred, y_in, y_out, lrp_fg, lrp_bg):
    """
    README §1:
      L_c = L_in + L_out + L_RF + L_RB   (equal weights — no upscaling)

    pred   : (B, 1, H, W) logits
    y_in   : (B, H, W) float  — Omega_I  (certain FG)
    y_out  : (B, H, W) float  — P_out envelope (1 inside, 0 outside)
    lrp_fg : (B, H, W) float  — Omega_RF (LRP resolved FG)
    lrp_bg : (B, H, W) float  — Omega_RB (LRP resolved BG)
    """
    p = pred.squeeze(1)   # (B, H, W)

    # Ω_I → predict 1
    l_in  = soft_dice(p, torch.ones_like(p),  mask=y_in.bool())

    # Ω_O (outside P_out) → predict 0
    l_out = soft_dice(p, torch.zeros_like(p), mask=(y_out == 0))

    # Ω_RF (LRP resolved FG) → predict 1
    lrf_mask = lrp_fg.bool()
    l_rf = (soft_dice(p, torch.ones_like(p), mask=lrf_mask)
            if lrf_mask.any() else
            torch.tensor(0.0, device=pred.device))

    # Ω_RB (LRP resolved BG) → predict 0
    lrb_mask = lrp_bg.bool()
    l_rb = (soft_dice(p, torch.zeros_like(p), mask=lrb_mask)
            if lrb_mask.any() else
            torch.tensor(0.0, device=pred.device))

    return l_in + l_out + l_rf + l_rb   # equal weights


# =========================================================================== #
# Loss 2 — L_ce : 4-class CCG cross-entropy  (README §3, λ2·L_ce)
# =========================================================================== #

def loss_ce(cls_logits, y_c):
    """
    4-class CCG classification.
      cls_logits : (B, 4, H, W)
      y_c        : (B, H, W)  int64
                   0 = certain BG  |  1 = global unc  |
                   2 = LRP unc     |  3 = certain FG
    """
    return F.cross_entropy(cls_logits, y_c)


# =========================================================================== #
# Loss 3 — L_conf : confidence-head supervision  (auxiliary, supports §3)
#
#   conf_head is trained to predict:
#     MU_HARD (0.3) inside LRP patches  → stricter entropy threshold
#     MU_EASY (0.5) outside             → tolerant entropy threshold
#
#   Note: direction is LOWER inside LRP (stricter), HIGHER outside (tolerant).
# =========================================================================== #

def loss_conf(conf_map, lrp_mask,
              mu_hard=MU_HARD, mu_easy=MU_EASY):
    """
    conf_map : (B, 1, H, W)  sigmoid output in (0,1)
    lrp_mask : (B, H, W)     float  — 1 inside LRP patches
    """
    target = torch.full_like(conf_map.squeeze(1), mu_easy)
    target[lrp_mask.bool()] = mu_hard
    return F.mse_loss(conf_map.squeeze(1), target)


# =========================================================================== #
# Loss 4 — L_PCL : difficulty-aware pixel contrastive loss  (README §2)
#
#   Phase 2 anchors: easy uncertain pixels   (Ω_Δ non-LRP)
#   Phase 3 anchors: ALL uncertain pixels    (easy + LRP uncertain strip)
#
#   Positives:  Ω_I pixels (certain FG prototype)
#   Negatives:  Ω_O pixels + memory queue
#
#   Both phases use a single call — the caller passes the correct anchor mask.
# =========================================================================== #

def loss_pcl(embeddings, y_in, y_out, anchor_mask, neg_queue, model,
             temp=PCL_TEMP, max_anchors=256, max_pos=128, max_neg=128):
    """
    InfoNCE contrastive loss.

    embeddings  : (B, D, H, W)  L2-normalised
    y_in        : (B, H, W)  float  — Ω_I mask  (positive source)
    y_out       : (B, H, W)  float  — P_out mask (BG = where y_out == 0)
    anchor_mask : (B, H, W)  float  — pixels to use as anchors this phase
    neg_queue   : (D, Q)     float  — memory queue (detached)
    """
    B, D, H, W = embeddings.shape
    device     = embeddings.device
    total_loss = torch.tensor(0.0, device=device)
    n_valid    = 0

    for b in range(B):
        emb    = embeddings[b]             # (D, H, W)
        m_anch = anchor_mask[b].bool()
        m_pos  = y_in[b].bool()
        m_neg  = (y_out[b] == 0)

        if not m_anch.any() or not m_pos.any() or not m_neg.any():
            continue

        # --- anchors (subsample for memory efficiency) ----------------------
        a_idx = m_anch.nonzero(as_tuple=False)              # (Na, 2)
        perm  = torch.randperm(a_idx.shape[0],
                               device=device)[:max_anchors]
        anch  = emb[:, a_idx[perm, 0], a_idx[perm, 1]].T  # (Na, D)

        # --- positive prototype = mean of Ω_I embeddings -------------------
        p_idx = m_pos.nonzero(as_tuple=False)               # (Np, 2)
        pp    = torch.randperm(p_idx.shape[0],
                               device=device)[:max_pos]
        pos   = emb[:, p_idx[pp, 0],
                    p_idx[pp, 1]].T.mean(0, keepdim=True)  # (1, D)

        # --- negatives = live Ω_O + memory queue ----------------------------
        n_idx    = m_neg.nonzero(as_tuple=False)            # (Nn, 2)
        np_      = torch.randperm(n_idx.shape[0],
                                  device=device)[:max_neg]
        neg_live = emb[:, n_idx[np_, 0],
                       n_idx[np_, 1]].T                     # (Nn, D)
        negatives = torch.cat([neg_live, neg_queue.T], dim=0)  # (Nn+Q, D)

        # --- InfoNCE --------------------------------------------------------
        sim_pos = (anch * pos).sum(dim=1, keepdim=True) / temp  # (Na, 1)
        sim_neg = (anch @ negatives.T) / temp                    # (Na, Nn+Q)
        logits  = torch.cat([sim_pos, sim_neg], dim=1)           # (Na, 1+Nn+Q)
        labels  = torch.zeros(logits.shape[0],
                              dtype=torch.long, device=device)

        total_loss += F.cross_entropy(logits, labels)
        n_valid    += 1

        # enqueue live negative embeddings
        with torch.no_grad():
            model.update_queue(neg_live.detach())

    return total_loss / max(n_valid, 1)


# =========================================================================== #
# Loss 5 — L_patch : boundary consistency inside LRP patches  (README §4)
#
#   README:
#     L_patch = Σ_{x ∈ LRP} KL( p(x) || p̄_category(x) )
#
#     p̄_category(x) = mean prediction of Ω_I  if x is closer to Ω_RF
#                    = mean prediction of Ω_O  if x is closer to Ω_RB
#
#   IMPORTANT: proto_fg is EMA over Ω_I (not lrp_fg).
#              proto_bg is EMA over Ω_O (not lrp_bg).
# =========================================================================== #

def loss_patch(pred, y_in, y_out, lrp_fg, lrp_bg, lrp_uncertain,
               proto_fg, proto_bg, eps=1e-6):
    """
    For each LRP uncertain pixel x:
      - compute distance to nearest Ω_RF pixel (lrp_fg) and
                distance to nearest Ω_RB pixel (lrp_bg)
      - if closer to Ω_RF → KL(p(x) || proto_fg)
      - if closer to Ω_RB → KL(p(x) || proto_bg)

    pred          : (B, 1, H, W)
    lrp_fg        : (B, H, W) float — Ω_RF binary mask
    lrp_bg        : (B, H, W) float — Ω_RB binary mask
    lrp_uncertain : (B, H, W) float — LRP uncertain strip
    proto_fg      : scalar tensor — EMA mean pred on Ω_I
    proto_bg      : scalar tensor — EMA mean pred on Ω_O
    """
    B          = pred.shape[0]    # explicit — no walrus operator
    prob       = torch.sigmoid(pred.squeeze(1))   # (B, H, W)
    total_loss = torch.tensor(0.0, device=pred.device)
    n_valid    = 0

    for b in range(B):
        unc_mask = lrp_uncertain[b].bool()
        if not unc_mask.any():
            continue

        # distance transform from each pixel to the nearest Ω_RF / Ω_RB pixel
        # cv2.distanceTransform wants 0-background, 255-foreground
        # We want dist-to-fg-zone, so invert: background of transform = fg zone
        fg_np = (lrp_fg[b].cpu().numpy() * 255).astype(np.uint8)
        bg_np = (lrp_bg[b].cpu().numpy() * 255).astype(np.uint8)

        # distanceTransform: distance of each 0-pixel to nearest 255-pixel
        fg_inv = (255 - fg_np).astype(np.uint8)   # 0 where lrp_fg=1
        bg_inv = (255 - bg_np).astype(np.uint8)   # 0 where lrp_bg=1

        dist_to_fg = torch.from_numpy(
            cv2.distanceTransform(fg_inv, cv2.DIST_L2, 5)
        ).to(pred.device)
        dist_to_bg = torch.from_numpy(
            cv2.distanceTransform(bg_inv, cv2.DIST_L2, 5)
        ).to(pred.device)

        closer_to_fg = (dist_to_fg <= dist_to_bg) & unc_mask
        closer_to_bg = (dist_to_fg >  dist_to_bg) & unc_mask

        p_unc  = prob[b].clamp(eps, 1.0 - eps)
        loss_b = torch.tensor(0.0, device=pred.device)

        # KL(p_strip || proto_fg)  for pixels closer to Ω_RF
        if closer_to_fg.any():
            p_strip = p_unc[closer_to_fg]
            fg      = proto_fg.clamp(eps, 1.0 - eps)
            kl = (fg * (fg.log() - p_strip.log()) +
                  (1 - fg) * ((1 - fg).log() - (1 - p_strip).log()))
            loss_b = loss_b + kl.mean()

        # KL(p_strip || proto_bg)  for pixels closer to Ω_RB
        if closer_to_bg.any():
            p_strip = p_unc[closer_to_bg]
            bg      = proto_bg.clamp(eps, 1.0 - eps)
            kl = (bg * (bg.log() - p_strip.log()) +
                  (1 - bg) * ((1 - bg).log() - (1 - p_strip).log()))
            loss_b = loss_b + kl.mean()

        total_loss += loss_b
        n_valid    += 1

    return total_loss / max(n_valid, 1)


# =========================================================================== #
# Prototype EMA update  (README: prototypes from Ω_I and Ω_O)
# =========================================================================== #

def update_prototypes(proto_fg, proto_bg, pred, y_in, y_out,
                      decay=EMA_DECAY):
    """
    proto_fg tracks mean sigmoid prediction over Ω_I pixels.
    proto_bg tracks mean sigmoid prediction over Ω_O pixels.

    README: p̄_category = mean prediction of Ω_I or Ω_O — NOT lrp_fg / lrp_bg.
    """
    prob    = torch.sigmoid(pred.squeeze(1)).detach()   # (B, H, W)
    fg_mask = y_in.bool()
    bg_mask = (y_out == 0)

    if fg_mask.any():
        mean_fg  = prob[fg_mask].mean()
        proto_fg = decay * proto_fg + (1.0 - decay) * mean_fg

    if bg_mask.any():
        mean_bg  = prob[bg_mask].mean()
        proto_bg = decay * proto_bg + (1.0 - decay) * mean_bg

    return proto_fg, proto_bg


# =========================================================================== #
# Curriculum phase helper
# =========================================================================== #

def get_phase(epoch, total_epochs):
    """Return training phase (1/2/3) from current epoch."""
    progress = epoch / total_epochs
    if progress < PHASE2_START:
        return 1
    elif progress < PHASE3_START:
        return 2
    else:
        return 3


# =========================================================================== #
# Metrics
# =========================================================================== #

def dice_coefficient(pred_bin, gt_bin, eps=1e-6):
    p = pred_bin.view(-1).float()
    g = gt_bin.view(-1).float()
    return (2.0 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps)


def iou_metric(pred_bin, gt_bin, eps=1e-6):
    p = pred_bin.view(-1).float()
    g = gt_bin.view(-1).float()
    i = (p * g).sum()
    return (i + eps) / (p.sum() + g.sum() - i + eps)


# =========================================================================== #
# Evaluation
# =========================================================================== #

def test(model, path, dataset, opt):
    """Run inference on test/val split and return mean Dice and IoU."""
    data_path = os.path.join(path, dataset)
    model.eval()
    loader = get_loader(
        image_root=f'{data_path}/images/',
        gt_root=f'{data_path}/masks/',
        batchsize=opt.test_batchsize,
        trainsize=opt.img_size,
        shuffle=False,
        augmentation=False
    )
    DSC = IOU = total = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch[0].cuda()
            gt     = batch[-1].cuda().float()
            preds  = model(images, mode='test')
            if isinstance(preds, list):
                preds = preds[-1]   # finest scale
            for i in range(images.shape[0]):
                p   = torch.sigmoid(preds[i]).squeeze()
                b   = (p >= 0.5).float()
                g   = (gt[i].squeeze() >= 0.5).float()
                DSC   += dice_coefficient(b, g).item()
                IOU   += iou_metric(b, g).item()
                total += 1
    n = max(total, 1)
    return DSC / n, IOU / n, int(total)


# =========================================================================== #
# Training — one epoch
# =========================================================================== #

def train(train_loader, model, optimizer, epoch, opt, model_name,
          proto_fg, proto_bg):
    """
    Train for one epoch.

    Phase 1: L = L_c
    Phase 2: L = L_c + λ1·L_PCL(easy) + λ2·L_ce  [+ aux: λ_conf·L_conf]
    Phase 3: L = L_c + λ1·L_PCL(all)  + λ2·L_ce  + λ3·L_patch
                                                    [+ aux: λ_conf·L_conf]

    Returns updated (proto_fg, proto_bg).
    """
    global best, total_train_time

    model.train()
    phase       = get_phase(epoch, opt.epoch)
    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates  = [0.75, 1.0, 1.25]
    total_step  = len(train_loader)

    print(f'\n  Phase {phase}  |  Epoch {epoch}/{opt.epoch}')

    for step, batch in enumerate(train_loader, start=1):
        # unpack the 10-tensor batch from dataloader_hup
        (images,
         y_in, y_out, omega_delta,
         lrp_fg, lrp_bg, lrp_uncertain, lrp_mask,
         y_c, gt) = batch

        images        = images.cuda()
        y_in          = y_in.cuda()
        y_out         = y_out.cuda()
        omega_delta   = omega_delta.cuda()
        lrp_fg        = lrp_fg.cuda()
        lrp_bg        = lrp_bg.cuda()
        lrp_uncertain = lrp_uncertain.cuda()
        lrp_mask      = lrp_mask.cuda()
        y_c           = y_c.cuda()

        for rate in size_rates:
            optimizer.zero_grad()

            # optional multi-scale training
            if rate != 1.0:
                sz       = int(round(opt.img_size * rate / 32) * 32)
                images_r = F.interpolate(images, size=(sz, sz),
                                         mode='bilinear', align_corners=True)
            else:
                images_r = images

            # forward pass
            preds, cls_logits, embeddings, conf_map = model(
                images_r, mode='train'
            )

            # target spatial size = finest prediction size
            pred_size = preds[-1].shape[2:]

            # ----------------------------------------------------------------
            # Resize all annotation maps to pred_size
            # ----------------------------------------------------------------
            def rsz_float(m):
                return F.interpolate(
                    m.unsqueeze(1).float(), size=pred_size, mode='nearest'
                ).squeeze(1)

            def rsz_long(m):
                return F.interpolate(
                    m.unsqueeze(1).float(), size=pred_size, mode='nearest'
                ).squeeze(1).long()

            y_in_r        = rsz_float(y_in)
            y_out_r       = rsz_float(y_out)
            omega_delta_r = rsz_float(omega_delta)
            lrp_fg_r      = rsz_float(lrp_fg)
            lrp_bg_r      = rsz_float(lrp_bg)
            lrp_unc_r     = rsz_float(lrp_uncertain)
            lrp_mask_r    = rsz_float(lrp_mask)
            y_c_r         = rsz_long(y_c)

            cls_r  = F.interpolate(cls_logits, size=pred_size,
                                   mode='bilinear', align_corners=False)
            conf_r = F.interpolate(conf_map, size=pred_size,
                                   mode='bilinear', align_corners=False)
            emb_r  = F.normalize(
                F.interpolate(embeddings, size=pred_size,
                              mode='bilinear', align_corners=False), dim=1
            )

            p_finest = preds[-1]   # finest scale prediction (B,1,H,W)

            # ================================================================
            # Phase 1:  L = L_c
            # ================================================================
            l_c = sum(
                loss_certain(p, y_in_r, y_out_r, lrp_fg_r, lrp_bg_r)
                for p in preds
            )

            l_ce    = torch.tensor(0.0, device=images.device)
            l_conf  = torch.tensor(0.0, device=images.device)
            l_pcl   = torch.tensor(0.0, device=images.device)
            l_patch = torch.tensor(0.0, device=images.device)

            loss = l_c

            # ================================================================
            # Phase 2:  L = L_c + λ1·L_PCL(easy) + λ2·L_ce
            #               + λ_conf·L_conf  [auxiliary]
            # ================================================================
            if phase >= 2:
                # 4-class CCG cross-entropy
                l_ce = loss_ce(cls_r, y_c_r)

                # confidence-head auxiliary supervision
                l_conf = loss_conf(conf_r, lrp_mask_r)

                # PCL with EASY anchors: Ω_Δ pixels NOT in any LRP patch
                easy_anchors = omega_delta_r * (1.0 - lrp_mask_r)
                l_pcl = loss_pcl(
                    emb_r, y_in_r, y_out_r, easy_anchors,
                    model.neg_queue.detach(), model
                )

                loss = (l_c
                        + LAMBDA_PCL  * l_pcl
                        + LAMBDA_CE   * l_ce
                        + LAMBDA_CONF * l_conf)

            # ================================================================
            # Phase 3:  L = L_c + λ1·L_PCL(all) + λ2·L_ce + λ3·L_patch
            #               + λ_conf·L_conf  [auxiliary]
            #
            # L_PCL(all) = single call with ALL uncertain pixels as anchors.
            # This replaces the Phase 2 easy-only PCL — no double-counting.
            # ================================================================
            if phase >= 3:
                # ALL uncertain anchors: global ring ∪ LRP uncertain strip
                all_anchors = torch.clamp(omega_delta_r + lrp_unc_r, 0.0, 1.0)

                # single PCL call over all anchors (README: "L_PCL(all)")
                l_pcl = loss_pcl(
                    emb_r, y_in_r, y_out_r, all_anchors,
                    model.neg_queue.detach(), model
                )

                # boundary consistency inside LRP patches
                l_patch = loss_patch(
                    p_finest,
                    y_in_r, y_out_r,
                    lrp_fg_r, lrp_bg_r, lrp_unc_r,
                    proto_fg, proto_bg
                )

                # single combined expression — no double-lambda
                loss = (l_c
                        + LAMBDA_PCL   * l_pcl
                        + LAMBDA_CE    * l_ce
                        + LAMBDA_PATCH * l_patch
                        + LAMBDA_CONF  * l_conf)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            # update EMA prototypes from Ω_I and Ω_O (NOT from lrp_fg / lrp_bg)
            proto_fg, proto_bg = update_prototypes(
                proto_fg, proto_bg,
                p_finest.detach(), y_in_r, y_out_r
            )

            # record loss at native scale only
            if rate == 1.0:
                loss_record.update(loss.item(), opt.batchsize)

        # logging
        if step % 50 == 0 or step == total_step:
            print(
                f'{datetime.now().strftime("%H:%M:%S")}  '
                f'Ep [{epoch}/{opt.epoch}]  Ph{phase}  '
                f'Step [{step}/{total_step}]  '
                f'Loss {loss_record.show():.4f}  '
                f'L_c {l_c.item():.3f}  '
                f'L_ce {l_ce.item():.3f}  '
                f'L_pcl {l_pcl.item():.3f}  '
                f'L_patch {l_patch.item():.3f}  '
                f'L_conf {l_conf.item():.4f}'
            )

    total_train_time += time.time() - epoch_start

    # save latest checkpoint
    os.makedirs(opt.train_save, exist_ok=True)
    ckpt_path = os.path.join(opt.train_save, f'{model_name}-last.pth')
    torch.save(model.state_dict(), ckpt_path)

    # validate
    model.eval()
    d_dice, d_iou, n_samples = test(model, opt.test_path, 'val', opt)
    print(f'  Val  Dice: {d_dice:.4f}  IoU: {d_iou:.4f}  '
          f'(n={n_samples})  '
          f'proto_fg={proto_fg.item():.3f}  proto_bg={proto_bg.item():.3f}')

    global best
    if d_dice > best:
        best = d_dice
        best_path = os.path.join(opt.train_save, f'{model_name}-best.pth')
        torch.save(model.state_dict(), best_path)
        print(f'  ✓  New best ({best:.4f}) saved → {best_path}')

    return proto_fg, proto_bg


# =========================================================================== #
# Entry point
# =========================================================================== #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HUPAnno WS-EMCADNet training'
    )
    parser.add_argument('--epoch',          type=int,   default=200)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--batchsize',      type=int,   default=8)
    parser.add_argument('--test_batchsize', type=int,   default=8)
    parser.add_argument('--img_size',       type=int,   default=352)
    parser.add_argument('--clip',           type=float, default=0.5)
    parser.add_argument('--train_path',     type=str,   required=True,
                        help='Path to training split (contains images/ masks/)')
    parser.add_argument('--test_path',      type=str,   required=True,
                        help='Root path (contains val/ and/or test/ subdirs)')
    parser.add_argument('--train_save',     type=str,   default='./model_pth_hup/')
    parser.add_argument('--K',              type=int,   default=2,
                        help='Number of LRP patches per image')
    opt = parser.parse_args()

    # --- device -------------------------------------------------------------
    device   = torch.device('cuda')
    proto_fg = torch.tensor(0.8, device=device)
    proto_bg = torch.tensor(0.2, device=device)

    # --- model --------------------------------------------------------------
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
    ).to(device)

    # --- optimiser & scheduler ----------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=opt.epoch, eta_min=1e-6
    )

    # --- data ---------------------------------------------------------------
    train_loader = get_loader(
        image_root=f'{opt.train_path}/images/',
        gt_root=f'{opt.train_path}/masks/',
        batchsize=opt.batchsize,
        trainsize=opt.img_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        augmentation=True,
        K=opt.K
    )

    # --- curriculum summary -------------------------------------------------
    p2_ep = int(opt.epoch * PHASE2_START)
    p3_ep = int(opt.epoch * PHASE3_START)
    print(f'\nHUPAnno training  |  {opt.epoch} epochs  |  K={opt.K} patches')
    print(f'  Phase 1  ep 1–{p2_ep}          : L = L_c')
    print(f'  Phase 2  ep {p2_ep+1}–{p3_ep}  '
          f': L = L_c + λ1·PCL(easy) + λ2·L_ce  [+aux L_conf]')
    print(f'  Phase 3  ep {p3_ep+1}–{opt.epoch}  '
          f': L = L_c + λ1·PCL(all) + λ2·L_ce + λ3·L_patch  [+aux L_conf]')
    print(f'  λ_PCL={LAMBDA_PCL}  λ_CE={LAMBDA_CE}  '
          f'λ_patch={LAMBDA_PATCH}  λ_conf={LAMBDA_CONF}')
    print(f'  μ_hard={MU_HARD} (inside LRP, stricter)  '
          f'μ_easy={MU_EASY} (outside, tolerant)\n')

    # --- training loop ------------------------------------------------------
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 300)
        proto_fg, proto_bg = train(
            train_loader, model, optimizer, epoch, opt,
            'ws_hupanno', proto_fg, proto_bg
        )
        scheduler.step()

    print(f'\nDone.')
    print(f'Total training time : {total_train_time / 3600:.2f} h')
    print(f'Best val Dice     s  : {best:.4f}')