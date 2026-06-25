# train_polyp.py  —  WS-EMCADNet + HUPAnno
# Every design decision traceable to annotation_ideas_detailed.md README
#
# README §1  →  loss_certain : Dice for all four terms, background Dice uses
#               capped random subsample to fix denominator blow-up
# README §2  →  loss_pcl    : LRP pixels are PRIMARY hard samples (ρ=0.85),
#               entropy pixels are SECONDARY
# README §3  →  loss_conf   : μ_hard=0.3 inside LRP (stricter),
#               μ=0.5 outside (tolerant)
# README §4  →  loss_patch  : KL toward mean of Ω_I / Ω_O
# README curriculum → Phase 1 / 2 / 3 at 0-30 / 30-70 / 70-100%

import os
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.ndimage import distance_transform_edt

from lib.networks import EMCADNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, AvgMeter


# =========================================================================== #
# Hyperparameters
# =========================================================================== #

LAMBDA_PCL   = 0.1
LAMBDA_CE    = 0.3
LAMBDA_PATCH = 0.2
LAMBDA_CONF  = 0.05

# README §3: μ_hard LOWER (stricter) inside LRP, μ HIGHER (tolerant) outside
MU_HARD = 0.3
MU_EASY = 0.5

# README §2: aggressive ratio for annotator-verified hard samples
RHO_LRP  = 0.85   # sampling ratio inside LRP patches
RHO_EASY = 0.5    # sampling ratio outside LRP patches

PCL_TEMP     = 0.07
PHASE2_START = 0.30
PHASE3_START = 0.70
EMA_DECAY    = 0.99

# background Dice subsample cap — prevents denominator blow-up
# when bg_mask covers ~80% of image (README §1 fix)
BG_SAMPLE_CAP = 2048

WARMUP_EPOCHS = 3

best             = 0.0
total_train_time = 0.0
epoch_history    = []


# =========================================================================== #
# LR helpers
# =========================================================================== #

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    lr = base_lr * (epoch / warmup_epochs)
    for pg in optimizer.param_groups:
        pg['lr'] = lr


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


def hd95_metric(pred_np, gt_np):
    pred = pred_np.astype(bool)
    gt   = gt_np.astype(bool)
    if not pred.any() or not gt.any():
        h, w = pred.shape
        return float(np.sqrt(h**2 + w**2))
    d1 = distance_transform_edt(~pred)[gt]
    d2 = distance_transform_edt(~gt)[pred]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


# =========================================================================== #
# Loss 1 — L_c  (README §1)
#
# "L_c = L_in(p,y_in) + L_out(p,y_out) + L_RF(p,y_RF) + L_RB(p,y_RB)"
# "These are dice losses computed on the LRP-resolved foreground and
#  background pixels. Treated with the same high confidence as Ω_I / Ω_O."
#
# All four terms are Dice.
# Background terms (l_out, l_rb) use a randomly capped subsample of bg pixels
# so the Dice denominator does not blow up when bg_mask covers ~80% of image.
# This is a numerical fix that does not change the mathematical intent.
# =========================================================================== #

def soft_dice_fg(pred_logit, mask, eps=1e-6):
    """Dice loss pushing masked pixels toward 1 (foreground)."""
    if not mask.any():
        return torch.tensor(0.0, device=pred_logit.device, requires_grad=True)
    prob  = torch.sigmoid(pred_logit[mask])
    inter = prob.sum()
    denom = prob.sum() + mask.sum().float()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def soft_dice_bg(pred_logit, mask, cap=BG_SAMPLE_CAP, eps=1e-6):
    """
    Dice loss pushing masked pixels toward 0 (background).
    Randomly subsamples to cap pixels to avoid denominator blow-up.
    README: same Dice formulation as foreground, just target=0.
    """
    if not mask.any():
        return torch.tensor(0.0, device=pred_logit.device, requires_grad=True)

    idx = mask.nonzero(as_tuple=False)          # (N, ndim)
    if idx.shape[0] > cap:
        perm = torch.randperm(idx.shape[0], device=pred_logit.device)[:cap]
        idx  = idx[perm]

    # rebuild subsampled mask
    sub_mask = torch.zeros_like(mask)
    if mask.dim() == 2:
        sub_mask[idx[:, 0], idx[:, 1]] = True
    else:
        # batch dim present — flatten approach
        flat_pred = pred_logit.view(-1)
        flat_mask = mask.view(-1)
        flat_idx  = flat_mask.nonzero(as_tuple=False).squeeze(1)
        if flat_idx.shape[0] > cap:
            perm     = torch.randperm(flat_idx.shape[0],
                                      device=pred_logit.device)[:cap]
            flat_idx = flat_idx[perm]
        prob  = torch.sigmoid(flat_pred[flat_idx])
        denom = prob.sum() + flat_idx.shape[0]
        return 1.0 - (eps) / (denom + eps)   # target=0 → inter=0 always

    prob  = torch.sigmoid(pred_logit[sub_mask])
    # Dice with target=0: inter = Σ p*(1-p)... No: target IS 0.
    # Dice(pred, target=0) = 2*Σ(pred*(1-target)) / (Σpred + Σ(1-target))
    # = 2*0 / ... but that makes no sense for background.
    # Correct formulation: treat background as foreground of (1-pred).
    prob_bg = 1.0 - prob                         # flip: bg=1, fg=0
    inter   = prob_bg.sum()
    denom   = prob_bg.sum() + sub_mask.sum().float()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def loss_certain(pred, y_in, y_out, lrp_fg, lrp_bg):
    """
    README §1:
      L_c = L_in + L_out + L_RF + L_RB   — all Dice, equal weights.

    l_in  : Dice toward 1 on Ω_I
    l_out : Dice toward 0 on Ω_O  (capped subsample)
    l_rf  : Dice toward 1 on Ω_RF (LRP resolved FG)
    l_rb  : Dice toward 0 on Ω_RB (LRP resolved BG, capped subsample)
    """
    p = pred.squeeze(1)   # (B, H, W)

    # Ω_I → predict foreground
    l_in  = soft_dice_fg(p, y_in.bool())

    # Ω_O → predict background (capped Dice)
    l_out = soft_dice_bg(p, (y_out == 0))

    # Ω_RF → predict foreground
    lrf   = lrp_fg.bool()
    l_rf  = soft_dice_fg(p, lrf) if lrf.any() else \
            torch.tensor(0.0, device=pred.device)

    # Ω_RB → predict background (capped Dice)
    lrb   = lrp_bg.bool()
    l_rb  = soft_dice_bg(p, lrb) if lrb.any() else \
            torch.tensor(0.0, device=pred.device)

    return l_in + l_out + l_rf + l_rb


# =========================================================================== #
# Loss 2 — L_ce  (README §3)
# =========================================================================== #

def loss_ce(cls_logits, y_c):
    return F.cross_entropy(cls_logits, y_c)


# =========================================================================== #
# Loss 3 — L_conf  (README §3)
#
# "μ_i = μ_hard if x_i ∈ LRP region, μ otherwise"
# μ_hard=0.3 (LOWER = stricter inside LRP)
# μ=0.5      (HIGHER = tolerant outside LRP)
# =========================================================================== #

def loss_conf(conf_map, lrp_mask, mu_hard=MU_HARD, mu_easy=MU_EASY):
    target = torch.full_like(conf_map.squeeze(1), mu_easy)
    target[lrp_mask.bool()] = mu_hard
    return F.mse_loss(conf_map.squeeze(1), target)


# =========================================================================== #
# Loss 4 — L_PCL  (README §2)
#
# "Primary hard samples: pixels within LRP patches (annotator-identified)"
# "Secondary hard samples: pixels in ΩΔ of easy segments with high entropy"
# "ρ = 0.85 in LRP-covered regions" (aggressive ratio)
#
# Implementation:
#   Phase 2 anchors = easy uncertain pixels (Ω_Δ non-LRP, entropy-selected)
#   Phase 3 anchors = LRP uncertain (primary, ρ=0.85) +
#                     easy uncertain (secondary, ρ=0.5)
# =========================================================================== #

def sample_anchors_lrp(emb, lrp_unc_mask, rho=RHO_LRP, max_n=256):
    """
    README §2: annotator-identified LRP pixels are primary hard samples.
    Sample aggressively (ρ=0.85 means keep 85% of LRP uncertain pixels).
    """
    idx = lrp_unc_mask.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return None
    n    = max(1, int(idx.shape[0] * rho))
    n    = min(n, max_n)
    perm = torch.randperm(idx.shape[0], device=emb.device)[:n]
    return emb[:, idx[perm, 0], idx[perm, 1]].T   # (n, D)


def sample_anchors_easy(emb, pred, easy_unc_mask, rho=RHO_EASY, max_n=256):
    """
    README §2: secondary hard samples from easy segments via entropy.
    Uses predictive entropy to select hard pixels within easy uncertain band.
    """
    idx = easy_unc_mask.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return None

    # entropy-based selection within easy uncertain pixels
    prob    = torch.sigmoid(pred.squeeze(0))   # (H, W)
    entropy = -(prob * (prob + 1e-6).log() +
                (1 - prob) * (1 - prob + 1e-6).log())
    ent_vals = entropy[idx[:, 0], idx[:, 1]]

    # keep top-ρ fraction by entropy
    n     = max(1, int(idx.shape[0] * rho))
    n     = min(n, max_n)
    topk  = torch.topk(ent_vals, n).indices
    sel   = idx[topk]
    return emb[:, sel[:, 0], sel[:, 1]].T   # (n, D)


def loss_pcl(embeddings, pred, y_in, y_out,
             omega_delta, lrp_mask, lrp_uncertain,
             neg_queue, model, phase,
             temp=PCL_TEMP, max_pos=128, max_neg=128):
    """
    README §2 faithful implementation.

    Phase 2: anchors = easy uncertain pixels (entropy-selected secondary)
    Phase 3: anchors = LRP primary (ρ=0.85) + easy secondary (ρ=0.5)
    """
    B, D, H, W = embeddings.shape
    device     = embeddings.device
    total_loss = torch.tensor(0.0, device=device)
    n_valid    = 0

    for b in range(B):
        emb    = embeddings[b]            # (D, H, W)
        m_pos  = y_in[b].bool()
        m_neg  = (y_out[b] == 0)

        if not m_pos.any() or not m_neg.any():
            continue

        # ── collect anchors ──────────────────────────────────────────────────
        anchor_list = []

        easy_unc = (omega_delta[b] == 1) & ~lrp_mask[b].bool()

        if phase == 2:
            # secondary only: easy uncertain with entropy
            a = sample_anchors_easy(emb, pred[b:b+1], easy_unc)
            if a is not None:
                anchor_list.append(a)

        elif phase >= 3:
            # primary: LRP uncertain (annotator-verified, aggressive ρ)
            lrp_unc = lrp_uncertain[b].bool()
            a_lrp   = sample_anchors_lrp(emb, lrp_unc)
            if a_lrp is not None:
                anchor_list.append(a_lrp)

            # secondary: easy uncertain with entropy
            a_easy = sample_anchors_easy(emb, pred[b:b+1], easy_unc)
            if a_easy is not None:
                anchor_list.append(a_easy)

        if not anchor_list:
            continue

        anch = torch.cat(anchor_list, dim=0)   # (Na, D)

        # ── positive prototype = mean of Ω_I ─────────────────────────────────
        p_idx = m_pos.nonzero(as_tuple=False)
        pp    = torch.randperm(p_idx.shape[0], device=device)[:max_pos]
        pos   = emb[:, p_idx[pp, 0], p_idx[pp, 1]].T.mean(0, keepdim=True)

        # ── negatives = live Ω_O + queue ─────────────────────────────────────
        n_idx    = m_neg.nonzero(as_tuple=False)
        np_      = torch.randperm(n_idx.shape[0], device=device)[:max_neg]
        neg_live = emb[:, n_idx[np_, 0], n_idx[np_, 1]].T
        negs     = torch.cat([neg_live, neg_queue.T], dim=0)

        # ── InfoNCE ──────────────────────────────────────────────────────────
        sim_pos = (anch * pos).sum(1, keepdim=True) / temp
        sim_neg = (anch @ negs.T) / temp
        logits  = torch.cat([sim_pos, sim_neg], dim=1)
        labels  = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        total_loss += F.cross_entropy(logits, labels)
        n_valid    += 1

        with torch.no_grad():
            model.update_queue(neg_live.detach())

    return total_loss / max(n_valid, 1)


# =========================================================================== #
# Loss 5 — L_patch  (README §4)
#
# "L_patch = Σ_{x ∈ LRP} KL(p(x) || p̄_category(x))"
# "p̄_category(x) = mean prediction of Ω_I  if x is LRP-foreground"
#                  "mean prediction of Ω_O  if x is LRP-background"
# =========================================================================== #

def loss_patch(pred, lrp_fg, lrp_bg, lrp_uncertain,
               proto_fg, proto_bg, eps=1e-6):
    """
    README §4 faithful:
      For each LRP uncertain pixel x:
        closer to Ω_RF → KL(p(x) || proto_fg)   proto_fg = EMA(mean Ω_I pred)
        closer to Ω_RB → KL(p(x) || proto_bg)   proto_bg = EMA(mean Ω_O pred)
    """
    B          = pred.shape[0]
    prob       = torch.sigmoid(pred.squeeze(1))
    total_loss = torch.tensor(0.0, device=pred.device)
    n_valid    = 0

    for b in range(B):
        unc_mask = lrp_uncertain[b].bool()
        if not unc_mask.any():
            continue

        fg_np  = (lrp_fg[b].cpu().numpy() * 255).astype(np.uint8)
        bg_np  = (lrp_bg[b].cpu().numpy() * 255).astype(np.uint8)
        fg_inv = (255 - fg_np).astype(np.uint8)
        bg_inv = (255 - bg_np).astype(np.uint8)

        dist_to_fg = torch.from_numpy(
            cv2.distanceTransform(fg_inv, cv2.DIST_L2, 5)
        ).to(pred.device)
        dist_to_bg = torch.from_numpy(
            cv2.distanceTransform(bg_inv, cv2.DIST_L2, 5)
        ).to(pred.device)

        closer_fg = (dist_to_fg <= dist_to_bg) & unc_mask
        closer_bg = (dist_to_fg >  dist_to_bg) & unc_mask
        p_unc     = prob[b].clamp(eps, 1 - eps)
        loss_b    = torch.tensor(0.0, device=pred.device)

        if closer_fg.any():
            ps = p_unc[closer_fg]
            fg = proto_fg.clamp(eps, 1 - eps)
            kl = fg*(fg.log()-ps.log()) + (1-fg)*((1-fg).log()-(1-ps).log())
            loss_b = loss_b + kl.mean()

        if closer_bg.any():
            ps = p_unc[closer_bg]
            bg = proto_bg.clamp(eps, 1 - eps)
            kl = bg*(bg.log()-ps.log()) + (1-bg)*((1-bg).log()-(1-ps).log())
            loss_b = loss_b + kl.mean()

        total_loss += loss_b
        n_valid    += 1

    return total_loss / max(n_valid, 1)


# =========================================================================== #
# Prototype EMA  (README §4: from Ω_I and Ω_O)
# =========================================================================== #

def update_prototypes(proto_fg, proto_bg, pred, y_in, y_out,
                      decay=EMA_DECAY):
    prob    = torch.sigmoid(pred.squeeze(1)).detach()
    fg_mask = y_in.bool()
    bg_mask = (y_out == 0)
    if fg_mask.any():
        proto_fg = decay * proto_fg + (1 - decay) * prob[fg_mask].mean()
    if bg_mask.any():
        proto_bg = decay * proto_bg + (1 - decay) * prob[bg_mask].mean()
    return proto_fg, proto_bg


# =========================================================================== #
# Phase helper
# =========================================================================== #

def get_phase(epoch, total_epochs):
    p = epoch / total_epochs
    if p < PHASE2_START: return 1
    elif p < PHASE3_START: return 2
    else: return 3


# =========================================================================== #
# Evaluation — Dice + IoU + HD95
# =========================================================================== #

def test(model, image_root, mask_root, opt):
    model.eval()
    loader = get_loader(
        image_root   = image_root,
        gt_root      = mask_root,
        batchsize    = opt.test_batchsize,
        trainsize    = opt.img_size,
        shuffle      = False,
        num_workers  = 2,
        pin_memory   = True,
        augmentation = False,
        K            = opt.K
    )
    DSC = IOU = HD = total = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch[0].cuda()
            gt     = batch[-1].cuda().float()
            preds  = model(images, mode='test')
            if isinstance(preds, list):
                preds = preds[-1]
            for i in range(images.shape[0]):
                p_prob = torch.sigmoid(preds[i]).squeeze()
                p_bin  = (p_prob >= 0.5).float()
                g_bin  = (gt[i].squeeze() >= 0.5).float()
                DSC   += dice_coefficient(p_bin, g_bin).item()
                IOU   += iou_metric(p_bin, g_bin).item()
                HD    += hd95_metric(
                    p_bin.cpu().numpy().astype(np.uint8),
                    g_bin.cpu().numpy().astype(np.uint8)
                )
                total += 1
    n = max(total, 1)
    return DSC/n, IOU/n, HD/n, int(total)


# =========================================================================== #
# Summary table
# =========================================================================== #

def print_summary_table():
    print(f'\n{"="*72}')
    print(f'  TRAINING SUMMARY')
    print(f'{"="*72}')
    print(f'  {"Ep":>4}  {"Ph"}  {"LR":>9}  {"Loss":>8}  '
          f'{"Dice":>8}  {"IoU":>8}  {"HD95":>8}')
    print(f'  {"─"*4}  {"──"}  {"─"*9}  {"─"*8}  '
          f'{"─"*8}  {"─"*8}  {"─"*8}')
    for r in epoch_history:
        marker = ' ✓' if r['best'] else ''
        print(f'  {r["epoch"]:>4}  Ph{r["phase"]}  '
              f'{r["lr"]:>9.2e}  '
              f'{r["loss"]:>8.4f}  '
              f'{r["dice"]:>8.4f}  '
              f'{r["iou"]:>8.4f}  '
              f'{r["hd95"]:>8.2f}'
              f'{marker}')
    print(f'{"="*72}\n')


# =========================================================================== #
# Training — one epoch
# =========================================================================== #

def train(train_loader, model, optimizer, epoch, opt, model_name,
          proto_fg, proto_bg):
    global best, total_train_time, epoch_history

    model.train()
    phase       = get_phase(epoch, opt.epoch)
    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates  = [0.75, 1.0, 1.25]
    total_step  = len(train_loader)
    current_lr  = get_lr(optimizer)

    print(f'\n{"="*65}')
    print(f'  Epoch {epoch}/{opt.epoch}  |  Phase {phase}  |  '
          f'LR {current_lr:.2e}  |  '
          f'{datetime.now().strftime("%H:%M:%S")}')
    print(f'{"="*65}')

    for step, batch in enumerate(train_loader, start=1):
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

            if rate != 1.0:
                sz       = int(round(opt.img_size * rate / 32) * 32)
                images_r = F.interpolate(images, size=(sz, sz),
                                         mode='bilinear', align_corners=True)
            else:
                images_r = images

            preds, cls_logits, embeddings, conf_map = model(
                images_r, mode='train'
            )
            pred_size = preds[-1].shape[2:]

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
            p_finest = preds[-1]

            # ── Phase 1: L = L_c ─────────────────────────────────────────────
            l_c = sum(
                loss_certain(p, y_in_r, y_out_r, lrp_fg_r, lrp_bg_r)
                for p in preds
            )

            l_ce    = torch.tensor(0.0, device=images.device)
            l_conf  = torch.tensor(0.0, device=images.device)
            l_pcl   = torch.tensor(0.0, device=images.device)
            l_patch = torch.tensor(0.0, device=images.device)
            loss    = l_c

            # ── Phase 2: L = L_c + λ1·PCL(easy) + λ2·Lce ───────────────────
            if phase >= 2:
                l_ce   = loss_ce(cls_r, y_c_r)
                l_conf = loss_conf(conf_r, lrp_mask_r)
                l_pcl  = loss_pcl(
                    emb_r, p_finest,
                    y_in_r, y_out_r,
                    omega_delta_r, lrp_mask_r, lrp_unc_r,
                    model.neg_queue.detach(), model,
                    phase=2
                )
                loss = (l_c
                        + LAMBDA_PCL  * l_pcl
                        + LAMBDA_CE   * l_ce
                        + LAMBDA_CONF * l_conf)

            # ── Phase 3: L = L_c + λ1·PCL(all) + λ2·Lce + λ3·Lpatch ────────
            if phase >= 3:
                # README: PCL(all) = LRP primary + easy secondary
                l_pcl = loss_pcl(
                    emb_r, p_finest,
                    y_in_r, y_out_r,
                    omega_delta_r, lrp_mask_r, lrp_unc_r,
                    model.neg_queue.detach(), model,
                    phase=3
                )
                # README §4: L_patch on LRP uncertain pixels
                l_patch = loss_patch(
                    p_finest,
                    lrp_fg_r, lrp_bg_r, lrp_unc_r,
                    proto_fg, proto_bg
                )
                loss = (l_c
                        + LAMBDA_PCL   * l_pcl
                        + LAMBDA_CE    * l_ce
                        + LAMBDA_PATCH * l_patch
                        + LAMBDA_CONF  * l_conf)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            proto_fg, proto_bg = update_prototypes(
                proto_fg, proto_bg,
                p_finest.detach(), y_in_r, y_out_r
            )

            if rate == 1.0:
                loss_record.update(loss.item(), opt.batchsize)

        if step % 50 == 0 or step == total_step:
            print(
                f'  {datetime.now().strftime("%H:%M:%S")}  '
                f'Step [{step:>4}/{total_step}]  '
                f'Loss {loss_record.show():.4f}  '
                f'Lc {l_c.item():.3f}  '
                f'Lce {l_ce.item():.3f}  '
                f'Lpcl {l_pcl.item():.3f}  '
                f'Lpatch {l_patch.item():.3f}  '
                f'Lconf {l_conf.item():.4f}'
            )

    epoch_time        = time.time() - epoch_start
    total_train_time += epoch_time

    os.makedirs(opt.train_save, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(opt.train_save, f'{model_name}-last.pth'))

    model.eval()
    d_dice, d_iou, d_hd95, n_samples = test(
        model, opt.val_image_root, opt.val_mask_root, opt
    )

    is_best = d_dice > best
    marker  = ' ← best' if is_best else ''

    print(f'\n  ┌─ Epoch {epoch:>3}/{opt.epoch}  Ph{phase}  '
          f'LR {current_lr:.2e}  ({epoch_time/60:.1f} min) ───────')
    print(f'  │  Train Loss : {loss_record.show():.4f}')
    print(f'  │  Val  Dice  : {d_dice:.4f}{marker}')
    print(f'  │  Val  IoU   : {d_iou:.4f}')
    print(f'  │  Val  HD95  : {d_hd95:.2f} px')
    print(f'  │  proto_fg={proto_fg.item():.3f}  '
          f'proto_bg={proto_bg.item():.3f}  n={n_samples}')
    print(f'  └{"─"*55}')

    if is_best:
        best = d_dice
        torch.save(model.state_dict(),
                   os.path.join(opt.train_save, f'{model_name}-best.pth'))
        print(f'  ✓  Best  Dice={best:.4f}  IoU={d_iou:.4f}  '
              f'HD95={d_hd95:.2f}')

    epoch_history.append({
        'epoch' : epoch,
        'phase' : phase,
        'lr'    : current_lr,
        'loss'  : loss_record.show(),
        'dice'  : d_dice,
        'iou'   : d_iou,
        'hd95'  : d_hd95,
        'best'  : is_best,
    })

    return proto_fg, proto_bg


# =========================================================================== #
# Entry point
# =========================================================================== #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',            type=int,   default=200)
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--batchsize',        type=int,   default=8)
    parser.add_argument('--test_batchsize',   type=int,   default=8)
    parser.add_argument('--img_size',         type=int,   default=352)
    parser.add_argument('--clip',             type=float, default=0.5)
    parser.add_argument('--K',                type=int,   default=2)
    parser.add_argument('--train_save',       type=str,
                        default='./model_pth_hup/')
    parser.add_argument('--train_image_root', type=str,   required=True)
    parser.add_argument('--train_mask_root',  type=str,   required=True)
    parser.add_argument('--val_image_root',   type=str,   required=True)
    parser.add_argument('--val_mask_root',    type=str,   required=True)
    opt = parser.parse_args()

    for label, path in [
        ('train_image_root', opt.train_image_root),
        ('train_mask_root',  opt.train_mask_root),
        ('val_image_root',   opt.val_image_root),
        ('val_mask_root',    opt.val_mask_root),
    ]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f'--{label} not found: {path}')
    print('✓  All data paths verified.')

    device           = torch.device('cuda')
    best             = 0.0
    total_train_time = 0.0
    epoch_history    = []
    proto_fg         = torch.tensor(0.8, device=device)
    proto_bg         = torch.tensor(0.2, device=device)

    model = EMCADNet(
        num_classes=1, kernel_sizes=[1,3,5],
        expansion_factor=2, dw_parallel=True, add=True,
        lgag_ks=3, activation='relu6',
        encoder='pvt_v2_b2', pretrain=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=opt.epoch, eta_min=1e-6
    )

    train_loader = get_loader(
        image_root   = opt.train_image_root,
        gt_root      = opt.train_mask_root,
        batchsize    = opt.batchsize,
        trainsize    = opt.img_size,
        shuffle      = True,
        num_workers  = 4,
        pin_memory   = True,
        augmentation = True,
        K            = opt.K
    )

    p2_ep = int(opt.epoch * PHASE2_START)
    p3_ep = int(opt.epoch * PHASE3_START)
    print(f'\nHUPAnno  |  {opt.epoch} epochs  |  K={opt.K}')
    print(f'  Phase 1  ep 1–{p2_ep}       : L = L_c')
    print(f'  Phase 2  ep {p2_ep+1}–{p3_ep}  : '
          f'L = L_c + PCL(easy,secondary) + Lce')
    print(f'  Phase 3  ep {p3_ep+1}–{opt.epoch}  : '
          f'L = L_c + PCL(LRP-primary+easy-secondary) + Lce + Lpatch')
    print(f'  ρ_LRP={RHO_LRP} (aggressive, annotator-verified)  '
          f'ρ_easy={RHO_EASY}')
    print(f'  μ_hard={MU_HARD} (LRP stricter)  μ_easy={MU_EASY} (tolerant)')
    print(f'  Train: {opt.train_image_root}')
    print(f'  Val  : {opt.val_image_root}\n')

    for epoch in range(1, opt.epoch + 1):
        if epoch <= WARMUP_EPOCHS:
            warmup_lr(optimizer, epoch, WARMUP_EPOCHS, opt.lr)
        else:
            scheduler.step()

        proto_fg, proto_bg = train(
            train_loader, model, optimizer, epoch, opt,
            'ws_hupanno', proto_fg, proto_bg
        )

    print_summary_table()
    print(f'Total time : {total_train_time/3600:.2f} h')
    print(f'Best Dice  : {best:.4f}')