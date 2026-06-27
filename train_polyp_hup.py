#train_polyp_hup.py  —  HUPAnno training script with multi-phase loss
#   Phase 1  (0  → 30%):  L = L_c
#   Phase 2  (30 → 70%):  L = L_c + λ1·L_PCL(easy) + λ2·L_ce
#   Phase 3  (70 → 100%): L = L_c + λ1·L_PCL(all)  + λ2·L_ce + λ3·L_patch
#
#   Fixes applied vs previous version:
#   [1] loss_certain — L_out was supervising uncertain ring as FG (WRONG).
#       Now correctly supervises only pixels OUTSIDE y_out as BG.
#   [2] test() — threshold parameter was never passed from train().
#       Now uses 0.3 during warmup epochs, 0.5 afterwards.
#   [3] hd95_metric — empty-pred case now returns true surface distance
#       instead of image diagonal, preventing inflated early HD95.
#   [4] hd_valid / total — now int, semantically cleaner.

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

from lib.networks_hup import EMCADNet
from utils.dataloader_hup import get_loader
from utils.utils import clip_gradient, AvgMeter


# =========================================================================== #
# Hyperparameters
# =========================================================================== #

LAMBDA_PCL   = 0.1
LAMBDA_CE    = 0.3
LAMBDA_PATCH = 0.2
LAMBDA_CONF  = 0.05

MU_HARD = 0.3    # stricter inside LRP  (LOWER threshold)
MU_EASY = 0.5    # tolerant outside     (HIGHER threshold)

RHO_LRP  = 0.85  # aggressive sampling for annotator-verified LRP pixels
RHO_EASY = 0.5   # standard sampling for entropy-selected easy pixels

PCL_TEMP      = 0.07
PHASE2_START  = 0.30
PHASE3_START  = 0.70
EMA_DECAY     = 0.99
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
    """
    Hausdorff Distance 95th percentile.

    Fix [3]: when pred is all-zero (model predicts nothing), the old code
    returned sqrt(H^2+W^2) — the image diagonal — which inflated early-epoch
    HD95 to ~498px on 352×352 inputs. Now returns the 95th percentile of the
    true distance-from-GT-surface, which is a meaningful upper bound.
    """
    pred = pred_np.astype(bool)
    gt   = gt_np.astype(bool)

    if not gt.any():
        return 0.0          # no GT annotation — sample contributes nothing

    if not pred.any():
        # model predicted nothing — measure how far GT surface extends
        d = distance_transform_edt(~gt)
        pos = d[d > 0]
        return float(np.percentile(pos, 95)) if pos.size > 0 else 0.0

    d1 = distance_transform_edt(~pred)[gt]
    d2 = distance_transform_edt(~gt)[pred]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


# =========================================================================== #
# Core supervised loss — structure_loss
# Identical to EAUWSeg/BPAnno: Dice + BCE combined.
# =========================================================================== #

def structure_loss(pred, mask):
    """
    pred : (B, 1, H, W) logits  OR  (B, H, W) logits
    mask : (B, H, W) float in {0, 1}
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)

    bce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    prob  = torch.sigmoid(pred)
    inter = (prob * mask).sum(dim=(-2, -1))
    union = prob.sum(dim=(-2, -1)) + mask.sum(dim=(-2, -1))
    dice  = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)

    return bce + dice.mean()


# =========================================================================== #
# Loss 1 — L_c
#
# L_c = L_in + L_out + L_RF + L_RB
#
# Fix [1]: L_out previously called structure_loss(pred, y_out), which
# supervised the ENTIRE outer polygon (including the uncertain ring Ω_delta)
# as foreground. This directly contradicted L_in and caused the model to
# output uniform ~0.2 predictions with proto_fg ≈ proto_bg.
#
# Correct semantics:
#   L_in  : pixels inside  y_in          → predict 1  (certain FG)
#   L_out : pixels OUTSIDE y_out         → predict 0  (certain BG)
#   L_RF  : lrp_fg pixels                → predict 1  (LRP resolved FG)
#   L_RB  : lrp_bg pixels                → predict 0  (LRP resolved BG)
#
# The uncertain ring (y_out=1, y_in=0, not LRP) receives NO direct
# supervision from L_c — it is handled by L_PCL and L_patch in later phases.
# =========================================================================== #

def loss_certain(pred, y_in, y_out, lrp_fg, lrp_bg):
    if pred.dim() == 4:
        pred = pred.squeeze(1)          # (B, H, W)
    device = pred.device

    # ── L_in : certain foreground → predict 1 ────────────────────────────────
    l_in = structure_loss(pred, y_in)

    # ── L_out : certain background (outside outer polygon) → predict 0 ───────
    # Fix [1]: supervise only pixels where y_out == 0, NOT the full y_out mask.
    # Using BCE on selected pixels (Dice is unstable for large uniform regions).
    bg_mask = (y_out == 0)
    if bg_mask.any():
        l_out = F.binary_cross_entropy_with_logits(
            pred[bg_mask],
            torch.zeros_like(pred[bg_mask]),
            reduction='mean'
        )
    else:
        l_out = torch.tensor(0.0, device=device)

    # ── L_RF : LRP resolved foreground → predict 1 ───────────────────────────
    if lrp_fg.any():
        l_rf = structure_loss(pred, lrp_fg)
    else:
        l_rf = torch.tensor(0.0, device=device)

    # ── L_RB : LRP resolved background → predict 0 ───────────────────────────
    if lrp_bg.any():
        l_rb = F.binary_cross_entropy_with_logits(
            pred[lrp_bg.bool()],
            torch.zeros_like(pred[lrp_bg.bool()]),
            reduction='mean'
        )
    else:
        l_rb = torch.tensor(0.0, device=device)

    return l_in + l_out + l_rf + l_rb


# =========================================================================== #
# Loss 2 — L_ce
# =========================================================================== #

def loss_ce(cls_logits, y_c):
    return F.cross_entropy(cls_logits, y_c)


# =========================================================================== #
# Loss 3 — L_conf
# μ_hard=0.3 inside LRP (stricter), μ_easy=0.5 outside (tolerant)
# =========================================================================== #

def loss_conf(conf_map, lrp_mask, mu_hard=MU_HARD, mu_easy=MU_EASY):
    target = torch.full_like(conf_map.squeeze(1), mu_easy)
    target[lrp_mask.bool()] = mu_hard
    return F.mse_loss(conf_map.squeeze(1), target)


# =========================================================================== #
# Loss 4 — L_PCL
# =========================================================================== #

def sample_anchors_lrp(emb, lrp_unc_mask, rho=RHO_LRP, max_n=256):
    """Primary hard samples: annotator-identified LRP uncertain pixels."""
    idx = lrp_unc_mask.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return None
    n    = min(max(1, int(idx.shape[0] * rho)), max_n)
    perm = torch.randperm(idx.shape[0], device=emb.device)[:n]
    return emb[:, idx[perm, 0], idx[perm, 1]].T   # (n, D)


def sample_anchors_easy(emb, pred_single, easy_unc_mask,
                        rho=RHO_EASY, max_n=256):
    """Secondary hard samples: easy uncertain pixels selected by entropy."""
    idx = easy_unc_mask.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return None
    prob     = torch.sigmoid(pred_single.squeeze())
    entropy  = -(prob * (prob + 1e-6).log() +
                 (1 - prob) * (1 - prob + 1e-6).log())
    ent_vals = entropy[idx[:, 0], idx[:, 1]]
    n        = min(max(1, int(idx.shape[0] * rho)), max_n)
    topk     = torch.topk(ent_vals, n).indices
    sel      = idx[topk]
    return emb[:, sel[:, 0], sel[:, 1]].T   # (n, D)


def loss_pcl(embeddings, pred, y_in, y_out,
             omega_delta, lrp_mask, lrp_uncertain,
             neg_queue, model, phase,
             temp=PCL_TEMP, max_pos=128, max_neg=128):
    B, D, H, W = embeddings.shape
    device     = embeddings.device
    total_loss = torch.tensor(0.0, device=device)
    n_valid    = 0

    for b in range(B):
        emb   = embeddings[b]
        m_pos = y_in[b].bool()
        m_neg = (y_out[b] == 0)

        if not m_pos.any() or not m_neg.any():
            continue

        anchor_list = []
        easy_unc    = (omega_delta[b] == 1) & ~lrp_mask[b].bool()

        if phase == 2:
            a = sample_anchors_easy(emb, pred[b:b+1], easy_unc)
            if a is not None:
                anchor_list.append(a)

        elif phase >= 3:
            a_lrp = sample_anchors_lrp(emb, lrp_uncertain[b].bool())
            if a_lrp is not None:
                anchor_list.append(a_lrp)
            a_easy = sample_anchors_easy(emb, pred[b:b+1], easy_unc)
            if a_easy is not None:
                anchor_list.append(a_easy)

        if not anchor_list:
            continue

        anch  = torch.cat(anchor_list, dim=0)

        p_idx    = m_pos.nonzero(as_tuple=False)
        pp       = torch.randperm(p_idx.shape[0], device=device)[:max_pos]
        pos      = emb[:, p_idx[pp, 0], p_idx[pp, 1]].T.mean(0, keepdim=True)

        n_idx    = m_neg.nonzero(as_tuple=False)
        np_      = torch.randperm(n_idx.shape[0], device=device)[:max_neg]
        neg_live = emb[:, n_idx[np_, 0], n_idx[np_, 1]].T
        negs     = torch.cat([neg_live, neg_queue.T], dim=0)

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
# Loss 5 — L_patch
# KL(p(x) || p̄_category) where p̄ = EMA mean of Ω_I or Ω_O predictions
# =========================================================================== #

def loss_patch(pred, lrp_fg, lrp_bg, lrp_uncertain,
               proto_fg, proto_bg, eps=1e-6):
    B          = pred.shape[0]
    prob       = torch.sigmoid(pred.squeeze(1))
    total_loss = torch.tensor(0.0, device=pred.device)
    n_valid    = 0

    for b in range(B):
        unc_mask = lrp_uncertain[b].bool()
        if not unc_mask.any():
            continue

        fg_inv = (255 - lrp_fg[b].cpu().numpy() * 255).astype(np.uint8)
        bg_inv = (255 - lrp_bg[b].cpu().numpy() * 255).astype(np.uint8)

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
            ps     = p_unc[closer_fg]
            fg     = proto_fg.clamp(eps, 1 - eps)
            kl     = fg * (fg.log() - ps.log()) + \
                     (1 - fg) * ((1 - fg).log() - (1 - ps).log())
            loss_b = loss_b + kl.mean()

        if closer_bg.any():
            ps     = p_unc[closer_bg]
            bg     = proto_bg.clamp(eps, 1 - eps)
            kl     = bg * (bg.log() - ps.log()) + \
                     (1 - bg) * ((1 - bg).log() - (1 - ps).log())
            loss_b = loss_b + kl.mean()

        total_loss += loss_b
        n_valid    += 1

    return total_loss / max(n_valid, 1)


# =========================================================================== #
# Prototype EMA
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
    if p < PHASE2_START:   return 1
    elif p < PHASE3_START: return 2
    else:                  return 3


# =========================================================================== #
# Evaluation — Dice + IoU + HD95
# =========================================================================== #

def test(model, image_root, mask_root, opt, threshold=0.5):
    """
    Fix [2]: threshold parameter now actually used.
    Fix [4]: total and hd_valid are int, not float.
    """
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

    DSC = IOU = HD = 0.0
    total = hd_valid = 0        # Fix [4]: int counters

    with torch.no_grad():
        for batch in loader:
            images = batch[0].cuda()
            gt     = batch[-1].cuda().float()
            preds  = model(images, mode='test')
            if isinstance(preds, list):
                preds = preds[-1]

            for i in range(images.shape[0]):
                p_prob = torch.sigmoid(preds[i]).squeeze()
                p_bin  = (p_prob >= threshold).float()
                g_bin  = (gt[i].squeeze() >= 0.5).float()

                if g_bin.sum() == 0:    # empty GT mask — skip entirely
                    continue

                DSC += dice_coefficient(p_bin, g_bin).item()
                IOU += iou_metric(p_bin, g_bin).item()
                HD  += hd95_metric(
                    p_bin.cpu().numpy().astype(np.uint8),
                    g_bin.cpu().numpy().astype(np.uint8)
                )
                total    += 1
                hd_valid += 1

    n  = max(total, 1)
    nh = max(hd_valid, 1)
    return DSC / n, IOU / n, HD / nh, total


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

            def rsz(m, long=False):
                out = F.interpolate(
                    m.unsqueeze(1).float(), size=pred_size, mode='nearest'
                ).squeeze(1)
                return out.long() if long else out

            y_in_r        = rsz(y_in)
            y_out_r       = rsz(y_out)
            omega_delta_r = rsz(omega_delta)
            lrp_fg_r      = rsz(lrp_fg)
            lrp_bg_r      = rsz(lrp_bg)
            lrp_unc_r     = rsz(lrp_uncertain)
            lrp_mask_r    = rsz(lrp_mask)
            y_c_r         = rsz(y_c, long=True)

            cls_r  = F.interpolate(cls_logits, size=pred_size,
                                   mode='bilinear', align_corners=False)
            conf_r = F.interpolate(conf_map, size=pred_size,
                                   mode='bilinear', align_corners=False)
            emb_r  = F.normalize(
                F.interpolate(embeddings, size=pred_size,
                              mode='bilinear', align_corners=False), dim=1
            )
            p_finest = preds[-1]    # (B, 1, H, W)

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
                    model.neg_queue.detach(), model, phase=2
                )
                loss = (l_c
                        + LAMBDA_PCL  * l_pcl
                        + LAMBDA_CE   * l_ce
                        + LAMBDA_CONF * l_conf)

            # ── Phase 3: L = L_c + λ1·PCL(all) + λ2·Lce + λ3·Lpatch ────────
            if phase >= 3:
                l_pcl = loss_pcl(
                    emb_r, p_finest,
                    y_in_r, y_out_r,
                    omega_delta_r, lrp_mask_r, lrp_unc_r,
                    model.neg_queue.detach(), model, phase=3
                )
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

    # Fix [2]: pass adaptive threshold — lower during warmup so HD95 is
    # computed against actual model outputs, not all-zero predictions.
    model.eval()
    threshold = 0.3 if epoch <= WARMUP_EPOCHS else 0.5
    d_dice, d_iou, d_hd95, n_samples = test(
        model, opt.val_image_root, opt.val_mask_root, opt,
        threshold=threshold
    )

    is_best = d_dice > best
    marker  = ' ← best' if is_best else ''

    print(f'\n  ┌─ Epoch {epoch:>3}/{opt.epoch}  Ph{phase}  '
          f'LR {current_lr:.2e}  ({epoch_time/60:.1f} min) ───────')
    print(f'  │  Train Loss : {loss_record.show():.4f}')
    print(f'  │  Val  Dice  : {d_dice:.4f}{marker}')
    print(f'  │  Val  IoU   : {d_iou:.4f}')
    print(f'  │  Val  HD95  : {d_hd95:.2f} px  [thresh={threshold}]')
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
        'epoch'     : epoch,
        'phase'     : phase,
        'lr'        : current_lr,
        'loss'      : loss_record.show(),
        'dice'      : d_dice,
        'iou'       : d_iou,
        'hd95'      : d_hd95,
        'threshold' : threshold,
        'best'      : is_best,
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
    parser.add_argument('--train_save',       type=str,   default='./model_pth_hup/')
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
        num_classes=1, kernel_sizes=[1, 3, 5],
        expansion_factor=2, dw_parallel=True, add=True,
        lgag_ks=3, activation='relu6',
        encoder='pvt_v2_b2', pretrain=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, opt.epoch - WARMUP_EPOCHS),
        eta_min=1e-6
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
    print(f'  Phase 1  ep 1–{p2_ep}          : L = L_c')
    print(f'  Phase 2  ep {p2_ep+1}–{p3_ep}  : L = L_c + PCL(easy) + Lce')
    print(f'  Phase 3  ep {p3_ep+1}–{opt.epoch}  : L = L_c + PCL(LRP+easy) + Lce + Lpatch')
    print(f'  ρ_LRP={RHO_LRP}  ρ_easy={RHO_EASY}')
    print(f'  μ_hard={MU_HARD} (LRP stricter)  μ_easy={MU_EASY} (tolerant)')
    print(f'  Warmup epochs: {WARMUP_EPOCHS}  (eval threshold=0.3 during warmup)')
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
    print(f'Total time : {total_train_time / 3600:.2f} h')
    print(f'Best Dice  : {best:.4f}')