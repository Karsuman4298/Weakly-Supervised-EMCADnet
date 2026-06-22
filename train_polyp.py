# train_polyp.py  — MRPAnno version
#
# Supervision comes ONLY from polygon-derived masks.
# GT mask is NEVER passed to any loss function.
#
# Loss breakdown
# ──────────────
# L_seg    : weighted multi-zone Dice+BCE on {Ω_I, Ω_Δ1, Ω_Δ2, Ω_O}
#            using soft labels for the two uncertain bands
# L_bound  : KL divergence across P_mid strip (boundary consistency)
# L_ce5    : cross-entropy for 5-class CCG head
# L_pcl    : pixel contrastive learning (MRPAnno hard-sample selection)
#
# L = L_seg + λ1*L_pcl + λ2*L_ce5 + λ3*L_bound

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import argparse
from tqdm import tqdm

from lib.networks import EMCADNet
from utils.dataloader import get_loader
from medpy.metric.binary import hd95
ALPHA = 0.70   # Ω_Δ1 (inner band, likely FG) pseudo-label
BETA  = 0.30


# ─────────────────────────────────────────────────────────────
# Metric helpers  (unchanged from original)
# ─────────────────────────────────────────────────────────────

def dice_coefficient(pred, label):
    if pred.device != label.device:
        label = label.to(pred.device)
    smooth = 1e-6
    p = pred.contiguous().view(-1)
    l = label.contiguous().view(-1)
    return (2. * (p * l).sum() + smooth) / (p.sum() + l.sum() + smooth)


def iou_score(pred, label):
    if pred.device != label.device:
        label = label.to(pred.device)
    smooth = 1e-6
    p = pred.contiguous().view(-1)
    l = label.contiguous().view(-1)
    inter = (p * l).sum()
    union = p.sum() + l.sum() - inter
    return (inter + smooth) / (union + smooth)


def get_binary_metrics(pred, gt):
    tp = (pred * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    try:
        hd_val = hd95(pred.cpu().numpy(), gt.cpu().numpy()) \
            if pred.sum() > 0 and gt.sum() > 0 else 100.0
    except Exception:
        hd_val = 100.0
    return sens, spec, prec, hd_val


# ─────────────────────────────────────────────────────────────
# Soft Dice loss  (supports soft float targets)
# ─────────────────────────────────────────────────────────────

def soft_dice_loss(pred_prob, soft_target, mask=None):
    """
    pred_prob  : (B,H,W) sigmoid probability
    soft_target: (B,H,W) float target in [0,1]
    mask       : (B,H,W) bool — pixels to include (None = all)
    """
    smooth = 1e-6
    if mask is not None:
        pred_prob   = pred_prob[mask]
        soft_target = soft_target[mask]
    inter = (pred_prob * soft_target).sum()
    denom = pred_prob.sum() + soft_target.sum()
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)


def soft_bce_loss(pred_logit, soft_target, mask=None):
    """
    pred_logit : (B,H,W) raw logit
    soft_target: (B,H,W) float target in [0,1]
    """
    loss = F.binary_cross_entropy_with_logits(
        pred_logit, soft_target, reduction='none'
    )
    if mask is not None:
        loss = loss[mask]
    return loss.mean()


# ─────────────────────────────────────────────────────────────
# MRPAnno segmentation loss  L_seg
# ─────────────────────────────────────────────────────────────

def mrpanno_seg_loss(pred_logit, y_in, y_out,
                     omega_d1, omega_d2, softlabel,
                     alpha=ALPHA, beta=BETA):
    """
    Weighted multi-zone Dice + BCE.

    Zones and their targets
    ───────────────────────
    Ω_I  (y_in==1)           → target 1.0,  weight 1.0
    Ω_O  (y_out==0)          → target 0.0,  weight 1.0
    Ω_Δ1 (omega_d1==1)       → target softlabel (≈0.7 avg), weight α
    Ω_Δ2 (omega_d2==1)       → target softlabel (≈0.3 avg), weight β

    GT is NEVER used here. All targets come from polygon masks.
    """
    p_sig  = torch.sigmoid(pred_logit)   # (B,H,W)

    # ── Certain foreground: Ω_I ───────────────────────────────
    mask_in  = y_in.bool()
    tgt_in   = torch.ones_like(p_sig)
    loss_in  = (soft_dice_loss(p_sig, tgt_in, mask_in) +
                soft_bce_loss(pred_logit, tgt_in, mask_in))

    # ── Certain background: Ω_O ───────────────────────────────
    mask_out = (~y_out.bool())
    tgt_out  = torch.zeros_like(p_sig)
    loss_out = (soft_dice_loss(p_sig, tgt_out, mask_out) +
                soft_bce_loss(pred_logit, tgt_out, mask_out))

    # ── Inner uncertain: Ω_Δ1 ────────────────────────────────
    mask_d1  = omega_d1.bool()
    loss_d1  = 0.0
    if mask_d1.any():
        tgt_d1  = softlabel.clamp(0.5, 1.0)   # Ω_Δ1 is likely FG
        loss_d1 = (soft_dice_loss(p_sig, tgt_d1, mask_d1) +
                   soft_bce_loss(pred_logit, tgt_d1, mask_d1))

    # ── Outer uncertain: Ω_Δ2 ────────────────────────────────
    mask_d2  = omega_d2.bool()
    loss_d2  = 0.0
    if mask_d2.any():
        tgt_d2  = softlabel.clamp(0.0, 0.5)   # Ω_Δ2 is likely BG
        loss_d2 = (soft_dice_loss(p_sig, tgt_d2, mask_d2) +
                   soft_bce_loss(pred_logit, tgt_d2, mask_d2))

    return loss_in + loss_out + alpha * loss_d1 + beta * loss_d2


# ─────────────────────────────────────────────────────────────
# Boundary consistency loss  L_bound
# ─────────────────────────────────────────────────────────────

def boundary_consistency_loss(pred_logit, pmid_strip,
                              omega_d1, omega_d2):
    """
    For every pixel in the P_mid strip, push the prediction to
    be more confident than its neighbours across the strip.

    Implementation: minimise prediction entropy inside the strip
    (the model should be either foreground or background — not 0.5).
    Additionally enforce that Ω_Δ1 side > Ω_Δ2 side via a margin loss.
    """
    p_sig  = torch.sigmoid(pred_logit)   # (B,H,W)
    strip  = pmid_strip.bool()

    loss = torch.tensor(0.0, device=pred_logit.device)

    if not strip.any():
        return loss

    # ── Entropy minimisation inside strip ────────────────────
    p_s   = p_sig[strip].clamp(1e-6, 1 - 1e-6)
    entropy = -(p_s * p_s.log() + (1 - p_s) * (1 - p_s).log())
    loss  = loss + entropy.mean()

    # ── Margin: Ω_Δ1 predictions > Ω_Δ2 predictions ─────────
    # (inner band should be more foreground than outer band)
    d1_in_strip = strip & omega_d1.bool()
    d2_in_strip = strip & omega_d2.bool()
    if d1_in_strip.any() and d2_in_strip.any():
        p_d1_mean = p_sig[d1_in_strip].mean()
        p_d2_mean = p_sig[d2_in_strip].mean()
        margin_loss = F.relu(0.1 - (p_d1_mean - p_d2_mean))
        loss = loss + margin_loss

    return loss


# ─────────────────────────────────────────────────────────────
# 5-class CCG loss  L_ce5
# ─────────────────────────────────────────────────────────────

def ccg5_loss(cls_logits, y_c5):
    """
    Standard cross-entropy for the 5-class CCG head.
    cls_logits : (B, 5, H, W)
    y_c5       : (B, H, W) int64  with values {0,1,2,3,4}
    """
    return F.cross_entropy(cls_logits, y_c5)


# ─────────────────────────────────────────────────────────────
# Pixel contrastive loss  L_pcl  (MRPAnno hard-sample selection)
# ─────────────────────────────────────────────────────────────

def mrpanno_contrastive_loss(embeddings, pred_logit,
                              y_in, y_out,
                              omega_d1, omega_d2,
                              pmid_strip, neg_queue,
                              temperature=0.5,
                              max_anchors=512,
                              pmid_dist_weight=True):
    """
    Pixel contrastive learning with MRPAnno hard-sample selection.

    Anchor selection (hard samples)
    ────────────────────────────────
    Primary   : pixels in pmid_strip (annotator-identified boundary)
    Secondary : uncertain pixels (Ω_Δ1 ∪ Ω_Δ2) with high entropy

    Positives : pixels in Ω_I  (certain foreground embeddings)
    Negatives : pixels in Ω_O  (certain background) + memory queue

    embeddings : (B, D, H, W)  L2-normalised
    pred_logit : (B,    H, W)  raw logit for entropy computation
    neg_queue  : (D, Q)        L2-normalised memory queue
    """
    B, D, H, W = embeddings.shape
    device = embeddings.device
    loss   = torch.tensor(0.0, device=device)
    count  = 0

    p_sig  = torch.sigmoid(pred_logit).detach()   # (B,H,W)
    entropy = -(p_sig.clamp(1e-6, 1-1e-6) * p_sig.clamp(1e-6, 1-1e-6).log() +
                (1-p_sig).clamp(1e-6, 1-1e-6) * (1-p_sig).clamp(1e-6, 1-1e-6).log())

    for b in range(B):
        emb_b   = embeddings[b]           # (D, H, W)
        strip_b = pmid_strip[b].bool()    # (H, W)
        d1_b    = omega_d1[b].bool()
        d2_b    = omega_d2[b].bool()
        in_b    = y_in[b].bool()
        out_b   = (~y_out[b].bool())
        ent_b   = entropy[b]              # (H, W)

        # ── Anchor set ────────────────────────────────────────
        # Primary: P_mid strip pixels (always included)
        primary_mask = strip_b

        # Secondary: uncertain pixels with entropy > 0.5
        uncertain_mask = (d1_b | d2_b) & (ent_b > 0.5) & (~strip_b)

        anchor_mask = primary_mask | uncertain_mask

        if not anchor_mask.any():
            continue

        anchor_idx = anchor_mask.nonzero(as_tuple=False)  # (N,2)
        if anchor_idx.shape[0] > max_anchors:
            perm       = torch.randperm(anchor_idx.shape[0], device=device)[:max_anchors]
            anchor_idx = anchor_idx[perm]

        # ── Positive set: Ω_I pixels ─────────────────────────
        pos_idx = in_b.nonzero(as_tuple=False)
        if pos_idx.shape[0] == 0:
            continue
        perm_p  = torch.randperm(pos_idx.shape[0], device=device)[:128]
        pos_idx = pos_idx[perm_p]
        pos_emb = emb_b[:, pos_idx[:, 0], pos_idx[:, 1]].T   # (P,D)
        pos_mean = F.normalize(pos_emb.mean(0, keepdim=True), dim=1)  # (1,D)

        # ── Negative set: Ω_O + memory queue ─────────────────
        neg_idx = out_b.nonzero(as_tuple=False)
        neg_embs_list = []
        if neg_idx.shape[0] > 0:
            perm_n  = torch.randperm(neg_idx.shape[0], device=device)[:128]
            neg_idx = neg_idx[perm_n]
            neg_embs_list.append(emb_b[:, neg_idx[:, 0], neg_idx[:, 1]].T)  # (N,D)
        neg_embs_list.append(neg_queue.T)                    # (Q,D)
        neg_emb = torch.cat(neg_embs_list, dim=0)            # (N+Q, D)

        # ── InfoNCE per anchor ────────────────────────────────
        anc_emb = emb_b[:, anchor_idx[:, 0], anchor_idx[:, 1]].T  # (A,D)

        sim_pos = (anc_emb * pos_mean).sum(dim=1, keepdim=True) / temperature  # (A,1)
        sim_neg = (anc_emb @ neg_emb.T) / temperature                          # (A,N+Q)

        logits  = torch.cat([sim_pos, sim_neg], dim=1)       # (A, 1+N+Q)
        labels  = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
        loss    = loss + F.cross_entropy(logits, labels)
        count  += 1

    return loss / max(count, 1)


# ─────────────────────────────────────────────────────────────
# Validation loop  (GT used for metrics only — not for loss)
# ─────────────────────────────────────────────────────────────

def validate(model, val_loader, device, epoch):
    model.eval()
    dice_sum, iou_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for pack in val_loader:
            (images, y_in, y_mid, y_out,
             omega_d1, omega_d2, softlabel,
             pmid_strip, y_c5, gt) = pack

            images = images.to(device)
            gt     = gt.to(device).squeeze(1).float()

            preds = model(images, mode='test')
            pred  = torch.sigmoid(preds[-1]).squeeze(1)  # (B,H,W)
            pred_bin = (pred >= 0.5).float()
            gt_bin   = (gt   >= 0.5).float()

            for b in range(pred_bin.shape[0]):
                dice_sum += dice_coefficient(pred_bin[b], gt_bin[b]).item()
                iou_sum  += iou_score(pred_bin[b], gt_bin[b]).item()
                n        += 1

    model.train()
    return dice_sum / max(n, 1), iou_sum / max(n, 1)


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(opt):
    os.makedirs(f'./model_pth/{opt.run_id}', exist_ok=True)
    os.makedirs('results_polyp', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Data ─────────────────────────────────────────────────
    train_loader = get_loader(
        image_root=opt.train_path + 'images/',
        gt_root=opt.train_path + 'masks/',
        batchsize=opt.batchsize,
        trainsize=opt.img_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        augmentation=opt.aug,
        split='train',
    )
    val_loader = get_loader(
        image_root=opt.val_path + 'images/',
        gt_root=opt.val_path + 'masks/',
        batchsize=opt.batchsize,
        trainsize=opt.img_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        augmentation=False,
        split='val',
    )

    # ── Model ────────────────────────────────────────────────
    model = EMCADNet(
        num_classes=1,
        kernel_sizes=opt.kernel_sizes,
        expansion_factor=opt.expansion_factor,
        dw_parallel=not opt.no_dw_parallel,
        add=not opt.concatenation,
        lgag_ks=opt.lgag_ks,
        activation=opt.activation_mscb,
        encoder=opt.encoder,
        pretrain=opt.pretrain,
        pretrained_dir=opt.pretrained_dir,
    ).to(device)

    # ── Optimiser & Scheduler ─────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=opt.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.epochs, eta_min=1e-6
    )

    # ── Loss weights ─────────────────────────────────────────
    lam1 = opt.lambda1   # L_pcl
    lam2 = opt.lambda2   # L_ce5
    lam3 = opt.lambda3   # L_bound

    best_dice  = 0.0
    log_rows   = []

    for epoch in range(1, opt.epochs + 1):
        model.train()

        # λ3 warm-up: ramp boundary loss from 0 → lam3 over first 30 epochs
        lam3_eff = lam3 * min(1.0, epoch / 30.0)

        epoch_loss = 0.0
        t0 = time.time()

        for step, pack in enumerate(tqdm(train_loader,
                                         desc=f'Epoch {epoch}/{opt.epochs}',
                                         leave=False)):
            (images, y_in, y_mid, y_out,
             omega_d1, omega_d2, softlabel,
             pmid_strip, y_c5, gt) = pack

            # ── Move to device ────────────────────────────────
            images     = images.to(device)
            y_in       = y_in.to(device)
            y_mid      = y_mid.to(device)
            y_out      = y_out.to(device)
            omega_d1   = omega_d1.to(device)
            omega_d2   = omega_d2.to(device)
            softlabel  = softlabel.to(device)
            pmid_strip = pmid_strip.to(device)
            y_c5       = y_c5.to(device)
            # gt stays on CPU — never used in loss

            # ── Forward pass ──────────────────────────────────
            preds, cls_logits, embeddings = model(images, mode='train')
            # preds[-1] = finest prediction (p1), shape (B,1,H,W)
            pred_logit = preds[-1].squeeze(1)   # (B,H,W)

            # ── L_seg: multi-zone weighted Dice+BCE ───────────
            loss_seg = mrpanno_seg_loss(
                pred_logit, y_in, y_out,
                omega_d1, omega_d2, softlabel,
                alpha=ALPHA, beta=BETA
            )

            # Auxiliary heads (same zone loss, lower weight)
            for aux_pred in preds[:-1]:
                aux_logit = aux_pred.squeeze(1)
                loss_seg  = loss_seg + 0.4 * mrpanno_seg_loss(
                    aux_logit, y_in, y_out,
                    omega_d1, omega_d2, softlabel
                )

            # ── L_ce5: 5-class CCG classification ─────────────
            loss_ce5 = ccg5_loss(cls_logits, y_c5)

            # ── L_bound: boundary consistency ─────────────────
            loss_bound = boundary_consistency_loss(
                pred_logit, pmid_strip, omega_d1, omega_d2
            )

            # ── L_pcl: contrastive (with MRPAnno hard sampling) 
            loss_pcl = mrpanno_contrastive_loss(
                embeddings, pred_logit,
                y_in, y_out,
                omega_d1, omega_d2,
                pmid_strip,
                model.neg_queue,
                temperature=opt.temperature,
                max_anchors=opt.max_anchors,
            )

            # ── Update neg queue with current Ω_O embeddings ──
            with torch.no_grad():
                bg_mask = (~y_out.bool())   # (B,H,W)
                for b in range(images.shape[0]):
                    idx = bg_mask[b].nonzero(as_tuple=False)
                    if idx.shape[0] > 0:
                        perm = torch.randperm(idx.shape[0])[:32]
                        vecs = embeddings[b, :,
                                          idx[perm, 0],
                                          idx[perm, 1]].T     # (32,D)
                        model.update_queue(vecs)

            # ── Total loss ────────────────────────────────────
            loss = loss_seg + lam1 * loss_pcl + lam2 * loss_ce5 + lam3_eff * loss_bound

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # ── Validation ───────────────────────────────────────
        val_dice, val_iou = validate(model, val_loader, device, epoch)
        elapsed = time.time() - t0

        print(f'Epoch {epoch:03d} | loss {avg_loss:.4f} | '
              f'val_dice {val_dice:.4f} | val_iou {val_iou:.4f} | '
              f'{elapsed:.1f}s | λ3_eff={lam3_eff:.3f}')

        log_rows.append({
            'epoch': epoch, 'loss': avg_loss,
            'val_dice': val_dice, 'val_iou': val_iou
        })

        # ── Save best model ───────────────────────────────────
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(),
                       f'./model_pth/{opt.run_id}/{opt.run_id}-best.pth')
            print(f'  ✓ New best saved  (dice={best_dice:.4f})')

        # ── Save latest checkpoint ────────────────────────────
        torch.save(model.state_dict(),
                   f'./model_pth/{opt.run_id}/{opt.run_id}-latest.pth')

    # ── Training log ─────────────────────────────────────────
    pd.DataFrame(log_rows).to_excel(
        f'results_polyp/TrainLog_{opt.run_id}.xlsx', index=False
    )
    print(f'\nTraining complete. Best val Dice = {best_dice:.4f}')


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id',           type=str,   required=True)
    parser.add_argument('--encoder',          type=str,   default='pvt_v2_b2')
    parser.add_argument('--expansion_factor', type=int,   default=2)
    parser.add_argument('--kernel_sizes',     type=int,   nargs='+', default=[1, 3, 5])
    parser.add_argument('--lgag_ks',          type=int,   default=3)
    parser.add_argument('--activation_mscb',  type=str,   default='relu6')
    parser.add_argument('--no_dw_parallel',   action='store_true', default=False)
    parser.add_argument('--concatenation',    action='store_true', default=False)
    parser.add_argument('--img_size',         type=int,   default=352)
    parser.add_argument('--batchsize',        type=int,   default=8)
    parser.add_argument('--epochs',           type=int,   default=100)
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--num_workers',      type=int,   default=4)
    parser.add_argument('--aug',              type=bool,  default=True)
    parser.add_argument('--pretrain',         type=bool,  default=True)
    parser.add_argument('--pretrained_dir',   type=str,   default='./pretrained_pth/pvt/')
    parser.add_argument('--train_path',       type=str,   default='./data/polyp/TrainDataset/')
    parser.add_argument('--val_path',         type=str,   default='./data/polyp/ValDataset/')
    # Loss weights
    parser.add_argument('--lambda1',          type=float, default=0.4,  help='L_pcl weight')
    parser.add_argument('--lambda2',          type=float, default=1.0,  help='L_ce5 weight')
    parser.add_argument('--lambda3',          type=float, default=0.3,  help='L_bound weight')
    # Contrastive
    parser.add_argument('--temperature',      type=float, default=0.5)
    parser.add_argument('--max_anchors',      type=int,   default=512)
    opt = parser.parse_args()

    train(opt)