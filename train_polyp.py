# train_polyp.py — MRPAnno version with original script structure restored

import os
import time
import torch
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
# Metric helpers — identical to original
# ─────────────────────────────────────────────────────────────

def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2. * intersection + smooth) / (total + smooth)

def iou(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def get_binary_metrics(pred, gt):
    tp = (pred * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision   = tp / (tp + fp + 1e-8)

    try:
        if pred.sum() > 0 and gt.sum() > 0:
            hd_val = hd95(pred.cpu().numpy(), gt.cpu().numpy())
        else:
            hd_val = 100.0
    except:
        hd_val = 100.0

    return sensitivity, specificity, precision, hd_val


# ─────────────────────────────────────────────────────────────
# MRPAnno loss functions
# ─────────────────────────────────────────────────────────────

def soft_dice_loss(pred_prob, soft_target, mask=None):
    smooth = 1e-6
    if mask is not None:
        pred_prob   = pred_prob[mask]
        soft_target = soft_target[mask]
    inter = (pred_prob * soft_target).sum()
    denom = pred_prob.sum() + soft_target.sum()
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)

def soft_bce_loss(pred_logit, soft_target, mask=None):
    loss = F.binary_cross_entropy_with_logits(
        pred_logit, soft_target, reduction='none'
    )
    if mask is not None:
        loss = loss[mask]
    return loss.mean()

def mrpanno_seg_loss(pred_logit, y_in, y_out,
                     omega_d1, omega_d2, softlabel,
                     alpha=ALPHA, beta=BETA):
    p_sig = torch.sigmoid(pred_logit)

    # Certain foreground Ω_I
    mask_in  = y_in.bool()
    tgt_in   = torch.ones_like(p_sig)
    loss_in  = (soft_dice_loss(p_sig, tgt_in, mask_in) +
                soft_bce_loss(pred_logit, tgt_in, mask_in))

    # Certain background Ω_O
    mask_out = (~y_out.bool())
    tgt_out  = torch.zeros_like(p_sig)
    loss_out = (soft_dice_loss(p_sig, tgt_out, mask_out) +
                soft_bce_loss(pred_logit, tgt_out, mask_out))

    # Inner uncertain Ω_Δ1
    mask_d1 = omega_d1.bool()
    loss_d1 = 0.0
    if mask_d1.any():
        tgt_d1  = softlabel.clamp(0.5, 1.0)
        loss_d1 = (soft_dice_loss(p_sig, tgt_d1, mask_d1) +
                   soft_bce_loss(pred_logit, tgt_d1, mask_d1))

    # Outer uncertain Ω_Δ2
    mask_d2 = omega_d2.bool()
    loss_d2 = 0.0
    if mask_d2.any():
        tgt_d2  = softlabel.clamp(0.0, 0.5)
        loss_d2 = (soft_dice_loss(p_sig, tgt_d2, mask_d2) +
                   soft_bce_loss(pred_logit, tgt_d2, mask_d2))

    return loss_in + loss_out + alpha * loss_d1 + beta * loss_d2

def boundary_consistency_loss(pred_logit, pmid_strip, omega_d1, omega_d2):
    p_sig = torch.sigmoid(pred_logit)
    strip = pmid_strip.bool()
    loss  = torch.tensor(0.0, device=pred_logit.device)

    if not strip.any():
        return loss

    p_s     = p_sig[strip].clamp(1e-6, 1 - 1e-6)
    entropy = -(p_s * p_s.log() + (1 - p_s) * (1 - p_s).log())
    loss    = loss + entropy.mean()

    d1_in_strip = strip & omega_d1.bool()
    d2_in_strip = strip & omega_d2.bool()
    if d1_in_strip.any() and d2_in_strip.any():
        margin_loss = F.relu(0.1 - (p_sig[d1_in_strip].mean() -
                                    p_sig[d2_in_strip].mean()))
        loss = loss + margin_loss

    return loss

def ccg5_loss(cls_logits, y_c5):
    return F.cross_entropy(cls_logits, y_c5)

def mrpanno_contrastive_loss(embeddings, pred_logit,
                              y_in, y_out,
                              omega_d1, omega_d2,
                              pmid_strip, neg_queue,
                              temperature=0.5,
                              max_anchors=512):
    B, D, H, W = embeddings.shape
    device = embeddings.device
    loss   = torch.tensor(0.0, device=device)
    count  = 0

    p_sig   = torch.sigmoid(pred_logit).detach()
    p_c     = p_sig.clamp(1e-6, 1 - 1e-6)
    entropy = -(p_c * p_c.log() + (1 - p_c) * (1 - p_c).log())

    for b in range(B):
        emb_b   = embeddings[b]
        strip_b = pmid_strip[b].bool()
        d1_b    = omega_d1[b].bool()
        d2_b    = omega_d2[b].bool()
        in_b    = y_in[b].bool()
        out_b   = (~y_out[b].bool())
        ent_b   = entropy[b]

        primary_mask   = strip_b
        uncertain_mask = (d1_b | d2_b) & (ent_b > 0.5) & (~strip_b)
        anchor_mask    = primary_mask | uncertain_mask

        if not anchor_mask.any():
            continue

        anchor_idx = anchor_mask.nonzero(as_tuple=False)
        if anchor_idx.shape[0] > max_anchors:
            perm       = torch.randperm(anchor_idx.shape[0],
                                        device=device)[:max_anchors]
            anchor_idx = anchor_idx[perm]

        pos_idx = in_b.nonzero(as_tuple=False)
        if pos_idx.shape[0] == 0:
            continue
        perm_p  = torch.randperm(pos_idx.shape[0], device=device)[:128]
        pos_idx = pos_idx[perm_p]
        pos_emb = emb_b[:, pos_idx[:, 0], pos_idx[:, 1]].T
        pos_mean = F.normalize(pos_emb.mean(0, keepdim=True), dim=1)

        neg_idx = out_b.nonzero(as_tuple=False)
        neg_list = []
        if neg_idx.shape[0] > 0:
            perm_n  = torch.randperm(neg_idx.shape[0], device=device)[:128]
            neg_list.append(emb_b[:, neg_idx[perm_n, 0],
                                     neg_idx[perm_n, 1]].T)
        neg_list.append(neg_queue.T)
        neg_emb = torch.cat(neg_list, dim=0)

        anc_emb = emb_b[:, anchor_idx[:, 0], anchor_idx[:, 1]].T
        sim_pos = (anc_emb * pos_mean).sum(dim=1, keepdim=True) / temperature
        sim_neg = (anc_emb @ neg_emb.T) / temperature
        logits  = torch.cat([sim_pos, sim_neg], dim=1)
        labels  = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        loss  = loss + F.cross_entropy(logits, labels)
        count += 1

    return loss / max(count, 1)


# ─────────────────────────────────────────────────────────────
# Test function — identical structure to original
# ─────────────────────────────────────────────────────────────

def test(model, path, dataset, opt, save_base=None):
    data_path  = os.path.join(path, dataset)
    image_root = os.path.join(data_path, 'images') + '/'
    gt_root    = os.path.join(data_path, 'masks')  + '/'
    model.eval()

    test_loader = get_loader(
        image_root=image_root,
        gt_root=gt_root,
        batchsize=1,
        trainsize=opt.img_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        augmentation=False,
        split='test',
    )

    DSC, IOU, total_images = 0.0, 0.0, 0
    detailed_results = []

    with torch.no_grad():
        for pack in tqdm(test_loader, desc=f'Testing on {dataset}'):
            # unpack — gt is last item, never used for loss
            (images, y_in, y_mid, y_out,
             omega_d1, omega_d2, softlabel,
             pmid_strip, y_c5, gt) = pack

            images = images.cuda()
            gt     = gt.cuda().squeeze(1).float()

            preds = model(images, mode='test')
            pred  = torch.sigmoid(preds[-1]).squeeze(1)   # (B,H,W)

            for i in range(images.shape[0]):
                pred_i = pred[i]
                gt_i   = gt[i]

                pred_i = (pred_i - pred_i.min()) / \
                         (pred_i.max() - pred_i.min() + 1e-8)

                pred_bin = (pred_i >= 0.5).float()
                gt_bin   = (gt_i  >= 0.5).float()

                d  = dice_coefficient(pred_bin, gt_bin).item()
                io = iou(pred_bin, gt_bin).item()
                sens, spec, prec, hd = get_binary_metrics(pred_bin, gt_bin)

                DSC += d
                IOU += io
                total_images += 1

                name = f'image_{total_images}.png'
                detailed_results.append({
                    'Name': name, 'Dice': d, 'IoU': io,
                    'Sensitivity': round(sens, 4),
                    'Specificity': round(spec, 4),
                    'Precision':   round(prec, 4),
                    'HD95':        round(hd,   4),
                })

                if save_base:
                    pred_img = (pred_bin.cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_base, name), pred_img)

    return DSC / total_images, IOU / total_images, detailed_results


# ─────────────────────────────────────────────────────────────
# Main training loop — same structure as original
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ── All original arguments preserved exactly ──────────────
    parser.add_argument('--run_id',            type=str,   required=True)
    parser.add_argument('--encoder',           type=str,   default='pvt_v2_b2')
    parser.add_argument('--expansion_factor',  type=int,   default=2)
    parser.add_argument('--kernel_sizes',      type=int,   nargs='+', default=[1, 3, 5])
    parser.add_argument('--lgag_ks',           type=int,   default=3)
    parser.add_argument('--activation_mscb',   type=str,   default='relu6')
    parser.add_argument('--no_dw_parallel',    action='store_true', default=False)
    parser.add_argument('--concatenation',     action='store_true', default=False)
    parser.add_argument('--img_size',          type=int,   default=352)
    parser.add_argument('--batchsize',         type=int,   default=8)
    parser.add_argument('--epochs',            type=int,   default=100)
    parser.add_argument('--lr',                type=float, default=1e-4)
    parser.add_argument('--num_workers',       type=int,   default=4)
    parser.add_argument('--aug',               type=bool,  default=True)
    parser.add_argument('--pretrain',          type=bool,  default=True)
    parser.add_argument('--pretrained_dir',    type=str,   default='./pretrained_pth/pvt/')
    parser.add_argument('--color_image',       default=True)
    # ── Paths — original style ────────────────────────────────
    parser.add_argument('--train_save',        type=str,   default='./model_pth/')
    parser.add_argument('--train_path',        type=str,   default='./data/polyp/TrainDataset/')
    parser.add_argument('--val_path',          type=str,   default='./data/polyp/ValDataset/')
    parser.add_argument('--test_path',         type=str,   default='./data/polyp/TestDataset/')
    parser.add_argument('--dataset_name',      type=str,   default='Kvasir')
    # ── MRPAnno loss weights ──────────────────────────────────
    parser.add_argument('--lambda1',           type=float, default=0.2,  help='contrastive loss weight')
    parser.add_argument('--lambda2',           type=float, default=0.4,  help='CCG-5 loss weight')
    parser.add_argument('--lambda3',           type=float, default=0.2,  help='boundary loss weight')
    parser.add_argument('--temperature',       type=float, default=0.5)
    parser.add_argument('--max_anchors',       type=int,   default=512)
    opt = parser.parse_args()

    # ── Paths ─────────────────────────────────────────────────
    save_path = os.path.join(opt.train_save, opt.run_id)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('results_polyp', exist_ok=True)

    # ── Model ─────────────────────────────────────────────────
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
    ).cuda()

    print('Total model params: %.2fM' % (
        sum(p.numel() for p in model.parameters()) / 1e6
    ))

    # ── Data loaders ──────────────────────────────────────────
    image_root = os.path.join(opt.train_path, 'images') + '/'
    gt_root    = os.path.join(opt.train_path, 'masks')  + '/'
    val_img    = os.path.join(opt.val_path,   'images') + '/'
    val_gt     = os.path.join(opt.val_path,   'masks')  + '/'

    train_loader = get_loader(
        image_root=image_root,
        gt_root=gt_root,
        batchsize=opt.batchsize,
        trainsize=opt.img_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        augmentation=opt.aug,
        split='train',
    )
    val_loader = get_loader(
        image_root=val_img,
        gt_root=val_gt,
        batchsize=opt.batchsize,
        trainsize=opt.img_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        augmentation=False,
        split='val',
    )

    total_step = len(train_loader)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Val samples:      {len(val_loader.dataset)}')

    # ── Optimiser — differential lr (backbone vs rest) ────────
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(),   'lr': opt.lr * 0.1},
        {'params': model.decoder.parameters(),    'lr': opt.lr},
        {'params': model.cls_head.parameters(),   'lr': opt.lr},
        {'params': model.embed_head.parameters(), 'lr': opt.lr},
        {'params': model.out_head1.parameters(),  'lr': opt.lr},
        {'params': model.out_head2.parameters(),  'lr': opt.lr},
        {'params': model.out_head3.parameters(),  'lr': opt.lr},
        {'params': model.out_head4.parameters(),  'lr': opt.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────
    best_dice = 0.0
    log_rows  = []

    for epoch in range(1, opt.epochs + 1):
        model.train()

        # Warm-up schedules
        lam1_eff = opt.lambda1 * min(1.0, (epoch / opt.epochs) * 2.0)
        lam3_eff = opt.lambda3 * min(1.0, epoch / 30.0)

        loss_total   = 0.0
        loss_seg_t   = 0.0
        loss_pcl_t   = 0.0
        loss_ce5_t   = 0.0
        loss_bound_t = 0.0

        for i, pack in enumerate(train_loader, start=1):
            (images, y_in, y_mid, y_out,
             omega_d1, omega_d2, softlabel,
             pmid_strip, y_c5, gt) = pack

            # gt is NEVER used in any loss below
            images     = images.cuda()
            y_in       = y_in.cuda()
            y_mid      = y_mid.cuda()
            y_out      = y_out.cuda()
            omega_d1   = omega_d1.cuda()
            omega_d2   = omega_d2.cuda()
            softlabel  = softlabel.cuda()
            pmid_strip = pmid_strip.cuda()
            y_c5       = y_c5.cuda()

            # Forward
            preds, cls_logits, embeddings = model(images, mode='train')
            pred_logit = preds[-1].squeeze(1)   # finest head (B,H,W)

            # L_seg — graduated multi-zone Dice+BCE
            loss_seg = mrpanno_seg_loss(
                pred_logit, y_in, y_out,
                omega_d1, omega_d2, softlabel,
            )
            # Auxiliary heads
            for aux in preds[:-1]:
                loss_seg = loss_seg + 0.4 * mrpanno_seg_loss(
                    aux.squeeze(1), y_in, y_out,
                    omega_d1, omega_d2, softlabel,
                )

            # L_ce5 — 5-class CCG classification
            loss_ce5 = ccg5_loss(cls_logits, y_c5)

            # L_bound — boundary consistency (warm-up)
            loss_bound = boundary_consistency_loss(
                pred_logit, pmid_strip, omega_d1, omega_d2
            )

            # L_pcl — pixel contrastive (warm-up)
            loss_pcl = mrpanno_contrastive_loss(
                embeddings, pred_logit,
                y_in, y_out,
                omega_d1, omega_d2,
                pmid_strip,
                model.neg_queue,
                temperature=opt.temperature,
                max_anchors=opt.max_anchors,
            )

            # Update negative queue with background embeddings
            with torch.no_grad():
                bg_mask = (~y_out.bool())
                for b in range(images.shape[0]):
                    idx = bg_mask[b].nonzero(as_tuple=False)
                    if idx.shape[0] > 0:
                        perm = torch.randperm(idx.shape[0])[:32]
                        vecs = embeddings[b, :,
                                          idx[perm, 0],
                                          idx[perm, 1]].T
                        model.update_queue(vecs)

            # Total loss
            loss = (loss_seg
                    + lam1_eff * loss_pcl
                    + opt.lambda2 * loss_ce5
                    + lam3_eff  * loss_bound)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_total   += loss.item()
            loss_seg_t   += loss_seg.item()
            loss_pcl_t   += (loss_pcl.item()
                             if isinstance(loss_pcl, torch.Tensor)
                             else loss_pcl)
            loss_ce5_t   += loss_ce5.item()
            loss_bound_t += (loss_bound.item()
                             if isinstance(loss_bound, torch.Tensor)
                             else loss_bound)

            # Print every 20 steps — same style as original
            if i % 20 == 0 or i == total_step:
                print(
                    f'Epoch [{epoch:03d}/{opt.epochs}] '
                    f'Step [{i:04d}/{total_step}] '
                    f'Loss: {loss.item():.4f} '
                    f'[seg={loss_seg.item():.3f} '
                    f'pcl={loss_pcl.item() if isinstance(loss_pcl,torch.Tensor) else loss_pcl:.3f} '
                    f'ce5={loss_ce5.item():.3f} '
                    f'bnd={loss_bound.item() if isinstance(loss_bound,torch.Tensor) else loss_bound:.3f}] '
                    f'λ1={lam1_eff:.3f} λ3={lam3_eff:.3f}'
                )

        scheduler.step()

        # ── Validation ────────────────────────────────────────
        model.eval()
        val_dice_sum, val_iou_sum, val_n = 0.0, 0.0, 0
        with torch.no_grad():
            for pack in val_loader:
                (images, y_in, y_mid, y_out,
                 omega_d1, omega_d2, softlabel,
                 pmid_strip, y_c5, gt) = pack

                images = images.cuda()
                gt     = gt.cuda().squeeze(1).float()

                preds    = model(images, mode='test')
                pred     = torch.sigmoid(preds[-1]).squeeze(1)
                pred_bin = (pred >= 0.5).float()
                gt_bin   = (gt   >= 0.5).float()

                for b in range(pred_bin.shape[0]):
                    val_dice_sum += dice_coefficient(
                        pred_bin[b], gt_bin[b]).item()
                    val_iou_sum  += iou(
                        pred_bin[b], gt_bin[b]).item()
                    val_n += 1

        val_dice = val_dice_sum / max(val_n, 1)
        val_iou  = val_iou_sum  / max(val_n, 1)
        model.train()

        n_steps = len(train_loader)
        print(
            f'\n>>> Epoch {epoch:03d} Summary | '
            f'avg_loss={loss_total/n_steps:.4f} | '
            f'val_dice={val_dice:.4f} | '
            f'val_iou={val_iou:.4f}\n'
        )

        log_rows.append({
            'epoch':      epoch,
            'loss':       loss_total   / n_steps,
            'loss_seg':   loss_seg_t   / n_steps,
            'loss_pcl':   loss_pcl_t   / n_steps,
            'loss_ce5':   loss_ce5_t   / n_steps,
            'loss_bound': loss_bound_t / n_steps,
            'val_dice':   val_dice,
            'val_iou':    val_iou,
        })

        # ── Save best model ───────────────────────────────────
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                model.state_dict(),
                os.path.join(save_path, f'{opt.run_id}-best.pth')
            )
            print(f'  ✓ New best saved  (val_dice={best_dice:.4f})\n')

        # Latest checkpoint every epoch
        torch.save(
            model.state_dict(),
            os.path.join(save_path, f'{opt.run_id}-latest.pth')
        )

    # ── Training log ──────────────────────────────────────────
    pd.DataFrame(log_rows).to_excel(
        f'results_polyp/TrainLog_{opt.run_id}.xlsx', index=False
    )
    print(f'\nTraining complete. Best val Dice = {best_dice:.4f}')
    print(f'Model saved to: {save_path}/{opt.run_id}-best.pth')