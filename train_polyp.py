# train_polyp.py — MRPAnno with PRC + EMA (fixed) + step warm-up

import os
import copy
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


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    p = predicted.contiguous().view(-1)
    l = labels.contiguous().view(-1)
    return (2. * (p * l).sum() + smooth) / (p.sum() + l.sum() + smooth)

def iou(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    p = predicted.contiguous().view(-1)
    l = labels.contiguous().view(-1)
    inter = (p * l).sum()
    union = p.sum() + l.sum() - inter
    return (inter + smooth) / (union + smooth)

def get_binary_metrics(pred, gt):
    tp = (pred * gt).sum().item()
    tn = ((1-pred)*(1-gt)).sum().item()
    fp = (pred*(1-gt)).sum().item()
    fn = ((1-pred)*gt).sum().item()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    try:
        hd_val = hd95(pred.cpu().numpy(), gt.cpu().numpy()) \
            if pred.sum() > 0 and gt.sum() > 0 else 100.0
    except:
        hd_val = 100.0
    return sens, spec, prec, hd_val


# ─────────────────────────────────────────────────────────────
# EMA — with warm-up aware decay
# ─────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model weights.
    Uses lower decay early in training so shadow tracks
    the model quickly during the first few epochs.
    """
    def __init__(self, model, decay=0.999, warmup_steps=500):
        self.decay        = decay
        self.warmup_steps = warmup_steps
        self.step_count   = 0
        self.shadow       = copy.deepcopy(model.state_dict())

    def _get_decay(self):
        # Ramp decay from 0.0 → target over warmup_steps
        # This means shadow tracks model closely early on
        if self.step_count < self.warmup_steps:
            return self.decay * (self.step_count / self.warmup_steps)
        return self.decay

    @torch.no_grad()
    def update(self, model):
        self.step_count += 1
        d = self._get_decay()
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = d * self.shadow[k] + (1.0 - d) * v
            else:
                self.shadow[k] = v  # copy integer buffers exactly

    def apply(self, model):
        """Load shadow weights into model."""
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model, backup):
        """Restore model from backup."""
        model.load_state_dict(backup, strict=False)


# ─────────────────────────────────────────────────────────────
# Progressive Ring Collapse (PRC)
# ─────────────────────────────────────────────────────────────

def get_prc_params(epoch, total_epochs):
    """
    Phase 1 (0 to 35%):  pure BPAnno — only Ω_I and Ω_O supervised
    Phase 2 (35% to 70%): gradual introduction of uncertain zones
    Phase 3 (70% to end): full MRPAnno
    """
    p1_end = int(total_epochs * 0.35)
    p2_end = int(total_epochs * 0.70)

    if epoch <= p1_end:
        return 0.0, 0.0, 1.0, 0.0, 1

    elif epoch <= p2_end:
        prog   = (epoch - p1_end) / max(p2_end - p1_end, 1)
        alpha  = 0.40 * prog
        beta   = 0.15 * prog
        tgt_d1 = 1.0 - 0.15 * prog
        tgt_d2 = 0.0 + 0.15 * prog
        return alpha, beta, tgt_d1, tgt_d2, 2

    else:
        return 0.40, 0.15, 0.85, 0.15, 3


# ─────────────────────────────────────────────────────────────
# Contrastive warm-up — step schedule
# ─────────────────────────────────────────────────────────────

def get_lam1(epoch, lambda1, total_epochs):
    if epoch < int(total_epochs * 0.10):
        return 0.0
    elif epoch < int(total_epochs * 0.25):
        return lambda1 * 0.33
    elif epoch < int(total_epochs * 0.40):
        return lambda1 * 0.67
    else:
        return lambda1


# ─────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────

def soft_dice_loss(pred_prob, soft_target, mask=None):
    smooth = 1e-6
    if mask is not None:
        pred_prob   = pred_prob[mask]
        soft_target = soft_target[mask]
    inter = (pred_prob * soft_target).sum()
    denom = pred_prob.sum() + soft_target.sum()
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)


def mrpanno_seg_loss(pred_logit, y_in, y_out,
                     omega_d1, omega_d2,
                     alpha, beta,
                     tgt_d1_val, tgt_d2_val):
    p_sig = torch.sigmoid(pred_logit)
    total = torch.tensor(0.0, device=pred_logit.device)

    # Certain foreground Ω_I
    mask_in = y_in.bool()
    if mask_in.any():
        tgt = torch.ones_like(p_sig)
        total = total + (
            soft_dice_loss(p_sig, tgt, mask_in) +
            F.binary_cross_entropy_with_logits(
                pred_logit[mask_in], tgt[mask_in], reduction='mean'
            )
        )

    # Certain background Ω_O
    mask_out = (~y_out.bool())
    if mask_out.any():
        tgt = torch.zeros_like(p_sig)
        total = total + (
            soft_dice_loss(p_sig, tgt, mask_out) +
            F.binary_cross_entropy_with_logits(
                pred_logit[mask_out], tgt[mask_out], reduction='mean'
            )
        )

    # Inner uncertain Ω_Δ1
    mask_d1 = omega_d1.bool()
    if mask_d1.any() and alpha > 0:
        tgt = torch.full_like(p_sig, tgt_d1_val)
        loss_d1 = (
            soft_dice_loss(p_sig, tgt, mask_d1) +
            F.binary_cross_entropy_with_logits(
                pred_logit[mask_d1], tgt[mask_d1], reduction='mean'
            )
        )
        total = total + alpha * loss_d1

    # Outer uncertain Ω_Δ2
    mask_d2 = omega_d2.bool()
    if mask_d2.any() and beta > 0:
        tgt = torch.full_like(p_sig, tgt_d2_val)
        loss_d2 = (
            soft_dice_loss(p_sig, tgt, mask_d2) +
            F.binary_cross_entropy_with_logits(
                pred_logit[mask_d2], tgt[mask_d2], reduction='mean'
            )
        )
        total = total + beta * loss_d2

    return total


def ccg3_loss(cls_logits, y_c3):
    return F.cross_entropy(cls_logits, y_c3)


def mrpanno_contrastive_loss(embeddings, pred_logit,
                              y_in, y_out,
                              omega_d1, omega_d2,
                              pmid_strip, neg_queue,
                              temperature=0.5,
                              max_anchors=256):
    B, D, H, W = embeddings.shape
    device = embeddings.device
    loss   = torch.tensor(0.0, device=device)
    count  = 0

    p_sig   = torch.sigmoid(pred_logit).detach()
    p_c     = p_sig.clamp(1e-6, 1-1e-6)
    entropy = -(p_c * p_c.log() + (1-p_c) * (1-p_c).log())

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
            perm       = torch.randperm(anchor_idx.shape[0], device=device)[:max_anchors]
            anchor_idx = anchor_idx[perm]

        pos_idx = in_b.nonzero(as_tuple=False)
        if pos_idx.shape[0] == 0:
            continue
        perm_p  = torch.randperm(pos_idx.shape[0], device=device)[:128]
        pos_emb = emb_b[:, pos_idx[perm_p, 0], pos_idx[perm_p, 1]].T
        pos_mean = F.normalize(pos_emb.mean(0, keepdim=True), dim=1)

        neg_idx  = out_b.nonzero(as_tuple=False)
        neg_list = []
        if neg_idx.shape[0] > 0:
            perm_n = torch.randperm(neg_idx.shape[0], device=device)[:128]
            neg_list.append(emb_b[:, neg_idx[perm_n, 0], neg_idx[perm_n, 1]].T)
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
# Validation — uses RAW model weights, not EMA
# EMA is only used for saving the best checkpoint
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device, epoch=0):
    model.eval()
    dice_sum, iou_sum, n = 0.0, 0.0, 0

    for pack in val_loader:
        (images, y_in, y_mid, y_out,
         omega_d1, omega_d2,
         pmid_strip, y_c3, gt) = pack

        images = images.to(device)
        gt_gpu = gt.to(device).squeeze(1).float()   # (B, H, W)

        preds    = model(images, mode='test')        # list of 4 tensors
        pred     = torch.sigmoid(preds[-1]).squeeze(1)  # finest scale
        pred_bin = (pred >= 0.5).float()
        gt_bin   = (gt_gpu >= 0.5).float()

        if n == 0 and epoch <= 3:
            print(f"    [val dbg ep{epoch}] "
                  f"pred mean={pred.mean():.4f} "
                  f"pred_bin mean={pred_bin.mean():.4f} "
                  f"gt_bin mean={gt_bin.mean():.4f}")

        for b in range(pred_bin.shape[0]):
            dice_sum += dice_coefficient(pred_bin[b], gt_bin[b]).item()
            iou_sum  += iou(pred_bin[b], gt_bin[b]).item()
            n        += 1

    model.train()
    return dice_sum / max(n, 1), iou_sum / max(n, 1)


# ─────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────

def test(model, path, dataset, opt, save_base=None):
    data_path  = os.path.join(path, dataset)
    image_root = os.path.join(data_path, 'images') + '/'
    gt_root    = os.path.join(data_path, 'masks')  + '/'
    model.eval()

    test_loader = get_loader(
        image_root=image_root, gt_root=gt_root,
        batchsize=1, trainsize=opt.img_size,
        shuffle=False, num_workers=4,
        pin_memory=True, augmentation=False,
        split='test',
    )

    DSC, IOU, total = 0.0, 0.0, 0
    detailed_results = []

    with torch.no_grad():
        for pack in tqdm(test_loader, desc=f'Testing {dataset}'):
            (images, y_in, y_mid, y_out,
             omega_d1, omega_d2,
             pmid_strip, y_c3, gt) = pack

            images = images.cuda()
            gt_gpu = gt.cuda().squeeze(1).float()

            preds    = model(images, mode='test')
            pred     = torch.sigmoid(preds[-1]).squeeze(1)
            pred_bin = (pred >= 0.5).float()
            gt_bin   = (gt_gpu >= 0.5).float()

            for b in range(pred_bin.shape[0]):
                d    = dice_coefficient(pred_bin[b], gt_bin[b]).item()
                io   = iou(pred_bin[b], gt_bin[b]).item()
                sens, spec, prec, hd = get_binary_metrics(pred_bin[b], gt_bin[b])
                DSC   += d
                IOU   += io
                total += 1
                detailed_results.append({
                    'Dice': d, 'IoU': io,
                    'Sensitivity': round(sens, 4),
                    'Specificity': round(spec, 4),
                    'Precision':   round(prec, 4),
                    'HD95':        round(hd,   4),
                })

                if save_base:
                    img_np = (pred_bin[b].cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_base, f'pred_{total}.png'), img_np)

    return DSC / total, IOU / total, detailed_results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id',           type=str,   required=True)
    parser.add_argument('--encoder',          type=str,   default='pvt_v2_b2')
    parser.add_argument('--expansion_factor', type=int,   default=2)
    parser.add_argument('--kernel_sizes',     type=int,   nargs='+', default=[1,3,5])
    parser.add_argument('--lgag_ks',          type=int,   default=3)
    parser.add_argument('--activation_mscb',  type=str,   default='relu6')
    parser.add_argument('--no_dw_parallel',   action='store_true', default=False)
    parser.add_argument('--concatenation',    action='store_true', default=False)
    parser.add_argument('--img_size',         type=int,   default=352)
    parser.add_argument('--batchsize',        type=int,   default=8)
    parser.add_argument('--epochs',           type=int,   default=100)
    parser.add_argument('--lr',               type=float, default=7.5e-5)
    parser.add_argument('--num_workers',      type=int,   default=4)
    parser.add_argument('--aug',              type=bool,  default=True)
    parser.add_argument('--pretrain',         type=bool,  default=True)
    parser.add_argument('--pretrained_dir',   type=str,   default='./pretrained_pth/pvt/')
    parser.add_argument('--color_image',      default=True)
    parser.add_argument('--train_save',       type=str,   default='./model_pth/')
    parser.add_argument('--train_path',       type=str,   default='./data/polyp/TrainDataset/')
    parser.add_argument('--val_path',         type=str,   default='./data/polyp/ValDataset/')
    parser.add_argument('--test_path',        type=str,   default='./data/polyp/TestDataset/')
    parser.add_argument('--dataset_name',     type=str,   default='Kvasir')
    parser.add_argument('--lambda1',          type=float, default=0.15)
    parser.add_argument('--lambda2',          type=float, default=0.40)
    parser.add_argument('--temperature',      type=float, default=0.5)
    parser.add_argument('--max_anchors',      type=int,   default=256)
    parser.add_argument('--ema_decay',        type=float, default=0.999)
    parser.add_argument('--ema_warmup',       type=int,   default=500)
    opt = parser.parse_args()

    # ── Paths ─────────────────────────────────────────────────
    save_path = os.path.join(opt.train_save, opt.run_id)
    os.makedirs(save_path,       exist_ok=True)
    os.makedirs('results_polyp', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    ).to(device)

    print(f'Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # ── EMA ───────────────────────────────────────────────────
    # warmup_steps controls how quickly EMA shadow tracks the model
    # 500 steps ≈ 3-4 epochs with batchsize=8, 800 samples
    ema = EMA(model, decay=opt.ema_decay, warmup_steps=opt.ema_warmup)

    # ── Data ──────────────────────────────────────────────────
    train_loader = get_loader(
        image_root=os.path.join(opt.train_path, 'images') + '/',
        gt_root=os.path.join(opt.train_path, 'masks') + '/',
        batchsize=opt.batchsize, trainsize=opt.img_size,
        shuffle=True, num_workers=opt.num_workers,
        pin_memory=True, augmentation=opt.aug, split='train',
    )
    val_loader = get_loader(
        image_root=os.path.join(opt.val_path, 'images') + '/',
        gt_root=os.path.join(opt.val_path, 'masks') + '/',
        batchsize=opt.batchsize, trainsize=opt.img_size,
        shuffle=False, num_workers=opt.num_workers,
        pin_memory=True, augmentation=False, split='val',
    )

    total_step = len(train_loader)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Val samples:      {len(val_loader.dataset)}')

    # ── Optimiser — differential LR ───────────────────────────
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

    best_dice = 0.0
    log_rows  = []

    # ── Training loop ─────────────────────────────────────────
    for epoch in range(1, opt.epochs + 1):
        model.train()

        (alpha_eff, beta_eff,
         tgt_d1, tgt_d2,
         prc_phase) = get_prc_params(epoch, opt.epochs)

        lam1_eff = get_lam1(epoch, opt.lambda1, opt.epochs)

        loss_total = 0.0
        loss_seg_t = 0.0
        loss_pcl_t = 0.0
        loss_ce3_t = 0.0

        for i, pack in enumerate(train_loader, start=1):
            (images, y_in, y_mid, y_out,
             omega_d1, omega_d2,
             pmid_strip, y_c3, gt) = pack

            images     = images.to(device)
            y_in       = y_in.to(device)
            y_out      = y_out.to(device)
            omega_d1   = omega_d1.to(device)
            omega_d2   = omega_d2.to(device)
            pmid_strip = pmid_strip.to(device)
            y_c3       = y_c3.to(device)
            # gt intentionally not sent to device — never used in loss

            preds, cls_logits, embeddings = model(images, mode='train')
            pred_logit = preds[-1].squeeze(1)   # (B, H, W)

            # L_seg — main head
            loss_seg = mrpanno_seg_loss(
                pred_logit, y_in, y_out,
                omega_d1, omega_d2,
                alpha_eff, beta_eff,
                tgt_d1, tgt_d2,
            )

            # L_seg — auxiliary heads
            aux_weights = [0.6, 0.4, 0.2]
            for aux_pred, w in zip(preds[:-1], aux_weights):
                loss_seg = loss_seg + w * mrpanno_seg_loss(
                    aux_pred.squeeze(1), y_in, y_out,
                    omega_d1, omega_d2,
                    alpha_eff, beta_eff,
                    tgt_d1, tgt_d2,
                )

            # L_ce3 — CCG classification
            loss_ce3 = ccg3_loss(cls_logits, y_c3)

            # L_pcl — pixel contrastive (step warm-up)
            loss_pcl = mrpanno_contrastive_loss(
                embeddings, pred_logit,
                y_in, y_out,
                omega_d1, omega_d2,
                pmid_strip,
                model.neg_queue,
                temperature=opt.temperature,
                max_anchors=opt.max_anchors,
            )

            # Update negative memory queue with Ω_O embeddings
            with torch.no_grad():
                bg = (~y_out.bool())
                for b in range(images.shape[0]):
                    idx = bg[b].nonzero(as_tuple=False)
                    if idx.shape[0] > 0:
                        perm = torch.randperm(idx.shape[0])[:32]
                        vecs = embeddings[b, :, idx[perm, 0], idx[perm, 1]].T
                        model.update_queue(vecs)

            # Total loss
            loss = (loss_seg
                    + lam1_eff * loss_pcl
                    + opt.lambda2 * loss_ce3)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)   # EMA tracks model with warm-up decay

            loss_total += loss.item()
            loss_seg_t += loss_seg.item()
            loss_pcl_t += loss_pcl.item() if isinstance(loss_pcl, torch.Tensor) else loss_pcl
            loss_ce3_t += loss_ce3.item()

            if i % 20 == 0 or i == total_step:
                pcl_val = loss_pcl.item() if isinstance(loss_pcl, torch.Tensor) else loss_pcl
                print(
                    f'Epoch [{epoch:03d}/{opt.epochs}] '
                    f'Step [{i:04d}/{total_step}] '
                    f'Loss: {loss.item():.4f} '
                    f'[seg={loss_seg.item():.3f} '
                    f'pcl={pcl_val:.3f} '
                    f'ce3={loss_ce3.item():.3f}] '
                    f'prc={prc_phase} '
                    f'lam1={lam1_eff:.3f} '
                    f'ema_d={ema._get_decay():.4f}'
                )

        # ── Scheduler step ────────────────────────────────────
        scheduler.step()

        # ── LR reduction at phase transitions ─────────────────
        if epoch == int(opt.epochs * 0.35):
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * 0.4
            print(f'  LR reduced at Phase1→2 (epoch {epoch})')

        if epoch == int(opt.epochs * 0.70):
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * 0.4
            print(f'  LR reduced at Phase2→3 (epoch {epoch})')

        if epoch == int(opt.epochs * 0.90):
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * 0.5
            print(f'  LR reduced for final fine-tuning (epoch {epoch})')

        # ── Validation — raw model weights ────────────────────
        # We validate with raw model (not EMA) so val_dice is meaningful
        # from epoch 1. EMA is only used when saving the best checkpoint.
        val_dice, val_iou = validate(model, val_loader, device, epoch=epoch)

        n_steps  = len(train_loader)
        avg_loss = loss_total / n_steps

        print(
            f'\n>>> Epoch {epoch:03d} | '
            f'avg_loss={avg_loss:.4f} | '
            f'val_dice={val_dice:.4f} | '
            f'val_iou={val_iou:.4f} | '
            f'prc_phase={prc_phase} | '
            f'ema_steps={ema.step_count}\n'
        )

        log_rows.append({
            'epoch':      epoch,
            'loss':       avg_loss,
            'loss_seg':   loss_seg_t / n_steps,
            'loss_pcl':   loss_pcl_t / n_steps,
            'loss_ce3':   loss_ce3_t / n_steps,
            'val_dice':   val_dice,
            'val_iou':    val_iou,
            'prc_phase':  prc_phase,
        })

        # Save best checkpoint using EMA weights
        if val_dice > best_dice:
            best_dice = val_dice
            # Temporarily apply EMA weights, save, then restore raw weights
            backup = copy.deepcopy(model.state_dict())
            ema.apply(model)
            torch.save(
                model.state_dict(),
                os.path.join(save_path, f'{opt.run_id}-best.pth')
            )
            model.load_state_dict(backup)
            print(f'  ✓ New best saved with EMA weights (val_dice={best_dice:.4f})\n')

        # Latest checkpoint — raw model weights
        torch.save(
            model.state_dict(),
            os.path.join(save_path, f'{opt.run_id}-latest.pth')
        )

    # ── Training log ──────────────────────────────────────────
    pd.DataFrame(log_rows).to_excel(
        f'results_polyp/TrainLog_{opt.run_id}.xlsx', index=False
    )
    print(f'\nTraining complete. Best val_dice = {best_dice:.4f}')
    print(f'Model saved: {save_path}/{opt.run_id}-best.pth')