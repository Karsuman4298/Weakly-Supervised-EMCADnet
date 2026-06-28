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

from lib.networks_bp import EMCADNet
from utils.dataloader_bp import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter, cal_params_flops


# ============================================================
# 1. LOSS FUNCTIONS  (paper-exact)
# ============================================================

def dual_mask_loss(seg_logit, y_in, y_en, use_edge=True):
    """
    Paper L_c = L_in + L_out

    L_in  : prediction INSIDE inner polygon (y_in)  should be 1
    L_out : prediction OUTSIDE outer envelope (y_en) should be 0

    y_in  : B x H x W  binary  (1 = certain foreground)
    y_en  : B x H x W  binary  (1 = inside outer envelope)
    """
    pred = torch.sigmoid(seg_logit)
    if pred.dim() == 4:
        pred = pred.squeeze(1)          # B x H x W

    eps = 1e-6

    # ── L_in : inside inner polygon prediction → 1 ──────────
    # Constrain only to pixels where y_in == 1
    L_in = F.binary_cross_entropy(
        (pred * y_in).clamp(eps, 1 - eps),
        y_in,
        reduction='mean'
    )

    # ── L_out : outside outer envelope prediction → 0 ────────
    # certain background = pixels where y_en == 0
    outside_mask = (1.0 - y_en)                     # 1 outside envelope
    L_out = F.binary_cross_entropy(
        (pred * outside_mask).clamp(eps, 1 - eps),
        torch.zeros_like(pred),
        reduction='mean'
    )

    Lc = L_in + L_out

    # ── Optional edge loss (boundary sharpening) ─────────────
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
        gt_edge   = F.conv2d(y_en_4d,  laplacian, padding=1).squeeze(1)
        edge_loss = F.l1_loss(pred_edge * y_en, gt_edge * y_en)
        Lc        = Lc + 0.3 * edge_loss

    return Lc


def classification_loss_ccg(cls_logits, y_c, omega_delta):
    """
    Paper CCG loss: 3-class cross-entropy ONLY inside
    uncertainty ring Ω_δ.

    cls_logits  : B x 3 x H x W
    y_c         : B x H x W  long  {0=bg, 1=uncertain, 2=fg}
    omega_delta : B x H x W  float {0, 1}
    """
    B, C, H, W  = cls_logits.shape

    logits_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = y_c.reshape(-1)
    ring_mask   = omega_delta.reshape(-1).bool()     # only inside ring

    if ring_mask.sum() == 0:
        return torch.tensor(
            0.0, device=cls_logits.device, requires_grad=True
        )

    return F.cross_entropy(
        logits_flat[ring_mask],
        labels_flat[ring_mask]
    )


# ============================================================
# 2. PSEUDO LABEL GENERATION  (paper Algorithm 1, exact order)
# ============================================================

@torch.no_grad()
def generate_pseudo_labels(seg_logit, cls_logits, omega_delta,
                            entropy_thresh=0.5):
    """
    Paper Algorithm 1:
      Step 1 : initialise U = -1  (all ignored)
      Step 2 : class-based assignment  inside Ω_δ
      Step 3 : seg-guided refinement   inside Ω_δ
      Step 4 : entropy OVERRIDES uncertain pixels back to -1
      Step 5 : pixels outside Ω_δ stay -1

    Returns U : B x H x W  long  {-1=ignore, 0=bg, 1=fg}
    """
    seg_prob = torch.sigmoid(seg_logit)
    if seg_prob.dim() == 4:
        seg_prob = seg_prob.squeeze(1)          # B x H x W

    cls_prob = F.softmax(cls_logits, dim=1)     # B x 3 x H x W
    cls_pred = cls_prob.argmax(dim=1)           # B x H x W

    in_ring = (omega_delta == 1)

    # Step 1 — all ignored
    U = torch.full_like(cls_pred, fill_value=-1, dtype=torch.long)

    # Step 2 — class-based labels inside ring
    U[in_ring & (cls_pred == 0)] = 0           # background class
    U[in_ring & (cls_pred == 2)] = 1           # foreground class
    # cls_pred == 1  stays -1  (uncertain class)

    # Step 3 — seg-model agreement refinement inside ring
    seg_fg_in_ring = in_ring & (seg_prob >= 0.5)
    U[seg_fg_in_ring] = 1

    # Step 4 — entropy overrides: high-entropy → ignore
    eps     = 1e-6
    entropy = -(
        seg_prob * torch.log(seg_prob + eps) +
        (1 - seg_prob) * torch.log(1 - seg_prob + eps)
    )
    U[in_ring & (entropy >= entropy_thresh)] = -1   # overrides steps 2 & 3

    # Step 5 — outside ring always ignored
    U[~in_ring] = -1

    return U                                         # B x H x W


# ============================================================
# 3. PIXEL CONTRASTIVE LOSS  (paper CCL)
# ============================================================

def contrastive_loss_ccl(embeddings, seg_logit, pseudo_labels,
                          omega_delta, y_in, y_en, neg_queue,
                          temperature=0.1, hard_ratio=0.7,
                          num_anchors=100, pixel_pool=512):
    """
    Paper CCL: pixel-level contrastive loss with hard/easy anchor
    sampling and a momentum negative queue.

    embeddings   : B x D x H x W  (L2-normalised in model forward)
    neg_queue    : D x Q           (detached; updated after backward)
    """
    neg_queue = neg_queue.detach()                   # never enters grad graph

    B, D, H, W = embeddings.shape
    device     = embeddings.device

    # Current seg prediction (detached)
    seg_pred = (
        torch.sigmoid(seg_logit.detach()).squeeze(1) >= 0.5
    ).long()                                         # B x H x W

    # Refined label map: use pseudo labels inside ring,
    # seg prediction elsewhere
    y_hat = seg_pred.clone()
    for b in range(B):
        valid         = (omega_delta[b] == 1) & (pseudo_labels[b] != -1)
        y_hat[b][valid] = pseudo_labels[b][valid]

    # Certain regions (outside ring)
    certain_fg   = (y_in  == 1)
    certain_bg   = (y_en  == 0) & (omega_delta == 0)
    certain_gt   = torch.zeros_like(seg_pred)
    certain_gt[certain_fg] = 1
    certain_mask = (omega_delta == 0)               # outside ring = certain

    all_anchor_embs = []
    all_anchor_lbls = []
    all_pixel_embs  = []
    all_pixel_lbls  = []

    for b in range(B):
        # Hard anchors: certain pixels mis-classified  OR  ring pixels
        # with valid pseudo label
        hard = (
            (certain_mask[b] & (seg_pred[b] != certain_gt[b])) |
            ((omega_delta[b] == 1) & (pseudo_labels[b] != -1))
        )
        # Easy anchors: certain pixels correctly classified
        easy = certain_mask[b] & (seg_pred[b] == certain_gt[b])

        h_idx = hard.nonzero(as_tuple=False)        # N_h x 2
        e_idx = easy.nonzero(as_tuple=False)        # N_e x 2

        if len(h_idx) == 0 or len(e_idx) == 0:
            continue

        n_h   = min(int(num_anchors * hard_ratio), len(h_idx))
        n_e   = min(num_anchors - n_h,             len(e_idx))

        h_sel = h_idx[torch.randperm(len(h_idx), device=device)[:n_h]]
        e_sel = e_idx[torch.randperm(len(e_idx), device=device)[:n_e]]
        anc   = torch.cat([h_sel, e_sel], dim=0)   # num_anchors x 2

        ah, aw = anc[:, 0], anc[:, 1]

        # Anchor embeddings and their labels
        all_anchor_embs.append(embeddings[b, :, ah, aw].T)   # N x D
        all_anchor_lbls.append(y_hat[b, ah, aw])             # N

        # Pixel pool for positives
        flat_emb = embeddings[b].reshape(D, -1).T            # HW x D
        flat_lbl = y_hat[b].reshape(-1)                      # HW
        pool_n   = min(pixel_pool, H * W)
        pool_idx = torch.randperm(H * W, device=device)[:pool_n]
        all_pixel_embs.append(flat_emb[pool_idx])            # pool_n x D
        all_pixel_lbls.append(flat_lbl[pool_idx])            # pool_n

    if len(all_anchor_embs) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    anc_emb = torch.cat(all_anchor_embs, dim=0)   # A x D
    anc_lbl = torch.cat(all_anchor_lbls, dim=0)   # A
    pix_emb = torch.cat(all_pixel_embs,  dim=0)   # P x D
    pix_lbl = torch.cat(all_pixel_lbls,  dim=0)   # P

    # Similarity matrices
    # anc_emb & pix_emb are already L2-normalised (done in model)
    # neg_queue shape: D x Q  →  anc @ neg_queue = A x Q
    sim_pos_pool  = torch.mm(anc_emb, pix_emb.T) / temperature  # A x P
    sim_neg_queue = torch.mm(anc_emb, neg_queue)  / temperature  # A x Q
    neg_denom     = torch.exp(sim_neg_queue).sum(dim=1)           # A

    total_loss  = torch.tensor(0.0, device=device)
    valid_count = 0

    for i in range(len(anc_emb)):
        li = anc_lbl[i].item()
        if li == -1:
            continue                             # ignore uncertain anchors

        pos_mask = (pix_lbl == li)
        if pos_mask.sum() == 0:
            continue

        pos_sim  = torch.exp(sim_pos_pool[i][pos_mask]).mean()
        denom    = pos_sim + neg_denom[i]
        loss_i   = -torch.log(pos_sim / (denom + 1e-6))
        total_loss  = total_loss + loss_i
        valid_count += 1

    return total_loss / max(valid_count, 1)


# ============================================================
# 4. METRICS
# ============================================================

def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth         = 1e-6
    p              = predicted.contiguous().view(-1)
    g              = labels.contiguous().view(-1)
    intersection   = (p * g).sum()
    return (2. * intersection + smooth) / (p.sum() + g.sum() + smooth)


def iou_score(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth       = 1e-6
    p            = predicted.contiguous().view(-1)
    g            = labels.contiguous().view(-1)
    intersection = (p * g).sum()
    union        = p.sum() + g.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ============================================================
# 5. EVALUATION
# ============================================================

def evaluate(model, image_root, gt_root, opt, split_name='val'):
    model.eval()

    loader = get_loader(
        image_root   = image_root,
        gt_root      = gt_root,
        batchsize    = opt.test_batchsize,
        trainsize    = opt.img_size,
        shuffle      = False,
        augmentation = False,
        split        = 'test',
        color_image  = opt.color_image
    )

    total_dice   = 0.0
    total_iou    = 0.0
    total_images = 0
    debug_done   = False

    with torch.no_grad():
        for pack in loader:
            images = pack[0].cuda()

            # ── Find GT index automatically ──────────────────
            # Test every index to find the mask
            if not debug_done:
                print(f'\n[DEBUG evaluate {split_name}]')
                print(f'  Pack length : {len(pack)}')
                for i, p in enumerate(pack):
                    if torch.is_tensor(p):
                        print(
                            f'  pack[{i}] : shape={tuple(p.shape)}'
                            f'  min={p.min():.3f}'
                            f'  max={p.max():.3f}'
                            f'  dtype={p.dtype}'
                        )
                debug_done = True

            # GT is pack[5] for BPAnno loader
            # (image, y_in, y_en, omega_delta, y_c, gt)
            gts = pack[5].cuda().float()

            # Normalise GT to [0,1] if stored as 0/255
            if gts.max().item() > 1.0:
                gts = gts / 255.0

            # Ensure shape B x 1 x H x W
            if gts.dim() == 3:
                gts = gts.unsqueeze(1)

            # ── Forward ──────────────────────────────────────
            ress = model(images, mode='test')
            if not isinstance(ress, list):
                ress = [ress]
            preds = ress[-1]                        # finest output

            # Ensure pred shape matches GT
            if preds.shape[2:] != gts.shape[2:]:
                preds = F.interpolate(
                    preds,
                    size=gts.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            # ── Per-image metrics ─────────────────────────────
            for idx in range(images.shape[0]):
                prob      = torch.sigmoid(preds[idx]).squeeze()   # H x W
                gt        = gts[idx].squeeze()                     # H x W
                pred_bin  = (prob >= 0.5).float()
                gt_bin    = (gt   >= 0.5).float()

                # Debug first image
                if total_images == 0:
                    print(
                        f'  [First image debug]\n'
                        f'    prob  : min={prob.min():.4f}'
                        f'  max={prob.max():.4f}'
                        f'  mean={prob.mean():.4f}\n'
                        f'    gt    : min={gt.min():.4f}'
                        f'  max={gt.max():.4f}'
                        f'  mean={gt.mean():.4f}\n'
                        f'    pred_bin sum : {pred_bin.sum().item():.0f}\n'
                        f'    gt_bin   sum : {gt_bin.sum().item():.0f}'
                    )

                total_dice   += dice_coefficient(pred_bin, gt_bin).item()
                total_iou    += iou_score(pred_bin, gt_bin).item()
                total_images += 1

    n = max(total_images, 1)
    return total_dice / n, total_iou / n, total_images


# ============================================================
# 6. TRAINING
# ============================================================

def train_one_epoch(train_loader, model, optimizer, epoch, opt, model_name):
    """
    One full training epoch following EAUWSeg:
      • epochs 1 … 0.3*E  : L_c only           (warmup)
      • epochs 0.3*E+1 … E : L_c + CCG + CCL   (full objective)
    """
    global best_val_dice, test_dice_at_best_val
    global total_train_time, dict_plot

    model.train()

    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates  = [0.75, 1.0, 1.25]
    total_step  = len(train_loader)

    # Paper: warm-up for first 30 % of epochs
    use_ccg_ccl = (epoch > int(0.3 * opt.epoch))

    for step, (images, y_in, y_en, omega_delta, y_c, gts) in enumerate(
            train_loader, start=1):

        # Move to GPU once per step
        images_ori      = Variable(images).cuda()
        y_in_ori        = y_in.float().cuda()
        y_en_ori        = y_en.float().cuda()
        omega_delta_ori = omega_delta.float().cuda()
        y_c_ori         = y_c.long().cuda()
        gts_ori         = Variable(gts).float().cuda()

        for rate in size_rates:
            optimizer.zero_grad(set_to_none=True)

            # ── Multi-scale resize ───────────────────────────
            if rate != 1.0:
                trainsize   = int(round(opt.img_size * rate / 32) * 32)
                images_r    = F.interpolate(
                    images_ori,
                    size=(trainsize, trainsize),
                    mode='bilinear',
                    align_corners=True
                )
            else:
                images_r = images_ori

            # ── Forward pass ─────────────────────────────────
            # model returns:
            #   train mode → (P_list, cls_logits, embeddings)
            #   test  mode → P_list
            P, cls_logits, embeddings = model(images_r, mode='train')

            if not isinstance(P, list):
                P = [P]

            # ── Resize helper (nearest for masks) ────────────
            def resize_mask(m, size):
                if m.dim() == 3:
                    m = m.unsqueeze(1)
                out = F.interpolate(m.float(), size=size, mode='nearest')
                return out.squeeze(1)

            target_size = P[0].shape[2:]            # H' x W' of decoder out
            y_in_r      = resize_mask(y_in_ori,        target_size)
            y_en_r      = resize_mask(y_en_ori,        target_size)

            # ── L_c : dual-mask loss on all 4 heads + ensemble ─
            seg_ensemble = P[0] + P[1] + P[2] + P[3]

            loss = (
                dual_mask_loss(P[0],         y_in_r, y_en_r) +
                dual_mask_loss(P[1],         y_in_r, y_en_r) +
                dual_mask_loss(P[2],         y_in_r, y_en_r) +
                dual_mask_loss(P[3],         y_in_r, y_en_r) +
                dual_mask_loss(seg_ensemble, y_in_r, y_en_r)
            )

            # ── CCG + CCL (after warmup) ──────────────────────
            emb_snapshot = None

            if use_ccg_ccl:
                H_img = images_r.shape[2]
                W_img = images_r.shape[3]

                # Resize all weak labels to image resolution
                y_in_full   = resize_mask(y_in_ori,        (H_img, W_img))
                y_en_full   = resize_mask(y_en_ori,        (H_img, W_img))
                od_full     = resize_mask(omega_delta_ori, (H_img, W_img))
                y_c_full    = F.interpolate(
                    y_c_ori.unsqueeze(1).float(),
                    size=(H_img, W_img),
                    mode='nearest'
                ).squeeze(1).long()

                # L_ce : CCG classification loss ONLY inside Ω_δ
                Lce = classification_loss_ccg(
                    cls_logits, y_c_full, od_full       # ← ring mask passed
                )

                # Pseudo labels for CCL
                pseudo = generate_pseudo_labels(
                    P[-1].detach(),
                    cls_logits.detach(),
                    od_full,
                    entropy_thresh=0.5
                )

                # L_PCL : pixel contrastive loss
                # embeddings are L2-normalised inside model.forward()
                LPCL = contrastive_loss_ccl(
                    embeddings,
                    P[-1].detach(),
                    pseudo,
                    od_full,
                    y_in_full,
                    y_en_full,
                    model.neg_queue.detach().clone(),
                    temperature=0.1,
                    hard_ratio=0.7,
                    num_anchors=100,
                    pixel_pool=512
                )

                loss = loss + opt.lambda1 * LPCL + opt.lambda2 * Lce

                # Snapshot embeddings BEFORE backward for queue update
                emb_snapshot = (
                    embeddings.detach()
                    .permute(0, 2, 3, 1)
                    .reshape(-1, model.embed_dim)
                )

            # ── Backward ─────────────────────────────────────
            loss.backward()

            # Queue update AFTER backward (no grad retained)
            if emb_snapshot is not None:
                with torch.no_grad():
                    idx = torch.randperm(
                        emb_snapshot.shape[0],
                        device=emb_snapshot.device
                    )[:64]
                    model.update_queue(emb_snapshot[idx])

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1.0:
                loss_record.update(loss.item(), opt.batchsize)

        # ── Logging ──────────────────────────────────────────
        if step % 100 == 0 or step == total_step:
            phase = 'CCG+CCL' if use_ccg_ccl else 'warmup-Lc'
            print(
                f'{datetime.now()} '
                f'Epoch [{epoch:03d}/{opt.epoch:03d}]  '
                f'Step [{step:04d}/{total_step:04d}]  '
                f'LR: {optimizer.param_groups[0]["lr"]:.7f}  '
                f'Loss: {loss_record.show():.4f}  [{phase}]'
            )

    total_train_time += (time.time() - epoch_start)

    # ── Save last checkpoint ──────────────────────────────────
    os.makedirs(opt.train_save, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(opt.train_save, f'{model_name}-last.pth')
    )

    # ── Evaluate val & test ───────────────────────────────────
    results = {}
    for split, img_root, gt_root in [
        ('val',  f'{opt.val_path}/images/',  f'{opt.val_path}/masks/'),
        ('test', f'{opt.test_path}/images/', f'{opt.test_path}/masks/'),
    ]:
        d, iou_val, n = evaluate(
            model, img_root, gt_root, opt, split_name=split
        )
        results[split] = d
        msg = (
            f'Epoch {epoch:03d} | {split:4s} | '
            f'Dice: {d:.4f} | IoU: {iou_val:.4f} | Images: {n}'
        )
        print(msg)
        logging.info(msg)
        dict_plot[split].append(d)

    # ── Save best checkpoint based on val Dice ────────────────
    if results['val'] > best_val_dice:
        msg = (
            f'### Best Model Updated '
            f'(Val Dice {best_val_dice:.4f} → {results["val"]:.4f}) ###'
        )
        print(msg)
        logging.info(msg)

        best_val_dice         = results['val']
        test_dice_at_best_val = results['test']

        torch.save(
            model.state_dict(),
            os.path.join(opt.train_save, f'{model_name}-best.pth')
        )


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == '__main__':

    dataset_name = 'ColonDB'

    parser = argparse.ArgumentParser(
        description='EAUWSeg / BPAnno weakly-supervised training'
    )

    # ── Model ────────────────────────────────────────────────
    parser.add_argument('--encoder',          type=str,  default='pvt_v2_b2')
    parser.add_argument('--expansion_factor', type=int,  default=2)
    parser.add_argument('--kernel_sizes',     type=int,  nargs='+',
                        default=[1, 3, 5])
    parser.add_argument('--lgag_ks',          type=int,  default=3)
    parser.add_argument('--activation_mscb',  type=str,  default='relu6')
    parser.add_argument('--no_dw_parallel',   action='store_true',
                        default=False)
    parser.add_argument('--concatenation',    action='store_true',
                        default=False)
    parser.add_argument('--no_pretrain',      action='store_true',
                        default=False)
    parser.add_argument('--pretrained_dir',   type=str,
                        default='./pretrained_pth/pvt/')

    # ── Training ─────────────────────────────────────────────
    parser.add_argument('--epoch',          type=int,   default=200)
    parser.add_argument('--lr',             type=float, default=5e-4)
    parser.add_argument('--lambda1',        type=float, default=0.3,
                        help='Weight for CCL loss L_PCL')
    parser.add_argument('--lambda2',        type=float, default=0.5,
                        help='Weight for CCG loss L_ce')
    parser.add_argument('--batchsize',      type=int,   default=8)
    parser.add_argument('--test_batchsize', type=int,   default=8)
    parser.add_argument('--img_size',       type=int,   default=352)
    parser.add_argument('--clip',           type=float, default=0.5)
    parser.add_argument('--decay_rate',     type=float, default=0.1)
    parser.add_argument('--decay_epoch',    type=int,   default=300)
    parser.add_argument('--color_image',    default=True)
    parser.add_argument('--augmentation',   default=True)

    # ── Paths ─────────────────────────────────────────────────
    parser.add_argument('--train_path',  type=str, required=True,
                        help='Root of training split  (contains images/ masks/)')
    parser.add_argument('--val_path',    type=str, required=True,
                        help='Root of validation split (contains images/ masks/)')
    parser.add_argument('--test_path',   type=str, required=True,
                        help='Root of test split       (contains images/ masks/)')
    parser.add_argument('--train_save',  type=str, default='',
                        help='Override checkpoint save dir (auto-generated if empty)')
    parser.add_argument('--resume',      type=str, default='',
                        help='Path to .pth checkpoint to resume from')

    opt = parser.parse_args()

    # ── Early path validation ─────────────────────────────────
    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    print('\n── Dataset paths ──────────────────────────────────')
    for split, path in [('train', opt.train_path),
                         ('val',   opt.val_path),
                         ('test',  opt.test_path)]:
        for sub in ['images', 'masks']:
            full = os.path.join(path, sub)
            if not os.path.isdir(full):
                raise FileNotFoundError(
                    f'[{split}] {sub}/ folder not found:\n'
                    f'  {full}\n'
                    f'Expected: {path}/images/  and  {path}/masks/'
                )
        n = len([
            f for f in os.listdir(os.path.join(path, 'images'))
            if f.lower().endswith(IMG_EXTS)
        ])
        print(f'  [{split:5s}]  {path}  ({n} images)')
    print()

    # ── Five independent runs (as in paper) ──────────────────
    for run in [1, 2, 3, 4, 5]:

        # Per-run globals
        dict_plot             = {'val': [], 'test': []}
        best_val_dice         = 0.0
        test_dice_at_best_val = 0.0
        total_train_time      = 0.0

        aggregation = 'concat' if opt.concatenation  else 'add'
        dw_mode     = 'series' if opt.no_dw_parallel else 'parallel'
        timestamp   = time.strftime('%H%M%S')

        run_id = (
            f"{dataset_name}_{opt.encoder}_EMCAD"
            f"_ks_{'_'.join(map(str, opt.kernel_sizes))}"
            f"_dw_{dw_mode}_{aggregation}"
            f"_lgag{opt.lgag_ks}"
            f"_ef{opt.expansion_factor}"
            f"_act_{opt.activation_mscb}"
            f"_bs{opt.batchsize}"
            f"_lr{opt.lr}"
            f"_e{opt.epoch}"
            f"_aug{opt.augmentation}"
            f"_run{run}_t{timestamp}"
        )

        save_dir       = opt.train_save or f'./model_pth/{run_id}/'
        opt.train_save = save_dir

        os.makedirs('logs',    exist_ok=True)
        os.makedirs(save_dir,  exist_ok=True)

        logging.basicConfig(
            filename=f'logs/train_log_{run_id}.log',
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            force=True
        )

        print(f'── Run {run}/5  ──  {run_id}')
        logging.info(f'Run {run}/5  {run_id}')

        # ── Build model ──────────────────────────────────────
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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Resume from checkpoint
        if opt.resume and os.path.isfile(opt.resume):
            print(f'  Resuming from: {opt.resume}')
            ckpt = torch.load(opt.resume, map_location=device)
            # Strip profiling keys (total_ops / total_params)
            if isinstance(ckpt, dict):
                for meta_key in ['model', 'state_dict', 'model_state_dict']:
                    if meta_key in ckpt:
                        ckpt = ckpt[meta_key]
                        break
                ckpt = {k: v for k, v in ckpt.items()
                        if torch.is_tensor(v)}
            model.load_state_dict(ckpt, strict=False)
            print('  Checkpoint loaded.')
        elif opt.resume:
            print(f'  WARNING: checkpoint not found at {opt.resume}')

        print(f'  Encoder : {opt.encoder}')
        print(f'  Device  : {device}')
        cal_params_flops(model, opt.img_size, logging)

        # ── Optimiser & scheduler ────────────────────────────
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

        # ── Data loader ──────────────────────────────────────
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

        print(f'  Train batches : {len(train_loader)}')
        print(f'  Epochs        : {opt.epoch}')
        print(f'  Warmup ends   : epoch {int(0.3 * opt.epoch)}')
        print(f'  λ1 (CCL)      : {opt.lambda1}')
        print(f'  λ2 (CCG)      : {opt.lambda2}\n')

        # ── Training loop ────────────────────────────────────
        for epoch in range(1, opt.epoch + 1):
            adjust_lr(optimizer, opt.lr, epoch,
                      opt.decay_rate, opt.decay_epoch)
            train_one_epoch(
                train_loader, model, optimizer,
                epoch, opt, run_id
            )
            scheduler.step()

        # ── Final summary ────────────────────────────────────
        summary = (
            f"\n{'='*55}\n"
            f"FINAL RESULTS  run {run}/5 : {run_id}\n"
            f"  Best Val Dice         : {best_val_dice:.4f}\n"
            f"  Test Dice @ Best Val  : {test_dice_at_best_val:.4f}\n"
            f"  Total Train Time      : {total_train_time:.1f}s  "
            f"({total_train_time/3600:.2f}h)\n"
            f"{'='*55}"
        )
        print(summary)
        logging.info(summary)