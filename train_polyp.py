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
    """Dice loss, optionally restricted to valid_mask pixels."""
    if valid_mask is not None:
        pred_prob = pred_prob * valid_mask
        target    = target    * valid_mask
    p = pred_prob.reshape(pred_prob.shape[0], -1)
    t = target.reshape(target.shape[0], -1).float()
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    return (1 - (2 * inter + smooth) / (denom + smooth)).mean()


def dual_mask_loss(seg_logit, y_in, y_en, use_edge=True):
    """
    Lc: adversarial dual-mask Dice supervision.
    
    Pixels in omega_delta get contradictory labels from y_in and y_en,
    forcing boundary-invariant feature learning.
    Edge loss on y_en boundary preserves your original boundary sensitivity.
    """
    pred = torch.sigmoid(seg_logit)
    if pred.dim() == 4:
        pred = pred.squeeze(1)          # (B,1,H,W) → (B,H,W)

   
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

        pred_4d   = pred.unsqueeze(1)           # (B,1,H,W)
        y_en_4d   = y_en.unsqueeze(1).float()   # (B,1,H,W)
        pred_edge = F.conv2d(pred_4d,   laplacian, padding=1).squeeze(1)
        gt_edge   = F.conv2d(y_en_4d,  laplacian, padding=1).squeeze(1)
        # Only penalise edge error inside the envelope
        # (no point computing edge loss in pure background)
        edge_loss = F.l1_loss(pred_edge * y_en, gt_edge * y_en)
        Lc = Lc + 0.3 * edge_loss
    return Lc


def classification_loss_ccg(cls_logits, y_c):
    """
    Lce: 3-class cross-entropy for CCG head.
    y_c values: 0=certain bg, 1=uncertain ring, 2=certain fg
    """
    B, C, H, W = cls_logits.shape
    logits_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = y_c.reshape(-1)
    return F.cross_entropy(logits_flat, labels_flat)


@torch.no_grad()
@torch.no_grad()
def generate_pseudo_labels(seg_logit, cls_logits, omega_delta,
                            entropy_thresh=0.5):
    """
    Build integrated confidence map U for omega_delta pixels.

    U = 1  → pseudo foreground
    U = 0  → pseudo background
    U = -1 → too uncertain, skip in CCL

    Three-tier resolution for each uncertain pixel:
      1. CCG classifier directly predicts class 0 or 2  → use that
      2. CCG predicts class 1 (still uncertain)         → use second-best of {0,2}
      3. Seg-head also says foreground in ring           → override to fg (paper fix)
      4. Entropy ≥ threshold                            → assign -1, drop from CCL
    """
    seg_prob = torch.sigmoid(seg_logit)
    if seg_prob.dim() == 4:
        seg_prob = seg_prob.squeeze(1)          # (B,H,W)

    cls_prob = F.softmax(cls_logits, dim=1)     # (B,3,H,W)
    cls_pred = cls_prob.argmax(dim=1)           # (B,H,W)

    p_bg = cls_prob[:, 0]                       # P(certain background)
    p_fg = cls_prob[:, 2]                       # P(certain foreground)

    # ── Step 1: CCG second-best fallback for uncertain pixels ─────────
    # Start with argmax between bg and fg only
    U_c = (p_fg >= p_bg).long()                 # 1=fg, 0=bg


    U_c[cls_pred == 0] = 0                      # directly predicted bg
    U_c[cls_pred == 2] = 1                      # directly predicted fg
    seg_fg_in_ring = (seg_prob >= 0.5) & (omega_delta == 1)
    U_c[seg_fg_in_ring] = 1
    # ─────────────────────────────────────────────────────────────────

    # ── Step 3: Entropy-based hard uncertainty masking ────────────────
    eps = 1e-6
    entropy = -(
        seg_prob       * torch.log(seg_prob       + eps) +
        (1 - seg_prob) * torch.log(1 - seg_prob   + eps)
    )                                           # (B,H,W)

    U_e = torch.zeros_like(U_c)
    U_e[entropy >= entropy_thresh] = -1         # solid uncertain → drop

    # ── Step 4: Integrate (entropy overrides everything) ─────────────
    # U_c + 2*U_e:
    #   normal pixel  → U_c + 0  = {0,1}    kept
    #   high entropy  → U_c - 2  = {-2,-1}  clamped to -1
    U = torch.clamp(U_c + 2 * U_e, min=-1)
    # Only relevant inside the uncertain ring
    U = U * omega_delta.long()
    U[omega_delta == 0] = -1                    # outside ring → skip
    return U                                    # (B,H,W)


def contrastive_loss_ccl(embeddings, seg_logit, pseudo_labels,
                               omega_delta, y_in, y_en, neg_queue,
                               temperature=0.1, hard_ratio=0.7,
                               num_anchors=100, pixel_pool=512):
    """
    LPCL: batched pixel-wise contrastive loss with hard-sample mining.

    Key differences from slow version:
    - Similarity matrix computed ONCE per batch (GPU batched matmul)
    - Pixel pool subsamples H*W to 512 per image (avoids O(H*W) per anchor)
    - Inner loop only iterates over anchors, not anchor x pixel
    - 5-10x faster in practice

    Hard samples:
      - Certain pixels (Ω_I ∪ Ω_O) that are WRONGLY predicted
      - Uncertain pixels (Ω_Δ) with confident pseudo-label (U ≠ -1)
    Easy samples:
      - Certain pixels that are CORRECTLY predicted
    Ratio: 70% hard, 30% easy
    """
    B, D, H, W = embeddings.shape
    device = embeddings.device

    seg_pred = (torch.sigmoid(seg_logit.detach()).squeeze(1) >= 0.5).long()

    # ── Build hybrid label map ŷ ──────────────────────────────────────
    # Certain pixels: use seg prediction
    # Uncertain pixels: use CCG pseudo-labels
    y_hat = seg_pred.clone()
    for b in range(B):
        valid = (omega_delta[b] == 1) & (pseudo_labels[b] != -1)
        y_hat[b][valid] = pseudo_labels[b][valid]

    # ── Ground truth for certain region ──────────────────────────────
    certain_fg   = (y_in  == 1)
    certain_bg   = (y_en  == 0) & (omega_delta == 0)
    certain_gt   = torch.zeros_like(seg_pred)
    certain_gt[certain_fg] = 1
    certain_gt[certain_bg] = 0
    certain_mask = (omega_delta == 0)           # Ω_I ∪ Ω_O

    # ── Collect anchors and pixel pools across batch ──────────────────
    all_anchor_embs = []
    all_anchor_lbls = []
    all_pixel_embs  = []
    all_pixel_lbls  = []

    for b in range(B):
        # Hard: wrong certain pixels OR resolved uncertain pixels
        hard = (
            (certain_mask[b] & (seg_pred[b] != certain_gt[b])) |
            ((omega_delta[b] == 1) & (pseudo_labels[b] != -1))
        )
        # Easy: correctly predicted certain pixels
        easy = certain_mask[b] & (seg_pred[b] == certain_gt[b])

        h_idx = hard.nonzero(as_tuple=False)    # (N_h, 2)
        e_idx = easy.nonzero(as_tuple=False)    # (N_e, 2)

        if len(h_idx) == 0 or len(e_idx) == 0:
            continue

        n_h = min(int(num_anchors * hard_ratio), len(h_idx))
        n_e = min(num_anchors - n_h,             len(e_idx))

        h_sel = h_idx[torch.randperm(len(h_idx), device=device)[:n_h]]
        e_sel = e_idx[torch.randperm(len(e_idx), device=device)[:n_e]]
        anc   = torch.cat([h_sel, e_sel], dim=0)   # (N,2)

        ah, aw = anc[:, 0], anc[:, 1]

        # Anchor embeddings and labels
        all_anchor_embs.append(embeddings[b, :, ah, aw].T)     # (N,D)
        all_anchor_lbls.append(y_hat[b, ah, aw])               # (N,)

        # Pixel pool: subsample H*W → pixel_pool points
        flat_emb = embeddings[b].reshape(D, -1).T              # (H*W,D)
        flat_lbl = y_hat[b].reshape(-1)                        # (H*W,)
        pool_n   = min(pixel_pool, H * W)
        pool_idx = torch.randperm(H * W, device=device)[:pool_n]
        all_pixel_embs.append(flat_emb[pool_idx])              # (pool_n,D)
        all_pixel_lbls.append(flat_lbl[pool_idx])              # (pool_n,)

    # Nothing to compute (can happen early in training)
    if len(all_anchor_embs) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # ── Concatenate across batch ──────────────────────────────────────
    anc_emb  = torch.cat(all_anchor_embs, dim=0)   # (A, D)
    anc_lbl  = torch.cat(all_anchor_lbls, dim=0)   # (A,)
    pix_emb  = torch.cat(all_pixel_embs,  dim=0)   # (P, D)
    pix_lbl  = torch.cat(all_pixel_lbls,  dim=0)   # (P,)

    # ── Batched similarity matrices (ONE GPU matmul each) ─────────────
    # sim_pos_pool: (A, P) — anchor vs pixel pool
    # sim_neg_queue:(A, Q) — anchor vs memory queue
    sim_pos_pool  = torch.mm(anc_emb, pix_emb.T)  / temperature  # (A,P)
    sim_neg_queue = torch.mm(anc_emb, neg_queue)   / temperature  # (A,Q)

    # Precompute neg denominator per anchor (same for all positives)
    # shape (A,)
    neg_denom = torch.exp(sim_neg_queue).sum(dim=1)

    # ── Contrastive loss per anchor ───────────────────────────────────
    total_loss  = torch.tensor(0.0, device=device)
    valid_count = 0

    for i in range(len(anc_emb)):
        li = anc_lbl[i].item()
        if li == -1:
            continue

        pos_mask = (pix_lbl == li)                  # which pool pixels are +
        if pos_mask.sum() == 0:
            continue

        # Average positive similarity (numerator)
        pos_sim = torch.exp(sim_pos_pool[i][pos_mask]).mean()

        # Denominator = pos + all negatives from queue
        denom = pos_sim + neg_denom[i]

        loss_i = -torch.log(pos_sim / (denom + 1e-6))
        total_loss  = total_loss + loss_i
        valid_count += 1

    return total_loss / max(valid_count, 1)




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
        for pack in test_loader:
            if len(pack) == 3:
                images, _, gts = pack
            else:
                images, gts = pack
            images = images.cuda()
            gts = gts.cuda().float()

            ress = model(images)
            if not isinstance(ress, list):
                ress = [ress]
            predictions = ress[-1]
            
            for i in range(len(images)):
                p = predictions[i].unsqueeze(0)
                pred_resized = torch.sigmoid(p).squeeze()
                gt_resized = gts[i].squeeze()
                input_binary = (pred_resized >= 0.5).float()
                target_binary = (gt_resized >= 0.5).float()
                DSC += dice_coefficient(
                    input_binary,
                    target_binary
                ).item()

                IOU += iou(
                    input_binary,
                    target_binary
                ).item()

                total_images += 1

    return (
        DSC / total_images,
        IOU / total_images,
        total_images
    )

def train(train_loader, model, optimizer, epoch, opt, model_name):
    model.train()
    global best, test_dice_at_best_val, total_train_time, dict_plot
    
    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates = [0.75, 1, 1.25] 
    total_step = len(train_loader)

    for i, (images,y_in,y_en,omega_delta,y_c,gts) in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            images = Variable(images).cuda()
            weak_masks = Variable(weak_masks).float().cuda()
            gts = Variable(gts).float().cuda()

            if rate != 1:
                trainsize = int(round(opt.img_size * rate / 32) * 32)
                images = F.interpolate(
                    images,
                    size=(trainsize, trainsize),
                    mode='bilinear',
                    align_corners=True
                )

            # ── BPAnno forward pass ───────────────────────────────────
            use_ccg_ccl = (epoch > int(0.3 * opt.epoch))
            if use_ccg_ccl:
                P, cls_logits, embeddings = model(images, mode='train')
            else:
                model_out = model(images, mode='train')
                # first epoch warmup: model still returns tuple
                if isinstance(model_out, tuple):
                    P, cls_logits, embeddings = model_out
                else:
                    P = model_out

            if not isinstance(P, list):
                P = [P]

            # Resize BPAnno masks to match each prediction head size
            def resize_mask(m, size):
                # m: (B,H,W) → (B,1,H,W) → resize → (B,H,W)
                return F.interpolate(
                    m.unsqueeze(1).float(), size=size, mode='nearest'
                ).squeeze(1)

            target_size = P[0].shape[2:]   # all heads same H,W after interp
            y_in_r  = resize_mask(y_in,  target_size)
            y_en_r  = resize_mask(y_en,  target_size)

            # ── Lc: dual-mask Dice on all 4 heads + ensemble ──────────
            seg_ensemble = P[0] + P[1] + P[2] + P[3]
            loss_p1 = dual_mask_loss(P[0], y_in_r, y_en_r)
            loss_p2 = dual_mask_loss(P[1], y_in_r, y_en_r)
            loss_p3 = dual_mask_loss(P[2], y_in_r, y_en_r)
            loss_p4 = dual_mask_loss(P[3], y_in_r, y_en_r)
            loss_ens= dual_mask_loss(seg_ensemble, y_in_r, y_en_r)

            loss = loss_p1 + loss_p2 + loss_p3 + loss_p4 + loss_ens

            # ── CCG + CCL (added after warmup) ────────────────────────
            if use_ccg_ccl:
                # Resize ring mask to image size (cls/embed are full-res)
                H_img = images.shape[2]
                W_img = images.shape[3]
                y_in_full  = resize_mask(y_in,  (H_img, W_img))
                y_en_full  = resize_mask(y_en,  (H_img, W_img))
                od_full    = resize_mask(omega_delta, (H_img, W_img))
                y_c_full   = F.interpolate(
                    y_c.unsqueeze(1).float(),
                    size=(H_img, W_img), mode='nearest'
                ).squeeze(1).long()

                # Lce
                Lce = classification_loss_ccg(cls_logits, y_c_full)

                # Pseudo labels from CCG
                pseudo = generate_pseudo_labels(
                    P[-1].detach(), cls_logits.detach(), od_full
                )

                # LPCL
                LPCL = contrastive_loss_ccl(
                    embeddings, P[-1].detach(), pseudo,
                    od_full, y_in_full, y_en_full,
                    model.neg_queue,
                    temperature=0.1,
                    hard_ratio=0.7,
                    num_anchors=100
                )

                loss = loss + opt.lambda1 * LPCL + opt.lambda2 * Lce

                # Update memory queue
                with torch.no_grad():
                    flat = embeddings.permute(0,2,3,1).reshape(-1, model.embed_dim)
                    idx  = torch.randperm(flat.shape[0])[:64]
                    model.update_queue(flat[idx].detach())
            # ─────────────────────────────────────────────────────────

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        if i % 100 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {loss_record.show():.4f}')
        
    total_train_time += (time.time() - epoch_start)
    
    # Save Last
    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-last.pth"))

    # Validation and Testing
    epoch_results = {}
    for ds in ['test', 'val']:
        d_dice, d_iou, _ = test(model, opt.test_path, ds, opt)
        epoch_results[ds] = d_dice
        logging.info(f'Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
        print(f'Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
        dict_plot[ds].append(d_dice)

    # Check if Best Validation Dice
    if epoch_results['val'] > best:
        logging.info(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
        print(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
        best = epoch_results['val']
        test_dice_at_best_val = epoch_results['test'] # Track test dice at peak val
        torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-best.pth"))
    
if __name__ == '__main__':
    # Initial defaults
    dataset_name = 'ClinicDB' 
    
    parser = argparse.ArgumentParser()
    # network related parameters
    parser.add_argument('--encoder', type=str,
                        default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
    parser.add_argument('--expansion_factor', type=int,
                        default=2, help='expansion factor in MSCB block')
    parser.add_argument('--kernel_sizes', type=int, nargs='+',
                        default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
    parser.add_argument('--lgag_ks', type=int,
                        default=3, help='Kernel size in LGAG')
    parser.add_argument('--activation_mscb', type=str,
                        default='relu6', help='activation used in MSCB: relu6 or relu')
    parser.add_argument('--no_dw_parallel', action='store_true', 
                        default=False, help='use this flag to disable depth-wise parallel convolutions')
    parser.add_argument('--concatenation', action='store_true', 
                        default=False, help='use this flag to concatenate feature maps in MSDC block')
    parser.add_argument('--no_pretrain', action='store_true', 
                        default=False, help='use this flag to turn off loading pretrained enocder weights')
    parser.add_argument('--pretrained_dir', type=str,
                        default='./pretrained_pth/pvt/', help='path to pretrained encoder dir')
    parser.add_argument('--supervision', type=str,
                    default='mutation', help='loss supervision: mutation, deep_supervision or last_layer')    
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005) 
    parser.add_argument('--alpha',   type=float, default=0.3)  # kept for compat
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
    parser.add_argument('--train_path', type=str, default=f'./data/polyp/target/{dataset_name}/train/')
    parser.add_argument('--test_path', type=str, default=f'./data/polyp/target/{dataset_name}/')
    parser.add_argument('--train_save', type=str, default='') 
    
    opt = parser.parse_args()

    for run in [1,2,3,4,5]:
        dict_plot = {'val': [], 'test': []}
        best = 0.0
        test_dice_at_best_val = 0.0
        total_train_time = 0

        if opt.concatenation:
            aggregation = 'concat'
        else: 
            aggregation = 'add'
        
        if opt.no_dw_parallel:
            dw_mode = 'series'
        else: 
            dw_mode = 'parallel'

        timestamp = time.strftime('%H%M%S')
        run_id = (f"{dataset_name}_{opt.encoder}_EMCAD_kernel_sizes_{opt.kernel_sizes}_dw_{dw_mode}_{aggregation}_lgag_ks_{opt.lgag_ks}_ef{opt.expansion_factor}_act_mscb_{opt.activation_mscb}_bs{opt.batchsize}_cas_lr{opt.lr}_"
                      f"e{opt.epoch}_aug{opt.augmentation}_run{run}_t{timestamp}")
        run_id = run_id.replace('[', '').replace(']', '').replace(', ', '_')
        opt.train_save = f'./model_pth/{run_id}/'
        
        os.makedirs('logs', exist_ok=True)
        os.makedirs(opt.train_save, exist_ok=True)
        
        logging.basicConfig(filename=f'logs/train_log_{run_id}.log', level=logging.INFO, 
                            format='[%(asctime)s] %(message)s', force=True)


        # Build model
        #model = EMCADNet(dw_parallel=dw_parallel, expansion_factor=expansion_factor, add=add, kernel_sizes=kernel_sizes, att_ks=att_ks, activation=activation, encoder=encoder, pretrain=pretrain, head=head, bbox=False, cds=False) # head='SAH'
        model = EMCADNet(num_classes=1, kernel_sizes=opt.kernel_sizes, expansion_factor=opt.expansion_factor, dw_parallel=not opt.no_dw_parallel, add=not opt.concatenation, lgag_ks=opt.lgag_ks, activation=opt.activation_mscb, encoder=opt.encoder, pretrain= not opt.no_pretrain, pretrained_dir=opt.pretrained_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        '''if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)'''

        model.to(device)

        print(f"Encoder: {opt.encoder} | Decoder: EMCAD")
        cal_params_flops(model, opt.img_size, logging)
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

        train_loader = get_loader(
            image_root=f'{opt.train_path}/images/', gt_root=f'{opt.train_path}/masks/',
            batchsize=opt.batchsize, trainsize=opt.img_size, 
            shuffle=True, augmentation=opt.augmentation, split='train', color_image=opt.color_image
        )

        for epoch in range(1, opt.epoch + 1):
            adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            train(train_loader, model, optimizer, epoch, opt, run_id)
            scheduler.step()
        # FINAL SUMMARY
        
        summary = (f"\n{'='*40}\nFINAL RESULTS: {run_id}\n"
                   f"Best Val Dice: {best:.4f}\n"
                   f"Test Dice at Best Val: {test_dice_at_best_val:.4f}\n"
                   f"Total Train Time: {total_train_time:.2f}s\n{'='*40}")
        print(summary)
        logging.info(summary)
        
        