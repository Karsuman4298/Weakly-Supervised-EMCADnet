# network.py  — MRPAnno version
# Key change: cls_head output expanded from 3 → 5 classes (CCG-5)

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from lib.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from lib.decoders import EMCAD


class EMCADNet(nn.Module):
    def __init__(self,
                 num_classes=1,
                 kernel_sizes=[1, 3, 5],
                 expansion_factor=2,
                 dw_parallel=True,
                 add=True,
                 lgag_ks=3,
                 activation='relu',
                 encoder='pvt_v2_b2',
                 pretrain=True,
                 pretrained_dir='./pretrained_pth/pvt/'):
        super(EMCADNet, self).__init__()

        # Grayscale → 3-channel converter
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # ── Backbone ─────────────────────────────────────────
        if encoder == 'pvt_v2_b0':
            self.backbone = pvt_v2_b0()
            path = pretrained_dir + '/pvt_v2_b0.pth'
            channels = [256, 160, 64, 32]
        elif encoder == 'pvt_v2_b1':
            self.backbone = pvt_v2_b1()
            path = pretrained_dir + '/pvt_v2_b1.pth'
            channels = [512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            path = pretrained_dir + '/pvt_v2_b2.pth'
            channels = [512, 320, 128, 64]
        elif encoder == 'pvt_v2_b3':
            self.backbone = pvt_v2_b3()
            path = pretrained_dir + '/pvt_v2_b3.pth'
            channels = [512, 320, 128, 64]
        elif encoder == 'pvt_v2_b4':
            self.backbone = pvt_v2_b4()
            path = pretrained_dir + '/pvt_v2_b4.pth'
            channels = [512, 320, 128, 64]
        elif encoder == 'pvt_v2_b5':
            self.backbone = pvt_v2_b5()
            path = pretrained_dir + '/pvt_v2_b5.pth'
            channels = [512, 320, 128, 64]
        elif encoder == 'resnet18':
            self.backbone = resnet18(pretrained=pretrain)
            channels = [512, 256, 128, 64]
        elif encoder == 'resnet34':
            self.backbone = resnet34(pretrained=pretrain)
            channels = [512, 256, 128, 64]
        elif encoder == 'resnet50':
            self.backbone = resnet50(pretrained=pretrain)
            channels = [2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            self.backbone = resnet101(pretrained=pretrain)
            channels = [2048, 1024, 512, 256]
        elif encoder == 'resnet152':
            self.backbone = resnet152(pretrained=pretrain)
            channels = [2048, 1024, 512, 256]
        else:
            print('Encoder not implemented! Defaulting to pvt_v2_b2.')
            self.backbone = pvt_v2_b2()
            path = pretrained_dir + '/pvt_v2_b2.pth'
            channels = [512, 320, 128, 64]

        if pretrain and 'pvt_v2' in encoder:
            save_model  = torch.load(path)
            model_dict  = self.backbone.state_dict()
            state_dict  = {k: v for k, v in save_model.items() if k in model_dict}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        print('Model %s created, param count: %d' %
              (encoder + ' backbone: ',
               sum(m.numel() for m in self.backbone.parameters())))

        # ── Decoder ──────────────────────────────────────────
        self.decoder = EMCAD(
            channels=channels,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            lgag_ks=lgag_ks,
            activation=activation
        )
        print('Model %s created, param count: %d' %
              ('EMCAD decoder: ',
               sum(m.numel() for m in self.decoder.parameters())))

        # ── Segmentation output heads ─────────────────────────
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

        # ── CCG-5: 5-class classification head ───────────────
        # Classes: 0=Ω_O, 1=Ω_Δ2, 2=P_mid strip, 3=Ω_Δ1, 4=Ω_I
        self.cls_head = nn.Sequential(
            nn.Conv2d(channels[3], channels[3] // 2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[3] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3] // 2, 5, kernel_size=1)   # ← 3→5
        )

        # ── CCL embedding head ────────────────────────────────
        embed_dim = 128
        self.embed_head = nn.Sequential(
            nn.Conv2d(channels[3], channels[3] // 2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[3] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3] // 2, embed_dim, kernel_size=1)
        )
        self.embed_dim  = embed_dim
        self.queue_size = 1024
        self.register_buffer(
            'neg_queue',
            F.normalize(torch.randn(embed_dim, self.queue_size), dim=0)
        )
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    # ─────────────────────────────────────────────────────────
    def forward(self, x, mode='test'):
        if x.size(1) == 1:
            x = self.conv(x)

        # Encoder
        x1, x2, x3, x4 = self.backbone(x)

        # Decoder
        dec_outs = self.decoder(x4, [x3, x2, x1])
        # dec_outs = [d4, d3, d2, d1]

        # Segmentation predictions
        p4 = F.interpolate(self.out_head4(dec_outs[0]),
                           scale_factor=32, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(dec_outs[1]),
                           scale_factor=16, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(dec_outs[2]),
                           scale_factor=8,  mode='bilinear', align_corners=False)
        p1 = F.interpolate(self.out_head1(dec_outs[3]),
                           scale_factor=4,  mode='bilinear', align_corners=False)

        if mode == 'test':
            return [p4, p3, p2, p1]

        # Training-only heads
        H, W = x.shape[2], x.shape[3]
        d1   = dec_outs[3]                             # finest feature map

        # CCG-5 classification (5 zones)
        cls_logits = self.cls_head(d1)
        cls_logits = F.interpolate(cls_logits, size=(H, W),
                                   mode='bilinear', align_corners=False)

        # CCL embeddings
        embeddings = self.embed_head(d1)
        embeddings = F.interpolate(embeddings, size=(H, W),
                                   mode='bilinear', align_corners=False)
        embeddings = F.normalize(embeddings, dim=1)

        return [p4, p3, p2, p1], cls_logits, embeddings

    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def update_queue(self, new_vecs):
        """Push new_vecs (N, D) into FIFO memory queue."""
        n   = new_vecs.shape[0]
        ptr = int(self.queue_ptr)
        end = ptr + n
        if end <= self.queue_size:
            self.neg_queue[:, ptr:end] = new_vecs.T
        else:
            first = self.queue_size - ptr
            self.neg_queue[:, ptr:]    = new_vecs[:first].T
            self.neg_queue[:, :n - first] = new_vecs[first:].T
        self.queue_ptr[0] = end % self.queue_size


if __name__ == '__main__':
    model = EMCADNet().cuda()
    x = torch.randn(1, 3, 352, 352).cuda()
    preds, cls, emb = model(x, mode='train')
    print([p.shape for p in preds])   # 4 × (1,1,352,352)
    print(cls.shape)                   # (1,5,352,352)
    print(emb.shape)                   # (1,128,352,352)