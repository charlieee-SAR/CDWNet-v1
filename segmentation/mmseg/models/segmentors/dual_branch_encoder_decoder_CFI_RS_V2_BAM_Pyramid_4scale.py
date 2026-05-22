import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


class CFI_RS_V2_FusionBlockMulti(nn.Module):
    """CFI-RS V2: asymmetric bidirectional gated cross-frequency interaction."""

    def __init__(self, level_indices=(1, 2), pool_kernel=3, gate_kernel=3):
        super().__init__()
        self.level_indices = tuple(sorted(set(level_indices)))
        self.pool_kernel = pool_kernel
        self.gate_kernel = gate_kernel
        self.level_pos = {idx: pos for pos, idx in enumerate(self.level_indices)}

        n_levels = max(1, len(self.level_indices))
        self.gate_o2s_refine = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=gate_kernel, padding=gate_kernel // 2, bias=True)
            for _ in range(n_levels)
        ])
        self.gate_s2o_refine = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=gate_kernel, padding=gate_kernel // 2, bias=True)
            for _ in range(n_levels)
        ])
        # Direction-specific learnable strengths (asymmetric parameterization).
        self.alpha_o2s = nn.Parameter(torch.ones(n_levels))
        self.beta_o2s = nn.Parameter(torch.ones(n_levels))
        self.alpha_s2o = nn.Parameter(torch.ones(n_levels))
        self.beta_s2o = nn.Parameter(torch.ones(n_levels))

    def _low_high(self, x):
        low = F.avg_pool2d(
            x,
            kernel_size=self.pool_kernel,
            stride=1,
            padding=self.pool_kernel // 2)
        high = x - low
        return low, high

    def _apply_one_level(self, fo, fs, level_idx):
        level_pos = self.level_pos[level_idx]
        low_o, high_o = self._low_high(fo)
        low_s, high_s = self._low_high(fs)

        gate_o2s_raw = torch.mean(high_o, dim=1, keepdim=True)
        gate_s2o_raw = torch.mean(low_s, dim=1, keepdim=True)
        gate_o2s = torch.sigmoid(self.gate_o2s_refine[level_pos](gate_o2s_raw))
        gate_s2o = torch.sigmoid(self.gate_s2o_refine[level_pos](gate_s2o_raw))

        # Cross-band source maps (single-channel, then broadcast to all channels).
        src_o2s = torch.tanh(gate_o2s_raw)
        src_s2o = torch.tanh(gate_s2o_raw)

        low_s_guided = (
            low_s * (1.0 + self.alpha_o2s[level_pos] * gate_o2s)
            + self.beta_o2s[level_pos] * src_o2s)
        high_o_guided = (
            high_o * (1.0 + self.alpha_s2o[level_pos] * gate_s2o)
            + self.beta_s2o[level_pos] * src_s2o)

        fo_new = low_o + high_o_guided
        fs_new = low_s_guided + high_s
        return fo_new, fs_new

    def forward(self, feats_optical, feats_sar):
        feats_optical = list(feats_optical)
        feats_sar = list(feats_sar)

        for level_idx in self.level_indices:
            if level_idx >= len(feats_optical) or level_idx >= len(feats_sar):
                continue
            fo, fs = feats_optical[level_idx], feats_sar[level_idx]
            fo_new, fs_new = self._apply_one_level(fo, fs, level_idx)
            feats_optical[level_idx] = fo_new
            feats_sar[level_idx] = fs_new

        return tuple(feats_optical), tuple(feats_sar)


class MS_BAM_Pyramid4Scale(nn.Module):
    """MS-BAM Pyramid: all-scale boundary enhancement with cross-scale prior fusion."""

    def __init__(self, level_indices=(0, 1, 2, 3), edge_kernel=3, refine_kernel=3):
        super().__init__()
        self.level_indices = tuple(sorted(set(level_indices)))
        self.edge_kernel = edge_kernel
        self.refine_kernel = refine_kernel

        n_levels = max(1, len(self.level_indices))
        self.level_pos = {idx: pos for pos, idx in enumerate(self.level_indices)}

        # Per-level boundary fusion (optical edge + sar edge -> local boundary map)
        self.local_boundary_fuse = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=1, bias=True) for _ in range(n_levels)
        ])
        # Per-level refinement before re-injection.
        self.level_refine = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=refine_kernel, padding=refine_kernel // 2, bias=True)
            for _ in range(n_levels)
        ])
        # Pyramid fusion after resizing all boundary maps to a shared reference scale.
        self.pyramid_fuse = nn.Sequential(
            nn.Conv2d(n_levels, n_levels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_levels, 1, kernel_size=1, bias=True)
        )

    def _edge_response(self, x):
        low = F.avg_pool2d(
            x,
            kernel_size=self.edge_kernel,
            stride=1,
            padding=self.edge_kernel // 2)
        edge = torch.abs(x - low)
        return torch.mean(edge, dim=1, keepdim=True)

    def forward(self, feats_optical, feats_sar, fused_feats):
        fused_feats = list(fused_feats)

        local_maps = []
        valid_levels = []
        for level_idx in self.level_indices:
            if (level_idx >= len(feats_optical) or level_idx >= len(feats_sar)
                    or level_idx >= len(fused_feats)):
                continue
            pos = self.level_pos[level_idx]
            edge_o = self._edge_response(feats_optical[level_idx])
            edge_s = self._edge_response(feats_sar[level_idx])
            # SAR-friendly smoothing before fusion.
            edge_s = F.avg_pool2d(edge_s, kernel_size=3, stride=1, padding=1)
            local = torch.sigmoid(
                self.local_boundary_fuse[pos](torch.cat([edge_o, edge_s], dim=1)))
            local_maps.append(local)
            valid_levels.append(level_idx)

        if not valid_levels:
            return tuple(fused_feats)

        # Build a shared multi-scale boundary prior on the highest resolution among selected levels.
        ref_level = min(valid_levels)
        ref_h, ref_w = fused_feats[ref_level].shape[2:]
        aligned_maps = []
        for local in local_maps:
            aligned_maps.append(
                F.interpolate(local, size=(ref_h, ref_w), mode='bilinear', align_corners=False))
        pyramid = torch.cat(aligned_maps, dim=1)
        global_prior = torch.sigmoid(self.pyramid_fuse(pyramid))

        # Re-inject both local and global boundary priors at each selected level.
        for local, level_idx in zip(local_maps, valid_levels):
            pos = self.level_pos[level_idx]
            global_lvl = F.interpolate(
                global_prior,
                size=fused_feats[level_idx].shape[2:],
                mode='bilinear',
                align_corners=False)
            local_refined = torch.sigmoid(self.level_refine[pos](local))
            attn = torch.clamp(0.5 * local_refined + 0.5 * global_lvl, min=0.0, max=1.0)
            fused_feats[level_idx] = fused_feats[level_idx] * (1.0 + attn)

        return tuple(fused_feats)


@SEGMENTORS.register_module()
class DualBranchEncoderDecoder_CFI_RS_V2_BAM_Pyramid_4Scale(EncoderDecoder):
    """Dual-branch segmentor with CFI-RS V2 and pyramid MS-BAM on all 4 scales."""

    def __init__(self,
                 backbone_optical,
                 backbone_sar,
                 decode_head,
                 optical_in_channels,
                 sar_in_channels=None,
                 fusion='concat',
                 cfi_rs=None,
                 ms_bam=None,
                 auto_infer_decode_channels=True,
                 infer_input_size=(256, 256),
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(EncoderDecoder, self).__init__(init_cfg)
        self.backbone_optical_cfg = backbone_optical.copy()
        self.backbone_sar_cfg = backbone_sar.copy()
        self.backbone_optical = builder.build_backbone(backbone_optical)
        self.backbone_sar = builder.build_backbone(backbone_sar)
        self.optical_in_channels = optical_in_channels
        self.sar_in_channels = sar_in_channels
        self.fusion = fusion

        self.cfi_rs_cfg = cfi_rs
        self.with_cfi_rs = cfi_rs is not None
        if self.with_cfi_rs:
            level_indices = cfi_rs.get('level_indices', cfi_rs.get('level_idx', [1, 2]))
            if isinstance(level_indices, int):
                level_indices = [level_indices]
            self.cfi_rs = CFI_RS_V2_FusionBlockMulti(
                level_indices=level_indices,
                pool_kernel=cfi_rs.get('pool_kernel', 3),
                gate_kernel=cfi_rs.get('gate_kernel', 3))

        self.ms_bam_cfg = ms_bam
        self.with_ms_bam = ms_bam is not None
        if self.with_ms_bam:
            level_indices = ms_bam.get('level_indices', ms_bam.get('level_idx', [1, 2]))
            if isinstance(level_indices, int):
                level_indices = [level_indices]
            self.ms_bam = MS_BAM_Pyramid4Scale(
                level_indices=level_indices,
                edge_kernel=ms_bam.get('edge_kernel', 3),
                refine_kernel=ms_bam.get('refine_kernel', 3))

        self.auto_infer_decode_channels = auto_infer_decode_channels
        self.infer_input_size = infer_input_size

        decode_head = decode_head.copy()
        if self.auto_infer_decode_channels:
            inferred = self._infer_decode_in_channels()
            decode_head['in_channels'] = inferred

        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def _infer_single_backbone_channels(self, cfg):
        backbone_type = cfg.get('type')
        if backbone_type == 'ExternalFlashInternImage':
            base = int(cfg.get('channels', 64))
            return [base, base * 2, base * 4, base * 8]
        if backbone_type == 'ExternalVMambaBackbone':
            base = int(cfg.get('dims', 96))
            return [base, base * 2, base * 4, base * 8]
        if backbone_type == 'MixVisionTransformer':
            embed_dims = int(cfg.get('embed_dims', 32))
            num_heads = list(cfg.get('num_heads', [1, 2, 5, 8]))
            return [int(embed_dims * head) for head in num_heads]
        if backbone_type in ('ResNet', 'ResNetV1c', 'ResNetV1d'):
            depth = int(cfg.get('depth', 50))
            if depth in (18, 34):
                return [64, 128, 256, 512]
            return [256, 512, 1024, 2048]
        return None

    def _infer_decode_in_channels(self):
        optical_channels = self._infer_single_backbone_channels(self.backbone_optical_cfg)
        sar_channels = self._infer_single_backbone_channels(self.backbone_sar_cfg)
        if optical_channels is not None and sar_channels is not None:
            if len(optical_channels) != len(sar_channels):
                raise ValueError(
                    f'Feature levels mismatch during cfg inference: '
                    f'optical={len(optical_channels)} vs sar={len(sar_channels)}')
            inferred = []
            for co, cs in zip(optical_channels, sar_channels):
                if self.fusion == 'concat':
                    inferred.append(int(co + cs))
                elif self.fusion == 'add':
                    if co != cs:
                        raise ValueError(
                            f'Cannot use add fusion with channel mismatch: {co} vs {cs}')
                    inferred.append(int(co))
                else:
                    raise ValueError(
                        f'Unsupported fusion type: {self.fusion}. '
                        f"Use 'concat' or 'add'.")
            return inferred

        h, w = self.infer_input_size
        if self.sar_in_channels is None:
            raise ValueError(
                'auto_infer_decode_channels=True requires sar_in_channels to '
                'be explicitly set.')
        sar_ch = self.sar_in_channels
        total_ch = self.optical_in_channels + sar_ch
        if total_ch <= 0:
            raise ValueError(
                'Invalid channels for auto inference: '
                f'optical={self.optical_in_channels}, sar={self.sar_in_channels}')

        def _module_device(module):
            for tensor in module.parameters():
                return tensor.device
            for tensor in module.buffers():
                return tensor.device
            return torch.device('cpu')

        opt_device = _module_device(self.backbone_optical)
        sar_device = _module_device(self.backbone_sar)
        infer_device = torch.device('cuda:0') if torch.cuda.is_available() else opt_device
        opt_training = self.backbone_optical.training
        sar_training = self.backbone_sar.training

        try:
            self.backbone_optical.eval()
            self.backbone_sar.eval()
            if opt_device != infer_device:
                self.backbone_optical.to(infer_device)
            if sar_device != infer_device:
                self.backbone_sar.to(infer_device)

            with torch.no_grad():
                dummy = torch.zeros(1, total_ch, h, w, device=infer_device)
                img_optical, img_sar = self._split_modalities(dummy)
                feats_optical = self.backbone_optical(img_optical)
                feats_sar = self.backbone_sar(img_sar)
        finally:
            if opt_device != infer_device:
                self.backbone_optical.to(opt_device)
            if sar_device != infer_device:
                self.backbone_sar.to(sar_device)
            self.backbone_optical.train(opt_training)
            self.backbone_sar.train(sar_training)

        if len(feats_optical) != len(feats_sar):
            raise ValueError(
                f'Feature levels mismatch during inference: '
                f'optical={len(feats_optical)} vs sar={len(feats_sar)}')

        inferred = []
        for fo, fs in zip(feats_optical, feats_sar):
            if self.fusion == 'concat':
                inferred.append(int(fo.shape[1] + fs.shape[1]))
            elif self.fusion == 'add':
                if fo.shape[1] != fs.shape[1]:
                    raise ValueError(
                        f'Cannot use add fusion with channel mismatch: '
                        f'{fo.shape[1]} vs {fs.shape[1]}')
                inferred.append(int(fo.shape[1]))
            else:
                raise ValueError(
                    f'Unsupported fusion type: {self.fusion}. '
                    f"Use 'concat' or 'add'.")
        return inferred

    def _split_modalities(self, img):
        total_channels = img.shape[1]
        optical_ch = self.optical_in_channels
        sar_ch = self.sar_in_channels
        if sar_ch is None:
            sar_ch = total_channels - optical_ch
        if optical_ch + sar_ch != total_channels:
            raise ValueError(
                f'Input channel mismatch: got {total_channels}, expected '
                f'{optical_ch}+{sar_ch}')
        img_optical = img[:, :optical_ch, :, :]
        img_sar = img[:, optical_ch:optical_ch + sar_ch, :, :]
        return img_optical, img_sar

    def _fuse_feats(self, feats_optical, feats_sar):
        if len(feats_optical) != len(feats_sar):
            raise ValueError(
                f'Feature levels mismatch: optical {len(feats_optical)} vs '
                f'sar {len(feats_sar)}')

        fused = []
        for fo, fs in zip(feats_optical, feats_sar):
            if self.fusion == 'concat':
                fused.append(torch.cat([fo, fs], dim=1))
            elif self.fusion == 'add':
                if fo.shape[1] != fs.shape[1]:
                    raise ValueError(
                        f'Cannot add features with channel mismatch: '
                        f'{fo.shape[1]} vs {fs.shape[1]}')
                fused.append(fo + fs)
            else:
                raise ValueError(
                    f'Unsupported fusion type: {self.fusion}. '
                    f"Use 'concat' or 'add'.")
        return fused

    def extract_feat(self, img):
        img_optical, img_sar = self._split_modalities(img)
        feats_optical = self.backbone_optical(img_optical)
        feats_sar = self.backbone_sar(img_sar)
        if self.with_cfi_rs:
            feats_optical, feats_sar = self.cfi_rs(feats_optical, feats_sar)
        fused_feats = self._fuse_feats(feats_optical, feats_sar)
        if self.with_ms_bam:
            fused_feats = self.ms_bam(feats_optical, feats_sar, fused_feats)
        if self.with_neck:
            fused_feats = self.neck(fused_feats)
        return fused_feats

    def encode_decode(self, img, img_metas):
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg):
        x = self.extract_feat(img)
        losses = dict()
        losses.update(
            self._decode_head_forward_train(x, img_metas, gt_semantic_seg))
        if self.with_auxiliary_head:
            losses.update(
                self._auxiliary_head_forward_train(x, img_metas,
                                                   gt_semantic_seg))
        return losses

    def forward_dummy(self, img):
        return self.encode_decode(img, None)
