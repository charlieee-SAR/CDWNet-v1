import torch

from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class DualBranchEncoderDecoder(EncoderDecoder):
    """Two-branch encoder-decoder segmentor for Optical/SAR inputs.

    Input ``img`` is expected to be channel-concatenated (N, C, H, W):
    [optical_channels..., sar_channels...].
    """

    def __init__(self,
                 backbone_optical,
                 backbone_sar,
                 decode_head,
                 optical_in_channels,
                 sar_in_channels=None,
                 fusion='concat',
                 auto_infer_decode_channels=True,
                 infer_input_size=(256, 256),
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(EncoderDecoder, self).__init__(init_cfg)
        self.backbone_optical = builder.build_backbone(backbone_optical)
        self.backbone_sar = builder.build_backbone(backbone_sar)
        self.optical_in_channels = optical_in_channels
        self.sar_in_channels = sar_in_channels
        self.fusion = fusion
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

    def _infer_decode_in_channels(self):
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

        with torch.no_grad():
            dummy = torch.zeros(1, total_ch, h, w)
            img_optical, img_sar = self._split_modalities(dummy)
            feats_optical = self.backbone_optical(img_optical)
            feats_sar = self.backbone_sar(img_sar)

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
        fused_feats = self._fuse_feats(feats_optical, feats_sar)
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
