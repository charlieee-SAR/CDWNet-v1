import importlib
import importlib.util
import inspect
import os
import sys
import types
from contextlib import contextmanager

import torch.nn as nn

from ..builder import BACKBONES


@contextmanager
def _prepend_sys_path(path):
    sys.path.insert(0, path)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == path:
            sys.path.pop(0)
        elif path in sys.path:
            sys.path.remove(path)


@BACKBONES.register_module()
class ExternalFlashInternImage(nn.Module):
    """Adapter backbone that wraps FlashInternImage from external DCNv4 repo.

    This class delays importing external dependencies until instantiation, so
    existing configs that do not use this backbone are unaffected.
    """

    def __init__(self,
                 external_repo_root=None,
                 **kwargs):
        super().__init__()
        if external_repo_root is None:
            external_repo_root = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '../../../third_party'))
        self.external_repo_root = external_repo_root
        self._kwargs = kwargs
        self.backbone = self._build_external_backbone()

    def _build_external_backbone(self):
        try:
            self._ensure_dcnv4_module()
            mod = self._load_flash_module_directly()
            ext_cls = getattr(mod, 'FlashInternImage')
            return ext_cls(**self._kwargs)
        except Exception as e:
            raise RuntimeError(
                'Failed to build ExternalFlashInternImage. '
                'Please ensure:\n'
                '1) DCNv4 repo exists and path is correct\n'
                '2) DCNv4 python package is installed\n'
                '3) required deps (timm/mmcv compat) are installed\n'
                f'Current external_repo_root: {self.external_repo_root}\n'
                f'Original error: {repr(e)}')

    def _load_flash_module_directly(self):
        """Load flash_intern_image.py directly to avoid mmseg_custom package side effects."""
        flash_py = os.path.join(
            self.external_repo_root,
            'flash_internimage',
            'flash_intern_image.py')
        if not os.path.exists(flash_py):
            raise FileNotFoundError(f'Not found: {flash_py}')
        spec = importlib.util.spec_from_file_location('external_flash_intern_image', flash_py)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Failed to create import spec for: {flash_py}')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _ensure_dcnv4_module(self):
        """Provide a fallback `DCNv4` module when compiled package is unavailable.

        Fallback maps `DCNv4.DCNv4` to a pure-PyTorch DCNv3 implementation so
        FlashInternImage can still be instantiated for Phase-1 debugging/running.
        """
        try:
            importlib.import_module('DCNv4')
            return
        except Exception:
            pass

        with _prepend_sys_path(self.external_repo_root):
            dcnv3_mod = importlib.import_module('ops_dcnv3.modules.dcnv3')

        class DCNv4Compat(nn.Module):
            """Interface-compatible wrapper over DCNv3_pytorch."""
            def __init__(self,
                         channels=64,
                         kernel_size=3,
                         stride=1,
                         pad=1,
                         dilation=1,
                         group=4,
                         offset_scale=1.0,
                         dw_kernel_size=None,
                         center_feature_scale=False,
                         remove_center=False,
                         output_bias=True,
                         without_pointwise=False,
                         **kwargs):
                super().__init__()
                # Map to pure-pytorch DCNv3 implementation.
                self.impl = dcnv3_mod.DCNv3_pytorch(
                    channels=channels,
                    kernel_size=kernel_size,
                    dw_kernel_size=dw_kernel_size,
                    stride=stride,
                    pad=pad,
                    dilation=dilation,
                    group=group,
                    offset_scale=offset_scale,
                    center_feature_scale=center_feature_scale)

            def forward(self, x, shape=None, level_idx=0):
                # FlashInternImage calls core_op(x, shape, level_idx).
                # DCNv3_pytorch consumes NHWC tensor only.
                if x.dim() == 3:
                    if shape is None:
                        raise ValueError('shape is required when input is flattened (N, L, C).')
                    h, w = shape
                    n, l, c = x.shape
                    x_4d = x.view(n, h, w, c)
                    y_4d = self.impl(x_4d)
                    return y_4d.view(n, l, c)
                return self.impl(x)

            def _reset_parameters(self):
                if hasattr(self.impl, '_reset_parameters'):
                    self.impl._reset_parameters()

        shim = types.ModuleType('DCNv4')
        # Keep interface name expected by flash_intern_image: getattr(DCNv4, core_op)
        shim.DCNv4 = DCNv4Compat
        shim.DCNv3 = dcnv3_mod.DCNv3
        shim.DCNv3_pytorch = dcnv3_mod.DCNv3_pytorch
        sys.modules['DCNv4'] = shim

    def init_weights(self, pretrained=None):
        if hasattr(self.backbone, 'init_weights'):
            # Be explicit by signature to avoid masking real TypeError inside init.
            sig = inspect.signature(self.backbone.init_weights)
            if 'pretrained' in sig.parameters:
                return self.backbone.init_weights(pretrained=pretrained)
            return self.backbone.init_weights()
        return None

    def forward(self, x):
        return self.backbone(x)
