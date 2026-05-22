import importlib
import os
import sys
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
class ExternalVMambaBackbone(nn.Module):
    """Adapter backbone that wraps VMamba Backbone_VSSM from external repo."""

    def __init__(self,
                 external_repo_root=None,
                 **kwargs):
        super().__init__()
        if external_repo_root is None:
            external_repo_root = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '../../../third_party/vmamba_classification'))
        self.external_repo_root = external_repo_root
        self._kwargs = kwargs
        self.backbone = self._build_external_backbone()

    def _build_external_backbone(self):
        try:
            with _prepend_sys_path(self.external_repo_root):
                mod = importlib.import_module('models.vmamba')
                ext_cls = getattr(mod, 'Backbone_VSSM')
                return ext_cls(**self._kwargs)
        except Exception as e:
            raise RuntimeError(
                'Failed to build ExternalVMambaBackbone. '
                'Please ensure:\n'
                '1) VMamba repo exists and path is correct\n'
                '2) required deps (timm/fvcore/selective_scan build) are installed\n'
                f'Current external_repo_root: {self.external_repo_root}\n'
                f'Original error: {repr(e)}')

    def init_weights(self, pretrained=None):
        if hasattr(self.backbone, 'load_pretrained') and pretrained is not None:
            self.backbone.load_pretrained(pretrained)
        return None

    def forward(self, x):
        return self.backbone(x)
