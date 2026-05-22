import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


@PIPELINES.register_module()
class LoadMultiSourceImageFromFile(object):
    """Load paired optical and SAR images, then concatenate on channel axis.

    The final output is stored in ``results['img']`` with shape (H, W, C_total).
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',
                 use_tifffile=True):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.use_tifffile = use_tifffile

    def _resolve_path(self, prefix_key, filename_key, results):
        prefix = results.get(prefix_key)
        filename = results['img_info'][filename_key]
        if prefix is not None:
            return osp.join(prefix, filename)
        return filename

    def _load_one(self, filename):
        is_tiff = filename.lower().endswith(('.tif', '.tiff'))
        if (self.use_tifffile or is_tiff) and HAS_TIFFFILE:
            img = tifffile.imread(filename)
        else:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if img is None:
            raise ValueError(f'Failed to load image: {filename}')

        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif img.ndim == 3 and img.shape[0] < img.shape[-1]:
            # Convert (C, H, W) to (H, W, C) if needed.
            img = np.transpose(img, (1, 2, 0))
        return img

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        optical_filename = self._resolve_path('optical_prefix', 'filename',
                                              results)
        sar_filename = self._resolve_path('sar_prefix', 'sar_filename', results)

        optical_img = self._load_one(optical_filename)
        sar_img = self._load_one(sar_filename)

        if optical_img.shape[:2] != sar_img.shape[:2]:
            raise ValueError(
                f'Optical/SAR shape mismatch: {optical_img.shape} vs '
                f'{sar_img.shape} | files: {optical_filename}, {sar_filename}')

        img = np.concatenate([optical_img, sar_img], axis=-1)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = f'{optical_filename}|{sar_filename}'
        results['ori_filename'] = (
            f"{results['img_info']['filename']}|"
            f"{results['img_info']['sar_filename']}")
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(to_float32={self.to_float32}, '
            f"color_type='{self.color_type}', "
            f"use_tifffile={self.use_tifffile}, "
            f"imdecode_backend='{self.imdecode_backend}')")
