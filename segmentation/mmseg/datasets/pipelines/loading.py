# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES

# 添加 TIFF 支持
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',
                 use_tifffile=False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.use_tifffile = use_tifffile  # 新增
    
    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        
        # 【新增】确保文件路径存在，如果不存在且是相对路径，则报告详细错误
        if not osp.isabs(filename) and not osp.exists(filename):
            # 尝试相对于当前工作目录查找
            import os
            cwd = os.getcwd()
            possible_path = osp.join(cwd, filename)
            if osp.exists(possible_path):
                filename = possible_path
            else:
                raise FileNotFoundError(
                    f"找不到图像文件: {filename}\n"
                    f"当前工作目录: {cwd}\n"
                    f"img_prefix: {results.get('img_prefix')}\n"
                    f"img_info: {results['img_info']}"
                )

        # 判断是否使用tifffile
        is_tiff = filename.lower().endswith(('.tif', '.tiff'))
        
        if (self.use_tifffile or is_tiff) and HAS_TIFFFILE:
            # 使用tifffile读取多波段TIFF
            try:
                img = tifffile.imread(filename)
                
                # 确保是 (H, W, C) 格式
                if img.ndim == 2:
                    # 单通道图像
                    img = img[:, :, np.newaxis]
                elif img.ndim == 3:
                    # 检查是否是 (C, H, W) 格式
                    if img.shape[0] < img.shape[2] and img.shape[0] <= 6:
                        # 转换为 (H, W, C)
                        img = np.transpose(img, (1, 2, 0))
                
                # 确保数据类型正确
                if img.dtype == np.uint16:
                    # 16位图像归一化到0-255范围（可选）
                    # img = (img / 256).astype(np.uint8)
                    pass
                elif img.dtype == np.float32 or img.dtype == np.float64:
                    # 浮点图像
                    pass
                    
            except Exception as e:
                print(f"tifffile读取失败: {filename}, 错误: {e}")
                raise
        else:
            # 使用OpenCV读取（仅支持1-4通道）
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)

        if img is None:
            raise ValueError(f'Failed to load image: {filename}')

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # 设置pad_shape以避免某些操作报错
        results.setdefault('pad_shape', img.shape)
        # 设置scale_factor
        results.setdefault('scale_factor', 1.0)
        results['img_fields'] = ['img']
        
        return results
    # def __call__(self, results):
    #     """Call functions to load image and get image meta information.

    #     Args:
    #         results (dict): Result dict from :obj:`mmseg.CustomDataset`.

    #     Returns:
    #         dict: The dict contains loaded image and meta information.
    #     """

    #     if self.file_client is None:
    #         self.file_client = mmcv.FileClient(**self.file_client_args)

    #     if results.get('img_prefix') is not None:
    #         filename = osp.join(results['img_prefix'],
    #                             results['img_info']['filename'])
    #     else:
    #         filename = results['img_info']['filename']
    #     img_bytes = self.file_client.get(filename)
    #     img = mmcv.imfrombytes(
    #         img_bytes, flag=self.color_type, backend=self.imdecode_backend)
    #     if self.to_float32:
    #         img = img.astype(np.float32)

    #     results['filename'] = filename
    #     results['ori_filename'] = results['img_info']['filename']
    #     results['img'] = img
    #     results['img_shape'] = img.shape
    #     results['ori_shape'] = img.shape
    #     # Set initial values for default meta_keys
    #     results['pad_shape'] = img.shape
    #     results['scale_factor'] = 1.0
    #     num_channels = 1 if len(img.shape) < 3 else img.shape[2]
    #     results['img_norm_cfg'] = dict(
    #         mean=np.zeros(num_channels, dtype=np.float32),
    #         std=np.ones(num_channels, dtype=np.float32),
    #         to_rgb=False)
    #     return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
