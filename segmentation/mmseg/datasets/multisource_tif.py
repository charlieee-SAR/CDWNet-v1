import os.path as osp

import mmcv
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MultiSourceTifDataset(CustomDataset):
    """Dataset for paired Optical/SAR semantic segmentation.

    Expected directory layout:

    data_root/
      optical/
        training/*.tif
        validation/*.tif
      sar/
        training/*.tif
        validation/*.tif
      annotations/
        training/*.png
        validation/*.png
    """

    def __init__(self,
                 pipeline,
                 optical_dir,
                 sar_dir,
                 optical_suffix='.tif',
                 sar_suffix='.tif',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 file_client_args=dict(backend='disk')):
        # Resolve directories before calling super().__init__ because
        # CustomDataset.__init__ immediately triggers load_annotations().
        resolved_optical_dir = optical_dir
        resolved_sar_dir = sar_dir
        resolved_ann_dir = ann_dir
        resolved_split = split
        if data_root is not None:
            if not osp.isabs(resolved_optical_dir):
                resolved_optical_dir = osp.join(data_root, resolved_optical_dir)
            if not osp.isabs(resolved_sar_dir):
                resolved_sar_dir = osp.join(data_root, resolved_sar_dir)
            if resolved_ann_dir is not None and not osp.isabs(resolved_ann_dir):
                resolved_ann_dir = osp.join(data_root, resolved_ann_dir)
            if resolved_split is not None and not osp.isabs(resolved_split):
                resolved_split = osp.join(data_root, resolved_split)

        self.sar_dir = resolved_sar_dir
        self.sar_suffix = sar_suffix
        super().__init__(
            pipeline=pipeline,
            img_dir=resolved_optical_dir,
            img_suffix=optical_suffix,
            ann_dir=resolved_ann_dir,
            seg_map_suffix=seg_map_suffix,
            split=resolved_split,
            # Paths are already resolved above; avoid double-joining in
            # CustomDataset.__init__.
            data_root=None,
            test_mode=test_mode,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            classes=classes,
            palette=palette,
            gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
            file_client_args=file_client_args)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        img_infos = []
        sar_file_lookup = {}

        for sar_file in self.file_client.list_dir_or_file(
                dir_path=self.sar_dir,
                list_dir=False,
                suffix=self.sar_suffix,
                recursive=True):
            sar_key = sar_file[:-len(self.sar_suffix)]
            sar_file_lookup[sar_key] = sar_file

        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                stem = line.strip()
                if stem.endswith(img_suffix):
                    stem = stem[:-len(img_suffix)]
                optical_name = stem + img_suffix
                if stem not in sar_file_lookup:
                    raise FileNotFoundError(
                        f'Cannot find paired SAR file for "{stem}" under '
                        f'{self.sar_dir}')
                sar_name = sar_file_lookup[stem]
                img_info = dict(filename=optical_name, sar_filename=sar_name)
                if ann_dir is not None:
                    img_info['ann'] = dict(seg_map=stem + seg_map_suffix)
                img_infos.append(img_info)
        else:
            for optical_file in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                stem = optical_file[:-len(img_suffix)]
                if stem not in sar_file_lookup:
                    raise FileNotFoundError(
                        f'Cannot find paired SAR file for "{stem}" under '
                        f'{self.sar_dir}')
                sar_name = sar_file_lookup[stem]
                img_info = dict(filename=optical_file, sar_filename=sar_name)
                if ann_dir is not None:
                    img_info['ann'] = dict(seg_map=stem + seg_map_suffix)
                img_infos.append(img_info)

            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(
            f'Loaded {len(img_infos)} paired multi-source images',
            logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['optical_prefix'] = self.img_dir
        results['sar_prefix'] = self.sar_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map
