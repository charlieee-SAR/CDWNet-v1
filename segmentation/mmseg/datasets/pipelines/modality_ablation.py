import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class ForceSingleModality(object):
    """Force one modality by zeroing channels of the other modality.

    This transform is designed for modality ablation experiments:
    - mode='optical': keep optical channels, zero SAR channels
    - mode='sar': keep SAR channels, zero optical channels
    """

    def __init__(self,
                 mode='optical',
                 optical_channels=3,
                 sar_channels=3,
                 fill_value=0.0):
        if mode not in ('optical', 'sar'):
            raise ValueError(f'Unsupported mode: {mode}')
        self.mode = mode
        self.optical_channels = int(optical_channels)
        self.sar_channels = int(sar_channels)
        self.fill_value = float(fill_value)

    def __call__(self, results):
        img = results['img']
        if img.ndim != 3:
            raise ValueError(f'Expected HWC image, got shape {img.shape}')

        total = self.optical_channels + self.sar_channels
        if img.shape[2] < total:
            raise ValueError(
                f'Image channels({img.shape[2]}) < expected total({total}). '
                f'optical={self.optical_channels}, sar={self.sar_channels}')

        if self.mode == 'optical':
            img[:, :, self.optical_channels:total] = self.fill_value
        else:
            img[:, :, :self.optical_channels] = self.fill_value

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        results['img'] = img
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(mode={self.mode}, '
                f'optical_channels={self.optical_channels}, '
                f'sar_channels={self.sar_channels}, '
                f'fill_value={self.fill_value})')


@PIPELINES.register_module()
class SelectSingleModality(object):
    """Keep only one modality channels and drop the other channels.

    This transform converts a concatenated multi-source tensor (optical+sar)
    into a true single-modality tensor:
    - mode='optical': output C=optical_channels
    - mode='sar': output C=sar_channels
    """

    def __init__(self, mode='optical', optical_channels=3, sar_channels=3):
        if mode not in ('optical', 'sar'):
            raise ValueError(f'Unsupported mode: {mode}')
        self.mode = mode
        self.optical_channels = int(optical_channels)
        self.sar_channels = int(sar_channels)

    def __call__(self, results):
        img = results['img']
        if img.ndim != 3:
            raise ValueError(f'Expected HWC image, got shape {img.shape}')

        total = self.optical_channels + self.sar_channels
        if img.shape[2] < total:
            raise ValueError(
                f'Image channels({img.shape[2]}) < expected total({total}). '
                f'optical={self.optical_channels}, sar={self.sar_channels}')

        if self.mode == 'optical':
            img = img[:, :, :self.optical_channels]
        else:
            start = self.optical_channels
            end = self.optical_channels + self.sar_channels
            img = img[:, :, start:end]

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(mode={self.mode}, '
                f'optical_channels={self.optical_channels}, '
                f'sar_channels={self.sar_channels})')
