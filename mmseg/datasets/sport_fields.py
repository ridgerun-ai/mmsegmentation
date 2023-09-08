# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SportFieldsDataset(BaseSegDataset):
    """SportFields dataset

    There is a single class which is 1: ``field``. 0 is considered
    background. ``img_suffix`` is fixed to '.jpg', and
    ``seg_map_suffix`` is fixed to '.png'.

    """
    METAINFO = dict(
        classes=(
            'background', 'field'),
        palette=[[0, 0, 0],
                 [128, 0, 128]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
