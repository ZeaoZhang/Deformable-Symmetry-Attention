# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from mmcv.transforms import to_tensor
import mmengine
import mmengine.fileio as fileio
import mmcv
import torch
import torch.nn.functional as F
import numpy as np
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmseg.datasets.transforms import PackSegInputs
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample
import pandas as pd
from tqdm import tqdm
from mmcv.image.geometric import _scale_size
import json
from mmcv.transforms import RandomResize, Resize
from mmcv.transforms.utils import cache_randomness
import numpy as np
import random

classes = ('background', 'malignant', 'benign')
palette = [[0, 0, 0], [0, 0, 255], [0, 255, 0]]


@DATASETS.register_module()
class MyDataset2(BaseSegDataset):
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.npy', seg_map_suffix='.png', **kwargs)


classes3 = ('background', 'left', 'right', 'heart')
palette3 = [[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]]

@DATASETS.register_module()
class MyDataset_checkXray2(BaseSegDataset):
  METAINFO = dict(classes = classes3, palette = palette3)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.npy', seg_map_suffix='.png', **kwargs)
    
@TRANSFORMS.register_module()
class LoadImageFromNPYFile(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = True,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        try:
            filename = results['img_path']

            # if self.file_client_args is not None:
            #     file_client = fileio.FileClient.infer_client(
            #         self.file_client_args, filename)
            #     img_bytes = file_client.get(filename)
            # else:
            #     img_bytes = fileio.get(
            #         filename, backend_args=self.backend_args)

            # img = mmcv.imfrombytes(
            #     img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            img = np.load(filename)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str 

@TRANSFORMS.register_module()
class MyContrastDistortion(BaseTransform):
    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5)):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range

    def convert(self,
                img: np.ndarray,
                alpha: int = 1,
                beta: int = 0) -> np.ndarray:
        """Multiple with alpha and add beat with clip.

        Args:
            img (np.ndarray): The input image.
            alpha (int): Image weights, change the contrast/saturation
                of the image. Default: 1
            beta (int): Image bias, change the brightness of the
                image. Default: 0

        Returns:
            np.ndarray: The transformed image.
        """

        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    # def brightness(self, img: np.ndarray) -> np.ndarray:
    #     """Brightness distortion.

    #     Args:
    #         img (np.ndarray): The input image.
    #     Returns:
    #         np.ndarray: Image after brightness change.
    #     """

    #     if random.randint(2):
    #         return self.convert(
    #             img,
    #             beta=random.uniform(-self.brightness_delta,
    #                                 self.brightness_delta))
    #     return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        """Contrast distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after contrast change.
        """

        if random.randint(0, 1):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        # img = self.brightness(img)
        img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}')
        return repr_str

