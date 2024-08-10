# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.optim import OptimWrapper
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

# Whether to join loss_kernel
@MODELS.register_module()
class DSASegmentors(EncoderDecoder):
    def __init__(self, loss_kernel=False, **kwargs):
        super().__init__(**kwargs)
        self.loss_kernel = loss_kernel

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        losses = dict()
        if self.loss_kernel:
            loss_kernel = self.backbone.loss()
            losses.update(loss_kernel)

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses
    
