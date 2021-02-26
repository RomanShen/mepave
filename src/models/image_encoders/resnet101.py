#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: romanshen 
@file: resnet101.py 
@time: 2021/02/26
@contact: xiangqing.shen@njust.edu.cn
"""

import logging

from torchvision.models import resnet101
from torch import nn

logger = logging.getLogger(__name__)


class ResNet101(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet101(pretrained=True)
        logger.info("currently using resnet101 for image encoding.")

    def forward(self, img):
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)

        region_features = self.resnet.layer4(x)
        global_features = self.resnet.avgpool(region_features)

        # [b, 2048, 7, 7], [b, 2048, 1, 1]
        global_features = global_features.squeeze()
        region_features = region_features.view((region_features.size(0), 2048, 49)).permute(0, 2, 1)

        # [b, 49, 2048], [b, 2048]
        return region_features, global_features
