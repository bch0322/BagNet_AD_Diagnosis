# from __future__ import division
import torch
import torch.nn as nn
import setting_2 as fst
import math
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
import utils as ut
from torch.autograd import Variable
from torch.backends import cudnn
from PIL import Image
from sklearn.svm import SVC
import argparse

import setting as st
from sklearn.metrics import confusion_matrix
import numpy.matlib as mr
import nibabel as nib
import os
from modules import *

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.kernel_size = kernel_size
    
    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            tmp = self.downsample[0].weight.shape[1]
            self.torch_filter = torch.zeros(tmp, 1, self.kernel_size, self.kernel_size, self.kernel_size).cuda()
            self.torch_filter[:, :, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2] = 1
            residual = F.conv3d(residual, self.torch_filter, stride=self.stride, padding=0, groups = self.torch_filter.shape[0])
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out

class Residual_Conv(nn.Module):
    def __init__(self, config, block, layers = [1,1,1,1], strides=[2,2,2,1], kernel3 = [1,1,1,1], in_p = 1, f_out = [16, 32, 64, 128]):

        """ init """
        self.num_classes = config.num_classes
        self.widening_factor = 1
        self.inplanes = in_p * self.widening_factor
        f_out = [f_out[i] * self.widening_factor for i in range(len(f_out))]
        self.kernel = kernel3
        self.stride = strides
        super(Residual_Conv, self).__init__()

        """ filter bank """
        self.block_1  = BasicConv_Block(in_planes=1, out_planes=self.inplanes, kernel_size=3, stride=1, padding=0,
                                        dilation=1, groups=1, act_func='relu', bn=True, bias=False)

        """ bottleneck blocks """
        self.layer1 = self._make_layer(block, f_out[0], layers[0], stride=strides[0], kernel3=kernel3[0], concat_f=0, keep_inplane=False)
        self.layer2 = self._make_layer(block, f_out[1], layers[1], stride=strides[1], kernel3=kernel3[1], concat_f=0, keep_inplane=False)
        self.layer3 = self._make_layer(block, f_out[2], layers[2], stride=strides[2], kernel3=kernel3[2], concat_f=0, keep_inplane=False)
        self.layer4 = self._make_layer(block, f_out[3], layers[3], stride=strides[3], kernel3=kernel3[3], concat_f=0, keep_inplane=False)

        """ classifier """
        f_out_encoder = f_out[-1] * block.expansion
        self.classifier = nn.Sequential(
            nn.Conv3d(f_out_encoder , self.num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        """ initialize """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, concat_f = 2, keep_inplane = False):
        downsample = None
        tmp_inplane = self.inplanes
        self.inplanes = self.inplanes + concat_f

        if stride != 1 or self.inplanes != planes * block.expansion or kernel3 != 0:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        if keep_inplane == True:
            self.inplanes = tmp_inplane

        return nn.Sequential(*layers)

    def forward(self, datas, *args):
        """ feature extraction grid patches """
        if len(datas.shape) != 5:
            datas = datas[:, :, 0, :, :, :]
        else:
            datas = datas

        if datas.shape[1] != 1: # GM only
            datas = datas[:, 0:1]

        x_0 = datas

        """ cropoping """
        if self.training == True:
            dict_result = ut.data_augmentation(x_0)
            x_0 = dict_result['datas']

        """ downsampling the dataset """
        if fst.flag_downSample == True:
            x_0 = nn.AvgPool3d(kernel_size=3, stride=2)(x_0) # batch, 1, 127, 127, 127

        """ encoder """
        x_0 = self.block_1(x_0)
        x_0 = self.layer1(x_0) # 31, 31, 31
        x_0 = self.layer2(x_0) # 15, 15, 15
        x_0 = self.layer3(x_0) # 7, 7, 7
        x_0 = self.layer4(x_0) # 6, 6, 6

        """ calculate patch-level logit and attention """
        logitMap = self.classifier(x_0)

        """ aggregation """
        image_level_logit = nn.AvgPool3d(kernel_size=logitMap.size()[-3:], stride=1)(logitMap)

        """ flatten """
        image_level_logit = image_level_logit.view(image_level_logit.size(0), -1)

        dict_result = {
            "logits": image_level_logit, # batch, 2
            "Aux_logits": logitMap,  # batch, 2
            "attn_1" : None, # batch, 1, w, h, d
            "attn_2": None,  # batch, 1, w, h, d
            "logitMap" : logitMap, # batch, 2, w, h ,d
            "final_evidence" : None, # batch, 2, w, h, d
            "featureMaps" : [],
        }
        return dict_result

def bagNet9(config):
    """BagNet 9 """
    # 19 receptive field size with average pooling
    model = Residual_Conv(config, Bottleneck, layers=[1, 2, 3, 2], strides=[2, 2, 2, 1], kernel3=[1, 1, 0, 0], in_p=8, f_out=[16, 32, 64, 128])
    return model

def bagNet17(config):
    """BagNet 17 """
    # 35 receptive field size with average pooling
    model = Residual_Conv(config, Bottleneck, layers=[1, 2, 3, 2], strides=[2, 2, 2, 1], kernel3=[1, 1, 1, 0], in_p=8, f_out=[16, 32, 64, 128])
    return model

def bagNet33(config):
    """BagNet 33 """
    # 67 receptive field size with average pooling
    model = Residual_Conv(config, Bottleneck, layers=[1, 2, 3, 2], strides=[2, 2, 2, 1], kernel3=[1, 1, 1, 1], in_p=8, f_out=[16, 32, 64, 128])
    return model

