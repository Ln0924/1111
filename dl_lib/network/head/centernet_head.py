#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class SingleHead(nn.Module):

    def __init__(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class CenternetHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, cfg):
        super(CenternetHead, self).__init__()
        self.cls_head = SingleHead(
            64,
            cfg.MODEL.CENTERNET.NUM_CLASSES,
            bias_fill=True,
            bias_value=cfg.MODEL.CENTERNET.BIAS_VALUE,
        )
        self.wh_head = SingleHead(64, 2)
        self.reg_head = SingleHead(64, 2)
    #     self._freeze()
    #
    # def _freeze(self):
    #     for param in self.cls_head.parameters():
    #         param.requires_grad=False
    #     for param in self.wh_head.parameters():
    #         param.requires_grad=False
    #     for param in self.reg_head.parameters():
    #         param.requires_grad = False

    def forward(self, x):
        cls = self.cls_head(x)
        cls = torch.sigmoid(cls)  #B,C,256,256
        # print("cls",cls.shape)
        wh = self.wh_head(x)  #B,2,256,256
        # print(" wh", wh.shape)
        reg = self.reg_head(x)  #B,2,256,256
        # print("reg",reg.shape)
        pred = {
            'cls': cls,
            'wh': wh,
            'reg': reg,
          #  'fm':x
        }
        return pred
