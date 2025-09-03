#!/usr/bin/python3
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn

from dl_lib.layers import DeformConvWithOff, ModulatedDeformConvWithOff


class DeconvLayer(nn.Module):

    def __init__(
        self, in_planes,
        out_planes, deconv_kernel,
        deconv_stride=2, deconv_pad=1,
        deconv_out_pad=0, modulate_deform=True,
    ):
        super(DeconvLayer, self).__init__()
        if modulate_deform:
            self.dcn = ModulatedDeformConvWithOff(
                in_planes, out_planes,
                kernel_size=3, deformable_groups=1,
            )
        else:
            self.dcn = DeformConvWithOff(
                in_planes, out_planes,
                kernel_size=3, deformable_groups=1,
            )

        self.dcn_bn = nn.BatchNorm2d(out_planes)
        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=deconv_kernel,
            stride=deconv_stride, padding=deconv_pad,
            output_padding=deconv_out_pad,
            bias=False,
        )
        self._deconv_init()
        self.up_bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.dcn(x)
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

class ConvLayer(nn.Module):

    def __init__(
        self, in_planes,
        out_planes, modulate_deform=True,
    ):
        super(ConvLayer, self).__init__()
        if modulate_deform:
            self.dcn1 = ModulatedDeformConvWithOff(
                in_planes, in_planes//2,
                kernel_size=3, deformable_groups=1,
            )
            self.dcn2 = ModulatedDeformConvWithOff(
                in_planes//2, out_planes,
                kernel_size=3, deformable_groups=1,
            )
        else:
            self.dcn1 = DeformConvWithOff(
                in_planes, in_planes//2,
                kernel_size=3, deformable_groups=1,
            )
            self.dcn2 = ModulatedDeformConvWithOff(
                in_planes//2, out_planes,
                kernel_size=3, deformable_groups=1,
            )

        self.dcn_bn1 = nn.BatchNorm2d(in_planes//2)

        self.dcn_bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dcn1(x)
        x = self.dcn_bn1(x)
        x = self.relu(x)
        x = self.dcn2(x)
        x = self.dcn_bn2(x)
        x = self.relu(x)
        return x


class CenternetDeconv1(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, cfg):
        super(CenternetDeconv1, self).__init__()
        # modify into config
        channels = cfg.MODEL.CENTERNET.DECONV_CHANNEL   #[256, 256, 256, 256, 256, 64]
        deconv_kernel = cfg.MODEL.CENTERNET.DECONV_KERNEL  #[4, 4, 4, 4]
        modulate_deform = cfg.MODEL.CENTERNET.MODULATE_DEFORM  #True
        # channels = [256, 256, 256, 256, 256, 64]
        # deconv_kernel = [4, 4, 4, 4]
        # modulate_deform =True
        self.pool = nn.MaxPool2d(2, 2)
        self.deconv0 = DeconvLayer(
            channels[0], channels[1],
            deconv_kernel=deconv_kernel[0],
            modulate_deform=modulate_deform,
        )
        self.deconv1 = DeconvLayer(
            channels[1]*2, channels[2],
            deconv_kernel=deconv_kernel[1],
            modulate_deform=modulate_deform,
        )
        self.deconv2 = DeconvLayer(
            channels[2]*2, channels[3],
            deconv_kernel=deconv_kernel[2],
            modulate_deform=modulate_deform,
        )
        self.deconv3 = DeconvLayer(
            channels[3]*2, channels[4],
            deconv_kernel=deconv_kernel[3],
            modulate_deform=modulate_deform,
        )
        self.conv4 = ConvLayer(
            channels[4] * 2, channels[5],
            modulate_deform=modulate_deform,
        )
        # self._freeze()

    # def _freeze(self):
    #     for param in self.deconv0.parameters():
    #         param.requires_grad=False
    #     for param in self.deconv1.parameters():
    #         param.requires_grad=False
    #     for param in self.deconv2.parameters():
    #         param.requires_grad=False
    #     for param in self.deconv3.parameters():
    #         param.requires_grad=False
    #     for param in self.conv4.parameters():
    #         param.requires_grad=False

    def forward(self, x):
        res2,res3,res4,res5 = x["res2"],x["res3"],x["res4"],x["res5"]
        center = self.deconv0(self.pool(res5))  #256,32,32
        dec5 = self.deconv1(torch.cat([center, res5], 1))  #256,64,64
        dec4 = self.deconv2(torch.cat([dec5, res4], 1))  #256,128,128
        dec3 = self.deconv3(torch.cat([dec4, res3], 1))  #256,256,256
        dec4 = self.conv4(torch.cat([dec3, res2], 1))  #64,256,256

        return dec4


# if __name__=="__main__":
#     from torchsummary import summary
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = CenternetDeconv1().to(device)
#     # [(256, 256, 256),(256, 128,128),(256, 64, 64),(256, 32, 32)]
#     summary(model, (512, 256, 256))
