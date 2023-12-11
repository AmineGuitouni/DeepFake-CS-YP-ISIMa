# ref: https://github.com/CSAILVision/unifiedparsing/blob/master/models/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_fwd

from utils.lib.nn import SynchronizedBatchNorm2d
from utils.lib.nn.prroi_pool.prroi_pool import PrRoIPool2D
from . import resnet

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

def rgb2gray(rgb):
    # changed from bgr to rgb
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = torch.unsqueeze(gray, 1)
    return gray

class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv2d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)


    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, n_classes, img_size, bayar):
        super(EncoderDecoder, self).__init__()

        self.bayar = bayar

        fpn_inplanes = (256, 512, 1024, 2048)

        self.encoder = Resnet(resnet.__dict__['resnet50'](pretrained=True))

        self.decoder = UPerNet(n_classes=n_classes,
                img_size=img_size,
                fc_dim=2048,
                use_softmax=False,
                fpn_dim=256,
                fpn_inplanes=fpn_inplanes)

        if (self.bayar):
            self.bayar_conv = BayarConv2d(in_channels=1, out_channels=3, padding=2)
            self.noise_encoder = Resnet(resnet.__dict__['resnet50'](pretrained=True))

            self.convs = []
            for fpn in fpn_inplanes:
                fpn_in = 2 * fpn if self.bayar else fpn
                self.convs.append(nn.Conv2d(fpn_in, fpn, kernel_size=3, padding=1))
            self.convs = nn.ModuleList(self.convs)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, img):
        fea_img = self.encoder(img, return_feature_maps=True)

        if (self.bayar):
            noise = self.bayar_conv(rgb2gray(img))
            fea_noise = self.noise_encoder(noise, return_feature_maps=True)

            x = []
            for i, conv in enumerate(self.convs):
                x.append(conv(torch.cat([fea_img[i], fea_noise[i]], dim = 1)))
        else:
            x = fea_img

        out_seg = self.decoder(x,img)
        return out_seg

class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

class UPerNet(nn.Module):
    def __init__(self, n_classes, img_size, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256,512,1024,2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.img_size = (img_size, img_size)

        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            # we use the feature map size instead of input image size, so down_scale = 1.0
            self.ppm_pooling.append(PrRoIPool2D(scale, scale, 1.))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_fusion = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)

        self.n_classes = n_classes

        # input: PPM out, input_dim: fpn_dim
        self.cls_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fpn_dim, self.n_classes, kernel_size=1, bias=True)
        )


        # input: FPN_2 (P2), input_dim: fpn_dim
        self.seg_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, self.n_classes, kernel_size=1, bias=True)
        )

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, conv_out, img):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        roi = [] # fake rois, just used for pooling
        for i in range(input_size[0]): # batch size
            roi.append(torch.Tensor([i, 0, 0, input_size[3], input_size[2]]).view(1, -1)) # b, x0, y0, x1, y1
        roi = torch.cat(roi, dim=0).type_as(conv5)
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.interpolate(
                pool_scale(conv5, roi.detach()),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)
        
        out_cls = self.cls_head(f).squeeze(3).squeeze(2)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = F.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse() # [P2 - P5]

        out_seg = self.seg_head(fpn_feature_list[0])
        out_seg = F.interpolate(out_seg, size=self.img_size, mode='bilinear', align_corners=False)

        return out_cls, out_seg