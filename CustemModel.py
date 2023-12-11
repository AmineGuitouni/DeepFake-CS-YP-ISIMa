from extractParams import *
import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd
import torch.nn.functional as F
from models.upernet import *

class DoubleConvLayer(nn.Module):
    def __init__(self, in_channels , out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ConvStruct = nn.Sequential(
            #false bias 5ater zayd ki nest3mlou batchnormalize
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),

            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
    
    def forward(self, x):
        return self.ConvStruct(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3 , out_channels=1, features=[64,256]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Down = nn.ModuleList()
        for feature in features:
            self.Down.append(DoubleConvLayer(self.in_channels, feature))
            self.in_channels = feature

        self.Up = nn.ModuleList()
        self.Up.append(nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2))
        self.Up.append(nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2))

        self.finalConv = nn.Conv2d(features[0], self.out_channels, kernel_size=1)
    
    def forward(self, fpn , img):
        for down in self.Down:
            img = down(img)
            img = self.pool(img)

        img = fpn + img * 0.5
        
        for up in self.Up:
            img = up(img)

        return self.finalConv(img)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1),
            nn.AvgPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(x.shape[0],-1))
        return x

class CusstemModule(nn.Module):
    def __init__(self, main_model , train_part, cls_head = None, seg_head = None, train_main = False):
        super().__init__()
        self.useCustemSeg = False if seg_head == None else True
        self.train_part = train_part
        self.ppm_pooling = get_ppm_pooling(main_model)
        self.ppm_conv = get_ppm_conv(main_model)
        self.ppm_last_conv = get_ppm_head(main_model)
        self.img_size = get_decoder(main_model).img_size

        if train_part == "seg":
            self.fpn_in = get_decoder_fpn_in(main_model)
            self.fpn_out = get_decoder_fpn_out(main_model)
            self.seg_head = get_seg_head(main_model)  

            for param in self.parameters():
                param.requires_grad = train_main
            self.seg_UNet = UNet() if seg_head == None else seg_head

        elif train_part == "cls":
            for param in self.parameters():
                param.requires_grad = train_main

            self.cls_head = Classifier() if cls_head == None else cls_head
        
        else:
            self.fpn_in = get_decoder_fpn_in(main_model)
            self.fpn_out = get_decoder_fpn_out(main_model)
            self.seg_head = get_seg_head(main_model)

            for param in self.parameters():
                param.requires_grad = train_main
            self.seg_UNet = UNet() if seg_head == None else seg_head
            self.cls_head = Classifier() if cls_head == None else cls_head

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, conv_out, img):
        conv5 = conv_out[-1]
        input_size = conv5.size()
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

        out_cls = None
        out_seg = None

        if self.train_part == "cls" or self.train_part == "None":
            out_cls = self.cls_head(f)
        
        if self.train_part == "seg" or self.train_part == "None":
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
            out_seg = self.seg_UNet(fpn_feature_list[0] , img) if (self.train_part != "cls" and self.useCustemSeg ) or self.train_part == "seg" else out_seg
            out_seg = F.interpolate(out_seg, size=self.img_size, mode='bilinear', align_corners=False)
            
        return out_cls , out_seg