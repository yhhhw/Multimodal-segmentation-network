import torch
import torch.nn as nn
import torch.nn.functional as F


import timm

from fpn import FPN
from fusion import HybridAttention
from mscam import MSCAM


class MSFFNet(nn.Module):
    def __init__(self,
                 pretrain_weight_path='',
                 pretrained=False,
                 num_classes=6,
                 norm_layer=nn.BatchNorm2d
                 ):
        super().__init__()

        # swsl_resnet50
        # out0 256, 64, 64
        # out1 512, 32, 32
        # out2 1024, 16, 16
        # out3 2048, 8, 8
        self.rgb_backbone = timm.create_model('swsl_resnet50', features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained, pretrained_cfg_overlay=dict(file=pretrain_weight_path))
        self.dsm_backbone = timm.create_model('swsl_resnet50', features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained, pretrained_cfg_overlay=dict(file=pretrain_weight_path))
        self.dsm_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        
        # timm库API，得到每层特征通道大小
        encoder_channels = self.rgb_backbone.feature_info.channels()

        # HBFM
        self.fusion1 = HybridAttention(dim=encoder_channels[0], kernel_size=7, num_groups=2, num_heads=1, sr_ratio=8, d_embedding=4096, depth=1)
        self.fusion2 = HybridAttention(dim=encoder_channels[1], kernel_size=7, num_groups=2, num_heads=2, sr_ratio=4, d_embedding=1024, depth=1)
        self.fusion3 = HybridAttention(dim=encoder_channels[2], kernel_size=7, num_groups=3, num_heads=4, sr_ratio=2, d_embedding=256, depth=3)
        self.fusion4 = HybridAttention(dim=encoder_channels[3], kernel_size=7, num_groups=4, num_heads=8, sr_ratio=1, d_embedding=64, depth=1)
        
        # MSCAM
        self.mscam1 = MSCAM(dim=encoder_channels[0], head_num=1, window_size=7)
        self.mscam2 = MSCAM(dim=encoder_channels[1], head_num=1, window_size=7)
        self.mscam3 = MSCAM(dim=encoder_channels[2], head_num=1, window_size=7)
        self.mscam4 = MSCAM(dim=encoder_channels[3], head_num=1, window_size=7)

        # FPN head
        self.decoder = FPN(num_classes=num_classes, feature_channels=encoder_channels)
        
    def forward(self, x, y):
        h, w = x.size()[-2:]
        input_size = (x.size()[2], x.size()[3])

        rgb_res1, rgb_res2, rgb_res3, rgb_res4 = self.rgb_backbone(x)
        dsm_res1, dsm_res2, dsm_res3, dsm_res4 = self.dsm_backbone(y)

        # add fusion
        # res1 = rgb_res1 + dsm_res1
        # res2 = rgb_res2 + dsm_res2
        # res3 = rgb_res3 + dsm_res3
        # res4 = rgb_res4 + dsm_res4
        
        # fusion
        res1 = self.fusion1(rgb_res1, dsm_res1)
        res2 = self.fusion2(rgb_res2, dsm_res2)
        res3 = self.fusion3(rgb_res3, dsm_res3)
        res4 = self.fusion4(rgb_res4, dsm_res4)
        
        # MSCAM
        res1 = self.mscam1(res1)
        res2 = self.mscam2(res2)
        res3 = self.mscam3(res3)
        res4 = self.mscam4(res4)

        # decoder
        out = self.decoder([res1, res2, res3, res4])
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)  # final output

        return out
        

if __name__=='__main__':
    device = torch.device("cuda:0")
    x = torch.rand(2, 3, 256, 256)
    y = torch.rand(2, 1, 256, 256)

    model = MSFFNet(num_classes=6).to(device)
    out = model(x.to(device), y.to(device))
    print("out shape:", out.shape)

    print('done')
