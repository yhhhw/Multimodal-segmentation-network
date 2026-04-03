import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, num_classes, feature_channels):
        super(FPN, self).__init__()
        self.num_classes = num_classes
        self.feature_channels = feature_channels
       
        self.toplayer = nn.Conv2d(self.feature_channels[3], 256, 1, 1, 0) 
        
        self.latlayer1 = nn.Conv2d(self.feature_channels[2], 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(self.feature_channels[1], 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(self.feature_channels[0], 256, 1, 1, 0)
        
        self.seg_head = nn.Conv2d(256, self.num_classes, kernel_size=1)
        

    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def forward(self, input):
        x1, x2, x3, x4 = input
        p4 = self.toplayer(x4)
        p3 = self._upsample_add(p4, self.latlayer1(x3))
        p2 = self._upsample_add(p3, self.latlayer2(x2))
        p1 = self._upsample_add(p2, self.latlayer3(x1))
        
        out = self.seg_head(p1)
        return out

