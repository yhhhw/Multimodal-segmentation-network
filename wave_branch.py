import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)
    return out


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) 


class DWT(nn.Module): 
    def __init__(self, channels, kernel_size=3):
        super(DWT, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding="same", groups=channels, bias=False)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding="same", groups=channels, bias=False)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding="same", groups=channels, bias=False)
        self.conv9 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding="same", groups=channels, bias=False)
        
        self.conv_cat = nn.Conv2d(channels*4, channels, kernel_size=3, padding=1, groups=channels, bias=False)

    def forward(self, x):
        
        aa =  DWTForward(J=1, mode='zero', wave='db3').cuda(device=0)
        yl, yh = aa(x)

        yh_out = yh[0]
        ylh = yh_out[:,:,0,:,:]
        yhl = yh_out[:,:,1,:,:]
        yhh = yh_out[:,:,2,:,:]

        conv_rec1 = self.conv1(yl)
        conv_rec5 = self.conv5(ylh)
        conv_rec7 = self.conv7(yhl)
        conv_rec9 = self.conv9(yhh)

        cat_all = torch.stack((conv_rec5, conv_rec7, conv_rec9),dim=2)
        rec_yh = []
        rec_yh.append(cat_all)


        ifm = DWTInverse(wave='db3', mode='zero').cuda(device=0)
        Y = ifm((conv_rec1, rec_yh))

        return Y


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ChannelMixer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., reduction=4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / reduction)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.sig(x)
        
        return x


class WAMAndChannelMixer(nn.Module):
  
    def __init__(self, d_model, d_k, d_v, h, dff=2048, dropout=.1):
        super(WAMAndChannelMixer, self).__init__()

        self.s = DWT(channels=1, kernel_size=9)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.token_mixer1 = Pooling(pool_size=3)
        self.token_mixer2 = Pooling(pool_size=3)
        self.token_mixer3 = Pooling(pool_size=3)

        self.cm = ChannelMixer(d_model, d_model * 4)

    def forward(self, x, H, W):
        x = x.unsqueeze(1)
        x = self.s(x)
        
        x = x.squeeze(1)
        
        att1 = self.token_mixer1(x)
        att2 = self.token_mixer2(att1)
        att3 = self.token_mixer3(att2)
        
        att = (att1 + att2 + att3) / 3  
        att = self.dropout(att)
        att = att.permute(0, 2, 1)
        out = self.cm(att, H, W)
        
        return out


class WTM(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, dff=2048, dropout_transformer=.1, d_embedding=512):
        super(WTM, self).__init__()
        
        self.Spatial_embedding= nn.Linear(d_embedding, d_embedding)

        self.encoder = WAMAndChannelMixer(d_model, d_k, d_v, n_head, dff, dropout_transformer)
                                     
    def forward(self, x):
        shape = x.shape
        x = x.view(shape[0], shape[1], -1)
        
        x = self.Spatial_embedding(x)
        in_encoder = x + sinusoid_encoding_table(x.shape[1], x.shape[2]).expand(x.shape).cuda(device=0)
        in_encoder = self.encoder(in_encoder, shape[2], shape[3])
        
        out = in_encoder.view(shape[0], shape[1], shape[2], shape[3])
        
        return out