import math
from torch import nn as nn
import torch.nn.functional as F


class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias),
        )

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat

    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        elif scale == 6:
            # x2
            m.append(conv(n_feat, 4 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(2))
            # x3
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))

            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EffEDSR(nn.Module):
    def __init__(self, n_resblocks=32, n_feats=256, scale=4, n_colors=31, conv=Conv3XC):
        super(EffEDSR, self).__init__()

        n_resblock = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        scale = scale
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size,
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]


        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        return x