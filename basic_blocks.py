import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)

    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1, c, h, w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h, w)


class SetBlock_3d(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock_3d, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.forward_block(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        n, s, c, h, w = x.size()
        if self.pooling:
            x = self.pool2d(x.view(-1, c, h, w))
            _, c, h, w = x.size()
        if mask is None:
            return x.view(n, s, c, h, w)
        return x.view(n, s, c, h, w)*mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
