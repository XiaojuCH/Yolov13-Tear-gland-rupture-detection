import torch.nn as nn
import torch
class CA(nn.Module):

    def __init__(self, inp_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp_channels // reduction)
        self.conv1 = nn.Conv2d(inp_channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, inp_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, inp_channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)  # (b,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b,c,w,1)
        x_cat = torch.cat([x_h, x_w], dim=2)  # (b,c,h+w,1)
        x_cat = self.conv1(x_cat)  # (b,mip,h+w,1)
        x_cat = self.bn1(x_cat)
        x_cat = self.act(x_cat)
        x_h_split, x_w_split = torch.split(x_cat, [h, w], dim=2)
        x_w_split = x_w_split.permute(0, 1, 3, 2)  # (b,mip,1,w)
        a_h = torch.sigmoid(self.conv_h(x_h_split))  # (b,c,h,1)
        a_w = torch.sigmoid(self.conv_w(x_w_split))  # (b,c,1,w)
        out = x * a_h * a_w
        return out