import typing as t
import torch
import torch.nn as nn
from einops import rearrange


class ODF(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(ODF, self).__init__()  # 调用 nn.Module 的构造函数
        self.dim = dim  # 特征维度
        self.head_num = head_num  # 注意力头数
        self.head_dim = dim // head_num  # 每个头的维度
        self.scaler = self.head_dim ** -0.5  # 缩放因子
        self.group_kernel_sizes = group_kernel_sizes  # 分组卷积核大小
        self.window_size = window_size  # 窗口大小
        self.qkv_bias = qkv_bias  # 是否使用偏置
        self.fuse_bn = fuse_bn  # 是否融合批归一化
        self.down_sample_mode = down_sample_mode  # 下采样模式

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'  # 确保维度可被4整除
        self.group_chans = group_chans = self.dim // 4  # 分组通道数

        # 定义一维注意力门控
        self.h_gate = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.w_gate = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # 定义局部和全局深度卷积层
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)

        # 注意力门控层
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(2, dim)  # 水平方向的归一化
        self.norm_w = nn.GroupNorm(2, dim)  # 垂直方向的归一化
        self.conv_d = nn.Identity()  # 直接连接
        self.norm = nn.GroupNorm(1, dim)  # 通道归一化
        # 定义查询、键和值的卷积层
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # 注意力丢弃层
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()  # 通道注意力门控
        self.se = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), nn.ReLU(),
                                nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid())
        self.h_fuse = nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=False)
        self.w_fuse = nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=False)

        # 根据窗口大小和下采样模式选择下采样函数
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans  # 重组合下采样
                # 维度降低
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)  # 平均池化
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)  # 最大池化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入张量 x 的维度为 (B, C, H, W)
        假设维度(1, 4, 256, 256)
        """
        b, c, h_, w_ = x.size()  # 获取输入的形状(1, 4, 256, 256)
        # 基于自注意力的通道注意力
        # 减少计算量
        y = self.down_func(x)  # 下采样 (B, C, H/window_size, W/window_size) (1, 4, 256/7, 256/7)
        y = self.conv_d(y)  # 维度转换
        _, _, h0, w0 = y.size()  # 获取形状 因为顺序问题，这里将原来将直接进行第二次操作的h_和w_使用新变量h0,w0代替

        # 先归一化，然后重塑 -> (B, H, W, C) -> (B, C, H * W)，并生成 q, k 和 v
        y = self.norm(y)  # 归一化
        q = self.q(y)  # 计算查询
        k = self.k(y)  # 计算键
        v = self.v(y)  # 计算值
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # 计算注意力
        attn = q @ k.transpose(-2, -1) * self.scaler  # 点积注意力计算
        attn = self.attn_drop(attn.softmax(dim=-1))  # 应用注意力丢弃
        # (B, head_num, head_dim, N)
        attn = attn @ v  # 加权值
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h0), w=int(w0))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)  # 求平均
        attn = self.se(attn)
        attn = self.ca_gate(attn)  # 应用通道注意力门控
        x_cao = x * attn + x  # 使用残差连接

        # 空间注意力
        # 1) 按行求平均得到 (B, C, H)
        x_h = x_cao.mean(dim=3)

        # 2) 划分成 4 段 (B, C/4, H)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)

        # 3) 各段 depthwise conv 后再拼接回 (B, C, H)
        h_cat = torch.cat([
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ], dim=1)  # shape = (B, C, H)

        # 4) 1×1 融合 + 归一化 + 激活
        h_fused = self.h_fuse(h_cat)  # (B, C, H)
        x_h_attn = self.sa_gate(self.norm_h(h_fused))  # (B, C, H)

        # 5) Gate 残差平滑
        r = x_h_attn  # (B, C, H)
        g = self.h_gate(r)  # (B, C, H)
        r2 = (r + g) * 0.5  # (B, C, H)
        x_h_attn_ = r2.unsqueeze(-1)  # (B, C, H, 1)

        # ---- 垂直注意力 ----
        # 1) 按列求平均得到 (B, C, W)
        x_w = x_cao.mean(dim=2)

        # 2) 划分成 4 段 (B, C/4, W)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        # 3) 各段 depthwise conv 后再拼接回 (B, C, W)
        w_cat = torch.cat([
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l),
        ], dim=1)  # shape = (B, C, W)

        # 4) 1×1 融合 + 归一化 + 激活
        w_fused = self.w_fuse(w_cat)  # (B, C, W)
        x_w_attn = self.sa_gate(self.norm_w(w_fused))  # (B, C, W)

        # 5) Gate 残差平滑
        r_w = x_w_attn  # (B, C, W)
        g_w = self.w_gate(r_w)  # (B, C, W)
        r2_w = (r_w + g_w) * 0.5  # (B, C, W)
        x_w_attn_ = r2_w.unsqueeze(2)  # (B, C, 1, W)

        # 计算最终的注意力加权
        return x_cao * x_h_attn_ * x_w_attn_


# x = torch.randn(1, 4, 256, 256)
# odf = ODF(4, 2)
# output = odf(x)
# print(output)
