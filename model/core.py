import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expansion, stride, bn_running_stats=True):
        super(Bottleneck, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        padding = (kernel_size - 1) // 2
        hidden_channels = expansion * in_channels

        self.conv = nn.Sequential(
            # pointwise-conv
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels, track_running_stats=bn_running_stats),
            nn.ReLU6(inplace=True),
            # depthwise-conv
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels, track_running_stats=bn_running_stats),
            nn.ReLU6(inplace=True),
            # pointwise-conv linear
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=bn_running_stats)
        )


    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
