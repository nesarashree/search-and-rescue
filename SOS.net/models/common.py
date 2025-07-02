# common.py

import torch
import torch.nn as nn

class Conv(nn.Module):
    # Standard convolutional layer with BatchNorm and activation (ReLU by default)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Args:
            c1 (int): Number of input channels
            c2 (int): Number of output channels
            k (int): Kernel size
            s (int): Stride
            p (int or None): Padding (if None, computed as k//2)
            g (int): Groups (default 1)
            act (bool or nn.Module): Activation function. If True, use ReLU, else None
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s,
                              padding=p if p is not None else k // 2,
                              groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = nn.ReLU(inplace=True)
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

        # Store args for convenience (used in your SNNConv patching)
        self.args = (c1, c2, k, s, p, g, act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
