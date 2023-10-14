import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=128):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.25))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class BaseBlk(nn.Module):
    def __init__(self, init='kaiming'):
        super(BaseBlk, self).__init__()
        self.init = init
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                self.init_module(module)

    def init_module(self, module):
        # Pytorch does kaiming initialiaztion at the moment, so do not need to inititliaze again.
        if 'kaiming' in self.init:
            return

        init_method = init_methods[self.init]

        if self.init == "bilinear" and isinstance(module, nn.ConvTranspose2d):
            init_method(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


class Fire(BaseBlk):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, bn=True, bn_d=0.1, init='kaiming'):
        super(Fire, self).__init__(init)
        self.inplanes = inplanes
        self.bn = bn

        self.activation = nn.ReLU(inplace=True)

        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        if self.bn:
            self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=bn_d)

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        if self.bn:
            self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=bn_d)

        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        if self.bn:
            self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=bn_d)
        self.reset_parameters()

    def forward(self, x):
        x = self.squeeze(x)
        if self.bn:
            x = self.squeeze_bn(x)
        x = self.activation(x)

        x_1x1 = self.expand1x1(x)
        if self.bn:
            x_1x1 = self.expand1x1_bn(x_1x1)
        x_1x1 = self.activation(x_1x1)

        x_3x3 = self.expand3x3(x)
        if self.bn:
            x_3x3 = self.expand3x3_bn(x_3x3)
        x_3x3 = self.activation(x_3x3)

        out = torch.cat([x_1x1, x_3x3], 1)
        return out


class FireDeconv(BaseBlk):
    """Fire deconvolution layer constructor.
    Args:
      inputs: input channels
      squeeze_planes: number of 1x1 filters in squeeze layer.
      expand1x1_planes: number of 1x1 filters in expand layer.
      expand3x3_planes: number of 3x3 filters in expand layer.
      stride: spatial upsampling factors.[1,2]
    Returns:
      fire deconv layer operation.
    """
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes,
                 stride=(1, 2), padding=(0, 1), bn=True, bn_d=0.1, init='kaiming'):
        super(FireDeconv, self).__init__(init)
        self.bn = bn

        self.activation = nn.ReLU(inplace=True)

        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        if self.bn:
            self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=bn_d)

        # upsampling
        self.squeeze_deconv = nn.ConvTranspose2d(squeeze_planes, squeeze_planes,
                                                 kernel_size=(1, 4),
                                                 stride=stride, padding=padding)

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        if self.bn:
            self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=bn_d)

        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        if self.bn:
            self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=bn_d)
        self.reset_parameters()

    def forward(self, x):
        x = self.squeeze(x)
        if self.bn:
            x = self.squeeze_bn(x)
        x = self.activation(x)

        x = self.activation(self.squeeze_deconv(x))

        x_1x1 = self.expand1x1(x)
        if self.bn:
            x_1x1 = self.expand1x1_bn(x_1x1)
        x_1x1 = self.activation(x_1x1)

        x_3x3 = self.expand3x3(x)
        if self.bn:
            x_3x3 = self.expand3x3_bn(x_3x3)
        x_3x3 = self.activation(x_3x3)

        out = torch.cat([x_1x1, x_3x3], 1)
        return out


class SELayer(nn.Module):
    """Squeeze and Excitation layer from SEnet
    """
    def __init__(self, in_features, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x_scaled = x * y.expand_as(x)
        return x_scaled


def init_bilinear(tensor):
    """Reset the weight and bias."""
    nn.init.constant_(tensor, 0)
    in_feat, out_feat, h, w = tensor.shape

    assert h == 1, 'Now only support size_h=1'
    assert in_feat == out_feat, \
        'In bilinear interporlation mode, input channel size and output' \
        'filter size should be the same'
    factor_w = (w + 1) // 2

    if w % 2 == 1:
        center_w = factor_w - 1
    else:
        center_w = factor_w - 0.5

    og_w = torch.reshape(torch.arange(w), (h, -1))
    up_kernel = (1 - torch.abs(og_w - center_w) / factor_w)
    for c in range(in_feat):
        tensor.data[c, c, :, :] = up_kernel


class PointSegNet(nn.Module):
    def __init__(self, num_classes):
        super(PointSegNet, self).__init__()
        bn_d = 0.1
        num_classes = num_classes
        self.p = 0.25
        self.bypass = True
        self.input_shape = [64, 1024, 5]

        h, w, c = self.input_shape

        ### Ecnoder part
        self.conv1a = nn.Sequential(nn.Conv2d(c, 64, kernel_size=3, stride=(1, 2), padding=1),
                                    nn.BatchNorm2d(64, momentum=bn_d),
                                    nn.ReLU(inplace=True))

        self.conv1b = nn.Sequential(nn.Conv2d(c, 64, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(64, momentum=bn_d),
                                    nn.ReLU(inplace=True))

        # First block
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.fire2 = Fire(64, 16, 64, 64, bn=True, bn_d=bn_d)
        self.fire3 = Fire(128, 16, 64, 64, bn=True, bn_d=bn_d)
        self.se1 = SELayer(128, reduction=2)

        # second block
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.fire4 = Fire(128, 32, 128, 128, bn=True, bn_d=bn_d)
        self.fire5 = Fire(256, 32, 128, 128, bn=True, bn_d=bn_d)
        self.se2 = SELayer(256, reduction=2)

        # third block
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.fire6 = Fire(256, 48, 192, 192, bn=True, bn_d=bn_d)
        self.fire7 = Fire(384, 48, 192, 192, bn=True, bn_d=bn_d)
        self.fire8 = Fire(384, 64, 256, 256, bn=True, bn_d=bn_d)
        self.fire9 = Fire(512, 64, 256, 256, bn=True, bn_d=bn_d)
        self.se3 = SELayer(512, reduction=2)

        self.aspp = ASPP(512, [6, 9, 12])

        ### Decoder part
        self.fdeconv_el = FireDeconv(128, 32, 128, 128, bn=True, bn_d=bn_d)

        self.fdeconv_1 = FireDeconv(512, 64, 128, 128, bn=True, bn_d=bn_d)
        self.fdeconv_2 = FireDeconv(512, 64, 64, 64, bn=True, bn_d=bn_d)
        self.fdeconv_3 = FireDeconv(128, 16, 32, 32, bn=True, bn_d=bn_d)
        self.fdeconv_4 = FireDeconv(64, 16, 32, 32, bn=True, bn_d=bn_d)

        self.drop = nn.Dropout2d(p=self.p)
        self.conv2 = nn.Sequential(nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_1a = self.conv1a(x) # (H, W/2)
        x_1b = self.conv1b(x)

        ### Encoder forward
        # first fire block
        x_p1 = self.pool1(x_1a)
        x_f2 = self.fire2(x_p1)
        x_f3 = self.fire3(x_f2)
        x_se1 = self.se1(x_f3)
        if self.bypass:
            x_se1 += x_f2

        # second fire block
        x_p2 = self.pool2(x_se1)
        x_f4 = self.fire4(x_p2)
        x_f5 = self.fire5(x_f4)
        x_se2 = self.se2(x_f5)
        if self.bypass:
            x_se2 += x_f4

        # third fire block
        x_p3 = self.pool3(x_se2)
        x_f6 = self.fire6(x_p3)
        x_f7 = self.fire7(x_f6)
        if self.bypass:
            x_f7 += x_f6
        x_f8 = self.fire8(x_f7)
        x_f9 = self.fire9(x_f8)
        x_se3  =self.se3(x_f9)
        if self.bypass:
            x_se3 += x_f8

        # EL forward
        x_el = self.aspp(x_se3)
        x_el = self.fdeconv_el(x_el)

        ### Decoder forward
        x_fd1 = self.fdeconv_1(x_se3)  # (H, W/8)
        x_fd1_fused = torch.add(x_fd1, x_se2)
        x_fd1_fused = torch.cat((x_fd1_fused, x_el), dim=1)

        x_fd2 = self.fdeconv_2(x_fd1_fused)  # (H, W/4)
        x_fd2_fused = torch.add(x_fd2, x_se1)

        x_fd3 = self.fdeconv_3(x_fd2_fused)  # (H, W/2)
        x_fd3_fused = torch.add(x_fd3, x_1a)

        x_fd4 = self.fdeconv_4(x_fd3_fused)  # (H, W/2)
        x_fd4_fused = torch.add(x_fd4, x_1b)

        x_d = self.drop(x_fd4_fused)
        x = self.conv2(x_d)

        return x