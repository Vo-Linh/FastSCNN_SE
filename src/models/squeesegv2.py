from functools import reduce
from operator import __add__

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)
      
class MaxPool2dSamePadding(nn.MaxPool2d):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        in_height, in_width = input.shape[2:]

        if type(self.stride) is not tuple:
            self.stride = (self.stride, self.stride)
        if type(self.kernel_size) is not tuple:
            self.kernel_size = (self.kernel_size, self.kernel_size)

        if (in_height % self.stride[0] == 0):
            pad_along_height = max(self.kernel_size[0] - self.stride[0], 0)
        else:
            pad_along_height = max(self.kernel_size[0] - (in_height % self.stride[0]), 0)
        if (in_width % self.stride[1] == 0):
            pad_along_width = max(self.kernel_size[1] - self.stride[1], 0)
        else:
            pad_along_width = max(self.kernel_size[1] - (in_width % self.stride[1]), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        input = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), value = float('-inf'))
        self.padding = (0, 0) # We did padding in the lane before. Force it to 0 by user

        return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        
class FireModule(nn.Module):
    def __init__(self, in_channels, conv1x1_1_size, conv1x1_2_size, conv3x3_size):
        super(FireModule, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, conv1x1_1_size, kernel_size=1, padding='same')
        self.batchnorm1 = nn.BatchNorm2d(conv1x1_1_size)
        self.conv1x1_2 = nn.Conv2d(conv1x1_1_size, conv1x1_2_size, kernel_size=1)
        self.conv3x3 = nn.Conv2d(conv1x1_1_size, conv3x3_size, kernel_size=3, padding='same')
        self.batchnorm2 = nn.BatchNorm2d(conv1x1_2_size)
        self.batchnorm3 = nn.BatchNorm2d(conv3x3_size)

    def forward(self, x):
        conv1x1_1 = F.relu(self.conv1x1_1(x))
        batchnorm1 = self.batchnorm1(conv1x1_1)
        conv1x1_2 = F.relu(self.conv1x1_2(conv1x1_1))
        batchnorm2 = self.batchnorm2(conv1x1_2)
        conv3x3 = F.relu(self.conv3x3(conv1x1_1))
        batchnorm3 = self.batchnorm3(conv3x3)
        return torch.cat([batchnorm2, batchnorm3], dim=1)

class FireDeconvModule(nn.Module):
    def __init__(self, in_channels, conv1x1_1_size, deconv_size, conv1x1_2_size, conv1x1_3_size):
        super(FireDeconvModule, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, conv1x1_1_size, kernel_size=1, padding='same')
        self.batchnorm1 = nn.BatchNorm2d(conv1x1_1_size)
        self.deconv = nn.ConvTranspose2d(conv1x1_1_size, deconv_size, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))
        self.batchnorm2 = nn.BatchNorm2d(deconv_size)
        self.conv1x1_2 = nn.Conv2d(deconv_size, conv1x1_2_size, kernel_size=1, padding='same')
        self.conv1x1_3 = nn.Conv2d(deconv_size, conv1x1_3_size, kernel_size=1, padding='same')
        self.batchnorm3 = nn.BatchNorm2d(conv1x1_2_size)
        self.batchnorm4 = nn.BatchNorm2d(conv1x1_3_size)

    def forward(self, x):
        conv1x1_1 = F.relu(self.conv1x1_1(x))
        batchnorm1 = self.batchnorm1(conv1x1_1)
        deconv = F.relu(self.deconv(batchnorm1))
        batchnorm2 = self.batchnorm2(deconv)
        conv1x1_2 = F.relu(self.conv1x1_2(batchnorm2))
        conv1x1_3 = F.relu(self.conv1x1_3(batchnorm2))
        batchnorm3 = self.batchnorm3(conv1x1_2)
        batchnorm4 = self.batchnorm4(conv1x1_3)

        return torch.cat([batchnorm3, batchnorm4], dim=1)

class CAMModule(nn.Module):
    def __init__(self, in_channels, conv1x1_1_size, conv1x1_2_size):
        super(CAMModule, self).__init__()
        self.MaxPooling = MaxPool2dSamePadding(kernel_size=7, stride=1)
        self.conv1x1_1 = nn.Conv2d(in_channels, conv1x1_1_size, kernel_size=1, padding='same')
        self.conv1x1_2 = nn.Conv2d(conv1x1_1_size, conv1x1_2_size, kernel_size=1, padding='same')

    def forward(self, x):
        conv1x1_1 = F.relu(self.conv1x1_1(self.MaxPooling(x)))
        conv1x1_2 = F.sigmoid(self.conv1x1_2(conv1x1_1))
        return torch.mul(x, conv1x1_2)

class DefaultEncoder(nn.Module):
    def __init__(self, in_channels, filter_size):
        super(DefaultEncoder, self).__init__()
        self.conv3x3 = Conv2dSamePadding(in_channels, filter_size, kernel_size=3, stride=(1, 2))
        self.batchnorm = nn.BatchNorm2d(filter_size)

    def forward(self, x):
        x = F.relu(self.conv3x3(x))
        x = self.batchnorm(x)
        return x

class SkipConv(nn.Module):
    def __init__(self, in_channels, filter_size):
        super(SkipConv, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, filter_size, kernel_size=1, padding='same')
        self.batchnorm = nn.BatchNorm2d(filter_size)

    def forward(self, x):
        x = F.relu(self.conv1x1(x))
        x = self.batchnorm(x)
        return x

class SqueezeSegV2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SqueezeSegV2, self).__init__()
        self.defaultDecoder = DefaultEncoder(input_size[-3], 64)
        self.skipconv = SkipConv(input_size[-3], 64)
        self.cam1 = CAMModule(64, 4, 64)
        self.maxpool = MaxPool2dSamePadding(kernel_size=3, stride=(1, 2), padding='same')
        self.fire2 = FireModule(64, 16, 64, 64)
        self.cam2 = CAMModule(128, 8, 128)
        self.fire3 = FireModule(128, 16, 64, 64)
        self.fire4 = FireModule(128, 32, 128, 128)
        self.fire5 = FireModule(256, 32, 128, 128)
        self.fire6 = FireModule(256, 32, 128, 128)
        self.fire7 = FireModule(256, 32, 128, 128)
        self.fire8 = FireModule(256, 48, 192, 192)
        self.fire9 = FireModule(384, 48, 192, 192)
        self.fire10 = FireModule(384, 48, 192, 192)
        self.fire11 = FireModule(384, 48, 192, 192)
        self.fire12 = FireModule(384, 64, 256, 256)
        self.fire13 = FireModule(512, 64, 256, 256)
        self.firedeconv14 = FireDeconvModule(512, 96, 96, 192, 192)
        self.firedeconv15 = FireDeconvModule(384, 64, 64, 128, 128)
        self.firedeconv16 = FireDeconvModule(256, 64, 64, 128, 128)
        self.firedeconv17 = FireDeconvModule(256, 32, 32, 64, 64)
        self.firedeconv18 = FireDeconvModule(128, 16, 16, 32, 32)
        self.firedeconv19 = FireDeconvModule(64, 16, 16, 32, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.conv_out = nn.Conv2d(64, num_classes, kernel_size=3, padding='same')

    def forward(self, x):
        defdecode = self.defaultDecoder(x)
        skipconv = self.skipconv(x)
        cam1 = self.cam1(defdecode)
        pool1 = self.maxpool(cam1)
        fire2 = self.fire2(pool1)
        cam2 = self.cam2(fire2)
        fire3 = self.fire3(cam2)
        pool2 = self.maxpool(fire3)
        fire4 = self.fire4(pool2)
        fire5 = self.fire5(fire4)
        pool3 = self.maxpool(fire5)
        fire6 = self.fire6(pool3)
        fire7 = self.fire7(fire6)
        pool4 = self.maxpool(fire7)
        fire8 = self.fire8(pool4)
        fire9 = self.fire9(fire8)
        pool5 = self.maxpool(fire9)
        fire10 = self.fire10(pool5)
        fire11 = self.fire11(fire10)
        fire12 = self.fire12(fire11)
        fire13 = self.fire13(fire12)
        fire14 = self.firedeconv14(fire13)
        add1 = torch.add(fire14, fire9)
        fire15 = self.firedeconv15(add1)
        add2 = torch.add(fire15, fire7)
        fire16 = self.firedeconv16(add2)
        add3 = torch.add(fire16, fire5)
        fire17 = self.firedeconv17(add3)
        add4 = torch.add(fire17, fire3)
        fire18 = self.firedeconv18(add4)
        add5 = torch.add(fire18, cam1)
        fire19 = self.firedeconv19(add5)
        add6 = torch.add(fire19, skipconv)
        x = self.dropout(add6)
        x = self.conv_out(x)
        return x