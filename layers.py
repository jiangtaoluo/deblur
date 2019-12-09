import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
from collections import OrderedDict


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        # split
        for i, conv in enumerate(self.conv):
            # print(i,conv(input).size())
            output.append(conv(input))
        # fusion
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        # select
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))
        # split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        # print(V.size())
        return V


class _SKMixLayer(nn.Sequential):
    def __init__(self, num_input_features, expansion, k1, k2, drop_rate):
        super(_SKMixLayer, self).__init__()
        self.skconv = SKConv(num_input_features, num_input_features)
        if k1 > 0:
            self.bn1_1 = nn.BatchNorm2d(num_input_features)
            self.conv1_1 = nn.Conv2d(num_input_features, expansion * k1, kernel_size=1, stride=1, bias=False)
            self.bn1_2 = nn.BatchNorm2d(expansion * k1)
            self.conv1_2 = nn.Conv2d(expansion * k1, k1, kernel_size=3, stride=1, padding=1, bias=False)

        if k2 > 0:
            self.bn2_1 = nn.BatchNorm2d(num_input_features)
            self.conv2_1 = nn.Conv2d(num_input_features, expansion * k2, kernel_size=1, stride=1, bias=False)
            self.bn2_2 = nn.BatchNorm2d(expansion * k2)
            self.conv2_2 = nn.Conv2d(expansion * k2, k2, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate
        self.relu = nn.ReLU(inplace=True)
        self.k1 = k1
        self.k2 = k2

    def forward(self, x):
        x = self.skconv(x)
        if self.k1 > 0:
            inner_link = self.bn1_1(x)
            inner_link = self.relu(inner_link)
            inner_link = self.conv1_1(inner_link)
            inner_link = self.bn1_2(inner_link)
            inner_link = self.relu(inner_link)
            inner_link = self.conv1_2(inner_link)

        if self.k2 > 0:
            outer_link = self.bn2_1(x)
            outer_link = self.relu(outer_link)
            outer_link = self.conv2_1(outer_link)
            outer_link = self.bn2_2(outer_link)
            outer_link = self.relu(outer_link)
            outer_link = self.conv2_2(outer_link)

        if self.drop_rate > 0:
            inner_link = F.dropout(inner_link, p=self.drop_rate, training=self.training)
            outer_link = F.dropout(outer_link, p=self.drop_rate, training=self.training)

        c = x.size(1)
        # print(c)
        if self.k1 > 0 and self.k1 < c:
            xl = x[:, 0: c - self.k1, :, :]
            xr = x[:, c - self.k1: c, :, :] + inner_link
            x = torch.cat((xl, xr), 1)
        elif self.k1 == c:
            x = x + inner_link

        if self.k2 > 0:
            out = torch.cat((x, outer_link), 1)
        else:
            out = x

        #print(out.size())
        return out


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=3, stride=2, padding=1,bias=False)
        self.batchNorm = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)


        #print(x.size())
        return x


class _DeTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_DeTransition, self).__init__()
        self.deconv = nn.ConvTranspose2d(num_input_features, num_output_features, kernel_size=3, stride=2, padding=1, output_padding=1,
                                         bias=False)
        self.batchNorm = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        #print(x.size())
        return x


class SKMix_Block(nn.Sequential):

    def __init__(self, num_layers, in_channels, k1, k2, drop_rate, expansion=2):
        super(SKMix_Block, self).__init__()
        self.features = nn.Sequential()
        for i in range(num_layers):
            layer = _SKMixLayer(in_channels + i * k2, expansion, k1, k2, drop_rate)
            self.features.add_module('mixlayer%d' % (i + 1), layer)

    def forward(self, x):
        out = self.features(x)
        #print(out.size())
        return out


