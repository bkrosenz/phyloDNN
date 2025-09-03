from collections import OrderedDict
from dataclasses import dataclass
from distutils.command.build import build
from functools import partial

import torch
# from torch._C import channels_last
import torch.nn as nn

# 1x3 convolution


def conv1x3(in_channels, out_channels, stride=1, kernel_size=3, bias=False):
    """input shape: [ntaxa,gene_length,ngenes,].
    Assumes we do not convolve across genes (this would break equivariance)"""
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=(kernel_size, 1),
                     stride=stride,
                     padding=(kernel_size // 2, 0),
                     bias=bias)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv1x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm2d(self.expanded_channels)

            })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.BatchNorm2d(out_channels)}))


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels,
                    conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels,
                    conv=self.conv, bias=False),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels,
                    self.conv, kernel_size=1),
            activation(),
            conv_bn(self.out_channels, self.out_channels, self.conv,
                    kernel_size=3, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels,
                    self.conv, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **
                  kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels=3,
                 blocks_sizes=[64, 128, 256, 512],
                 depths=[2, 2, 2, 2],
                 activation=nn.ReLU,
                 block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        gate = nn.Sequential(
            nn.Conv2d(
                in_channels, self.blocks_sizes[0],
                kernel_size=(7, 1), stride=(2, 1),
                padding=(3, 0), bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation,
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])
        self.add_module('gate', gate)
        self.add_module('blocks', blocks)

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output 
    to a vector using a fully connected layer.
    """

    def __init__(self, in_features, n_classes):
        super().__init__()
        # if there are multiple genes, this will average across the gene dim
        self.add_module('avg', nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('decoder', nn.Linear(in_features, n_classes))

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)  # last 2 dims are now 1,1
        x = self.decoder(x)
        return x


class ResNet(nn.Module):

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        encoder = ResNetEncoder(in_channels, *args, **kwargs)
        decoder = ResnetDecoder(
            in_features=encoder.blocks[-1].blocks[-1].expanded_channels,
            n_classes=n_classes)
        self.add_module('encoder', encoder)
        self.add_module('decoder', decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def to(self, device):
        self = super().to(device)
        self.device = device
        return self


def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, depths=[2, 2, 2, 2])


def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, depths=[3, 4, 6, 3])


def resnet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, depths=[3, 4, 6, 3])


def resnet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, depths=[3, 4, 23, 3])


def simple_cnn(in_channels, n_classes):
    layers = [in_channels] + [2048, 1024, 512]+[n_classes]
    return build_cnn(layers, kernel_width=17, stride_width=3)


def build_cnn(layer_sizes, kernel_width=7, stride_width=2, dropout=0):
    in_channels, *hidden_channels, n_classes = layer_sizes
    layers = []
    final_dim = n_classes
    for num_filters in hidden_channels:
        conv_layer = nn.Conv2d(
            in_channels,
            num_filters,
            kernel_size=(kernel_width, 1),
            stride=(stride_width, 1),
            padding=(3, 0),
            bias=False)
        layers.extend([nn.BatchNorm2d(in_channels),
                       conv_layer,
                       nn.ELU(), ])
        in_channels = num_filters
    layers.extend([
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(
            in_channels, final_dim,
            kernel_size=(1, 1), stride=(1, 1),
            padding=0, bias=False),
        nn.ELU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.BatchNorm2d(final_dim),
        nn.Flatten(),
        nn.Linear(final_dim, n_classes)
    ])

    return nn.Sequential(*layers)


def large_cnn(in_channels, n_classes, kernel_width=21, dropout=0.1):
    layers = [nn.BatchNorm2d(in_channels), nn.Dropout(p=dropout)]
    final_dim = n_classes
    for num_filters in [2048, 1024, 512, 512, 256]:
        layers.extend([
            # nn.BatchNorm2d(in_channels),
            # nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels, num_filters,
                kernel_size=(kernel_width, 1), stride=(2, 1),
                padding=(3, 0), bias=False),
            nn.ELU(), ])
        in_channels = num_filters
    layers.extend([
        # nn.BatchNorm2d(in_channels),
        # nn.Dropout(p=dropout),
        nn.Conv2d(
            in_channels, final_dim,
            kernel_size=(1, 1), stride=(1, 1),
            padding=0, bias=False),
        nn.ELU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        # nn.BatchNorm2d(final_dim),
        nn.Flatten(),
        nn.Linear(final_dim, n_classes)
    ])
    return nn.Sequential(*layers)
