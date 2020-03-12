import torch
import torch.nn as nn
import numpy as np


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, same_padding=False, dilation=1, groups=1, relu=True, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) * dilation / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, relu=True, padding=0, bn=False, groups=1):
        super(ConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Fire(nn.Module):
    def __init__(self, in_channel, squeeze_channel, expand1x1_channel, expand3x3_channel, dilation=1, bn=False):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channel, squeeze_channel, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_channel, eps=0.001, momentum=0, affine=True) if bn else None
        self.squeeze_relu = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_channel, expand1x1_channel, kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_channel, eps=0.001, momentum=0, affine=True) if bn else None
        self.expand1x1_relu = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(squeeze_channel, expand3x3_channel, kernel_size=3, padding=dilation, dilation=dilation)
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_channel, eps=0.001, momentum=0, affine=True) if bn else None
        self.expand3x3_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        if self.squeeze_bn is not None:
            x = self.squeeze_bn(x)
        x = self.squeeze_relu(x)

        x1 = self.expand1x1(x)
        if self.expand1x1_bn is not None:
            x1 = self.expand1x1_bn(x1)
        x1 = self.expand1x1_relu(x1)

        x3 = self.expand3x3(x)
        if self.expand3x3_bn is not None:
            x3 = self.expand3x3_bn(x3)
        x3 = self.expand3x3_relu(x3)

        return torch.cat((x1, x3), 1)


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    # print('load from file: %s' % fname)
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def load_net_safe(fname, net):
    import h5py
    print('load from file: %s' % fname)
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        try:
            param = torch.from_numpy(np.asarray(h5f[k]))
        except KeyError:
            print('do not find %s in h5 file' % k)
        else:
            print('loading %s from h5 file' % k)
            v.copy_(param)


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

