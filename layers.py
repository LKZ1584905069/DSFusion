import torch.nn as nn

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, norm=None, activation='LReLU', kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)]
        if norm == 'Batch':
            model += [nn.BatchNorm2d(out_channels)]
        if activation == 'LReLU': ## 默认使用LeakyReLU作为激活函数
            model += [nn.LeakyReLU(inplace=True)]
        elif activation == 'Sigmoid':
            model += [nn.Sigmoid()]
        elif activation == 'ReLU':
            model += [nn.ReLU()]
        elif activation == 'Tanh':
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)