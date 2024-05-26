import torch.nn as nn
from layers import ConvLeakyRelu2d
import torch
from options import TrainOptions


opt = TrainOptions().parse()
conv_out = 40




class Encoder(nn.Module):
    def __init__(self,in_channel=1):
        super(Encoder, self).__init__()
        # 源图像拉成40个特征图
        self.conv1 = nn.Sequential(ConvLeakyRelu2d(in_channel, conv_out, norm='Batch', activation='LReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, activation='LReLU'))
        # 主干
        self.conv2 = nn.Sequential(ConvLeakyRelu2d(conv_out, conv_out, norm='Batch', activation='LReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, activation='LReLU'))
        self.conv3 = nn.Sequential(ConvLeakyRelu2d(conv_out, conv_out, norm='Batch', activation='LReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, activation='LReLU'))
        self.conv4 = nn.Sequential(ConvLeakyRelu2d(conv_out, conv_out, norm='Batch', activation='LReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, activation='LReLU'))
        self.conv5 = nn.Sequential(ConvLeakyRelu2d(conv_out, conv_out, norm='Batch', activation='LReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, activation='LReLU'))
        # 上分支
        self.one1 = nn.Sequential(ConvLeakyRelu2d(conv_out, conv_out, kernel_size=(1,5), norm='Batch', activation='LReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, kernel_size=(5,1), norm='Batch', activation='Sigmoid'))
        self.one2 = nn.Sequential(ConvLeakyRelu2d(conv_out, conv_out, kernel_size=(1,5), norm='Batch', activation='LReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, kernel_size=(5,1), norm='Batch', activation='Sigmoid'))
        self.one3 = nn.Sequential(ConvLeakyRelu2d(conv_out, conv_out, kernel_size=(1,5), norm='Batch', activation='LReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, kernel_size=(5,1), norm='Batch', activation='Sigmoid'))
        self.one4 = nn.Sequential(ConvLeakyRelu2d(conv_out, conv_out, kernel_size=(1, 5), norm='Batch', activation='LReLU'),
                                  ConvLeakyRelu2d(conv_out, conv_out, kernel_size=(5, 1), norm='Batch', activation='Sigmoid'))
        # 下分支
        self.down1 = nn.Sequential(ConvLeakyRelu2d(conv_out, conv_out,stride=2, norm='Batch', activation='LReLU'),
                                   ConvLeakyRelu2d(conv_out, conv_out, norm='Batch', activation='Sigmoid'))
        self.down2 = nn.Sequential(ConvLeakyRelu2d(conv_out, conv_out, norm='Batch', activation='LReLU'),
                                   ConvLeakyRelu2d(conv_out, conv_out, norm='Batch', activation='Sigmoid'))
        self.down3 = nn.Sequential(ConvLeakyRelu2d(conv_out, conv_out, norm='Batch', activation='LReLU'),
                                   ConvLeakyRelu2d(conv_out, conv_out, norm='Batch', activation='Sigmoid'))
        self.down4 = nn.Sequential(ConvLeakyRelu2d(conv_out, conv_out, norm='Batch', activation='LReLU'),
                                   ConvLeakyRelu2d(conv_out, conv_out, norm='Batch', activation='Sigmoid'))
        # 下分支需要用到的上采样
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self,x):
        conv1 = self.conv1(x)
        one1 = self.one1(conv1)
        one1 = one1 * conv1
        down1 = self.up(self.down1(conv1)) * conv1

        conv2 = self.conv2(conv1+one1+down1)
        one2 = self.one2(one1) * conv2
        down2 = self.down2(down1) * conv2

        conv3 = self.conv3(conv2+one2+down2)
        one3 = self.one3(one2) * conv3
        down3 = self.down3(down2) * conv3

        conv4 = self.conv4(conv3 + one3+down3)
        one4 = self.one4(one3) * conv4
        down4 = self.down4(down3) * conv4

        conv5 = self.conv5(conv4 + one4+down4)

        return conv5


# 空间注意力机制采用max（VIS空间，IR空间）* 混合特征
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(conv_out, conv_out // 4, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(conv_out//4, 1, kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        return self.sigmoid(x)


# 通道注意力机制采用（混合 - IR）作为VIS的通道注意力特征 * VIS，同理 IR 也是
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=conv_out, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1),
            nn.LeakyReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1)
        )
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 解码器的输入是 两个模块的连接
class Decoder(nn.Module):
    def __init__(self,in_channel = int(opt.Conv_out*2)):
        super(Decoder, self).__init__()
        channels = [in_channel,48,16,1]
        self.conv1 = ConvLeakyRelu2d(channels[0], channels[1], norm='Batch', activation='LReLU')
        self.conv2 = ConvLeakyRelu2d(channels[1], channels[2], norm='Batch',activation='LReLU')
        self.conv3 = ConvLeakyRelu2d(channels[2], channels[3], activation='Tanh')

    def forward(self,x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        return conv3


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.en_vis = Encoder()
        self.en_ir = Encoder()
        self.channel_vis = ChannelAttention()
        self.channel_ir = ChannelAttention()
        self.spatial_vis = SpatialAttention()
        self.spatial_ir = SpatialAttention()
        self.de = Decoder()

    def forward(self,vis,ir):
        en_vis = self.en_vis(vis)
        en_ir = self.en_ir(ir)
        channel_vis = self.channel_vis(en_vis-en_ir) * en_vis
        channel_ir = self.channel_ir(en_ir - en_vis) * en_ir
        spatial_vis = self.spatial_vis(channel_vis) * channel_vis
        spatial_ir = self.spatial_vis(channel_ir) * channel_ir
        print(spatial_ir.shape)
        print(spatial_vis.shape)
        out = self.de(torch.cat((spatial_vis,spatial_ir),1))
        return out





