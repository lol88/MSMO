import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import numpy as np
import torch.nn.functional as F
from torchvision import transforms, models
# from att import *
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
import math

class MSAG(nn.Module):
    """
    Multi-scale attention gate
    Arxiv: https://arxiv.org/abs/2210.13012
    """
    def __init__(self, channel):
        super(MSAG, self).__init__()
        self.channel = channel
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 3, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv(x)
        _x = self.relu(torch.cat((x1, x2, x3), dim=1))
        _x = self.voteConv(_x)
        x = x + x * _x
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerBlock(nn.Module):
    def __init__(self, dim=1024, depth=7, k=7):
        super(ConvMixerBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(dim, dim, kernel_size=(k, k), groups=dim, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )

    def forward(self, x):
        x = self.block(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, input_size, hidden_size, kernel_size, padding):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size , kernel_size, padding=padding)

    def forward(self, input_, prev_state_spatial):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        if prev_state_spatial is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state_spatial = (torch.zeros(state_size).cuda(), torch.zeros(state_size).cuda())
        prev_hidden_spatial, prev_cell_spatial = prev_state_spatial
        stacked_inputs = torch.cat([input_, prev_hidden_spatial], 1)
        gates = self.Gates(stacked_inputs)
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * prev_cell_spatial) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        state = [hidden, cell]

        return state


class CLSTM_Decoder(nn.Module):
    """
    The recurrent decoder
    """
    def __init__(self, args, dim_mult=4, hidden_size = 512, with_masg=True, msag_dim=[16, 64, 128, 256]):
        super(CLSTM_Decoder,self).__init__()

        self.args = args
        self.with_masg = with_masg
        self.class_num = args.class_num
        self.hidden_size = hidden_size
        self.kernel_size = 3
        padding = 0 if self.kernel_size == 1 else 1
        self.dropout_p = 0.3
        self.skip_mode = 'concat'
        self.dropout = nn.Dropout(p=self.dropout_p)

        # skip_dims_out = [self.hidden_size, int(self.hidden_size / 2),
        #                  int(self.hidden_size / 4), int(self.hidden_size / 8), int(self.hidden_size / 16)]

        skip_dims_out = [self.hidden_size, int(self.hidden_size / 2),
                         int(self.hidden_size / 4), int(self.hidden_size / 8)]

        self.clstm_list = nn.ModuleList()  # Initialize layers for deconv stages

        self.msag4 = MSAG(msag_dim[3] * dim_mult)
        self.msag3 = MSAG(msag_dim[2] * dim_mult)
        self.msag2 = MSAG(msag_dim[1] * dim_mult)
        self.msag1 = MSAG(msag_dim[0] * dim_mult)

        for i in range(len(skip_dims_out)):
            if i == 0:
                clstm_in_dim = self.hidden_size * 2 #1024
            else:
                clstm_in_dim = skip_dims_out[i-1]
                if self.skip_mode == 'concat':
                    clstm_in_dim *= 2

            clstm_i = ConvLSTMCell(clstm_in_dim, skip_dims_out[i], self.kernel_size, padding = padding)
            self.clstm_list.append(clstm_i)
            del clstm_i

        self.conv_1 = nn.Conv2d(skip_dims_out[-1], skip_dims_out[-1]//2, self.kernel_size, padding = padding)
        self.bn1 = nn.BatchNorm2d(skip_dims_out[-1]//2, affine=True)
        self.conv_out = nn.Conv2d(skip_dims_out[-1]//2, 1,self.kernel_size, padding = padding)
        # self.conv_out = nn.Conv2d(skip_dims_out[-1], 1, self.kernel_size, padding = padding)
        # self.bn_out = nn.BatchNorm2d(1, affine=True)


    def forward(self, skip_feats_all, path=None, name=None):

        if self.with_masg:
            skip_feats_all[3] = self.msag4(skip_feats_all[3])
            skip_feats_all[2] = self.msag3(skip_feats_all[2])
            skip_feats_all[1] = self.msag2(skip_feats_all[1])
            skip_feats_all[0] = self.msag1(skip_feats_all[0])


        out_masks = []
        hidden_spatial = None


        for t in range(0, self.class_num):
            hidden_list = []
            clstm_in = skip_feats_all[4]
            skip_feats = skip_feats_all[:4][::-1]

            for i in range(len(skip_feats)):

                if hidden_spatial is None:
                    state = self.clstm_list[i](clstm_in, None)
                else:
                    state = self.clstm_list[i](clstm_in, hidden_spatial[i])

                hidden_list.append(state)
                hidden = state[0]

                if self.dropout_p > 0:
                    hidden = self.dropout(hidden)

                if i < len(skip_feats)-1:  # Skip connection
                    # skip_vec = skip_feats[i]
                    upsample = nn.UpsamplingBilinear2d(size=(skip_feats[i].size()[-2], skip_feats[i].size()[-1]))
                    hidden = upsample(hidden)
                    if self.skip_mode == 'concat':
                        clstm_in = torch.cat([hidden, skip_feats[i]], 1)
                    elif self.skip_mode == 'sum':
                        clstm_in = hidden + skip_feats[i]
                    elif self.skip_mode == 'mul':
                        clstm_in = hidden * skip_feats[i]
                    elif self.skip_mode == 'none':
                        clstm_in = hidden
                    else:
                        raise Exception('Skip connection mode not supported !')
                else:

                    self.upsample = nn.UpsamplingBilinear2d(size = (hidden.size()[-2]*2,hidden.size()[-1]*2))
                    # print("@@@@@@@@@@@@@@@@@@@")
                    hidden = self.upsample(hidden)
                    clstm_in = hidden

            out_1 = F.relu(self.bn1(self.conv_1(clstm_in)), inplace=True)
            del clstm_in, hidden
            # if not self.training:
            #     save_heat_img(out_1, path, name, f"E_{t}")

            out_mask = self.conv_out(out_1)
            out_masks.append(out_mask)
            hidden_spatial = hidden_list

        out_masks = torch.cat(out_masks, 1)

        return out_masks

class Decoder(nn.Module):

    def __init__(self, args, dim_mult=4, with_masg=True):
        super(Decoder, self).__init__()
        self.with_masg = with_masg
        self.args = args
        self.Up5 = up_conv(ch_in=256 * dim_mult, ch_out=128 * dim_mult)
        self.Up_conv5 = conv_block(ch_in=128 * 2 * dim_mult, ch_out=128 * dim_mult)
        self.Up4 = up_conv(ch_in=128 * dim_mult, ch_out=64 * dim_mult)
        self.Up_conv4 = conv_block(ch_in=64 * 2 * dim_mult, ch_out=64 * dim_mult)
        self.Up3 = up_conv(ch_in=64 * dim_mult, ch_out=32 * dim_mult)
        self.Up_conv3 = conv_block(ch_in=32 * 2 * dim_mult, ch_out=32 * dim_mult)
        self.Up2 = up_conv(ch_in=32 * dim_mult, ch_out=16 * dim_mult)
        self.Up_conv2 = conv_block(ch_in=16 * 2 * dim_mult, ch_out=16 * dim_mult)
        self.Conv_1x1 = nn.Conv2d(16 * dim_mult, self.args.class_num, kernel_size=1, stride=1, padding=0)

        self.msag4 = MSAG(128 * dim_mult)
        self.msag3 = MSAG(64 * dim_mult)
        self.msag2 = MSAG(32 * dim_mult)
        self.msag1 = MSAG(16 * dim_mult)

    def forward(self, feature):
        x1, x2, x3, x4, x5 = feature
        if self.with_masg:
            x4 = self.msag4(x4)
            x3 = self.msag3(x3)
            x2 = self.msag2(x2)
            x1 = self.msag1(x1)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        return d1


class MGCC(nn.Module):
    def __init__(self, args, img_ch=3, length=(3, 3, 3), k=7):
        """
        Multi-Level Global Context Cross Consistency Model
        Args:
            img_ch : input channel.
            output_ch: output channel.
            length: number of convMixer layers
            k: kernal size of convMixer

        """
        super(MGCC, self).__init__()

        self.args = args

        dim_mult = 4
        ConvMixer_dim = 256
        hidden_size = 512
        msag_dim = [16, 32, 64, 128]

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ResidualBlock(in_channels=img_ch, out_channels=16 * dim_mult)
        self.Conv2 = ResidualBlock(in_channels=16 * dim_mult, out_channels=32 * dim_mult)
        self.Conv3 = ResidualBlock(in_channels=32 * dim_mult, out_channels=64 * dim_mult)
        self.Conv4 = ResidualBlock(in_channels=64 * dim_mult, out_channels=128 * dim_mult)
        self.Conv5 = ResidualBlock(in_channels=128 * dim_mult, out_channels=256 * dim_mult)

        # self.Conv1 = conv_block(ch_in=img_ch, ch_out=16 * dim_mult)
        # self.Conv2 = conv_block(ch_in=16 * dim_mult, ch_out=32 * dim_mult)
        # self.Conv3 = conv_block(ch_in=32 * dim_mult, ch_out=64 * dim_mult)
        # self.Conv4 = conv_block(ch_in=64 * dim_mult, ch_out=128 * dim_mult)
        # self.Conv5 = conv_block(ch_in=128 * dim_mult, ch_out=256 * dim_mult)


        self.ConvMixer1 = ConvMixerBlock(dim=ConvMixer_dim * dim_mult, depth=length[0], k=k)
        self.ConvMixer2 = ConvMixerBlock(dim=ConvMixer_dim * dim_mult, depth=length[1], k=k)
        self.ConvMixer3 = ConvMixerBlock(dim=ConvMixer_dim * dim_mult, depth=length[2], k=k)

        # # # main Decoder
        self.main_decoder = CLSTM_Decoder(args=args, dim_mult=dim_mult, hidden_size=hidden_size, with_masg=True, msag_dim= msag_dim)
        # aux Decoder
        self.aux_decoder1 = CLSTM_Decoder(args=args, dim_mult=dim_mult, hidden_size=hidden_size, with_masg=True, msag_dim= msag_dim)
        self.aux_decoder2 = CLSTM_Decoder(args=args, dim_mult=dim_mult, hidden_size=hidden_size, with_masg=True, msag_dim= msag_dim)
        self.aux_decoder3 = CLSTM_Decoder(args=args, dim_mult=dim_mult, hidden_size=hidden_size, with_masg=True, msag_dim= msag_dim)

        # # main Decoder
        self.main_decoder = Decoder(args=args, dim_mult=dim_mult, with_masg=True)
        # aux Decoder
        self.aux_decoder1 = Decoder(args=args, dim_mult=dim_mult, with_masg=True)
        self.aux_decoder2 = Decoder(args=args, dim_mult=dim_mult, with_masg=True)
        self.aux_decoder3 = Decoder(args=args, dim_mult=dim_mult, with_masg=True)

    def forward(self, x):

        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)


        if not self.training:
            x5 = self.ConvMixer1(x5)
            x5 = self.ConvMixer2(x5)
            x5 = self.ConvMixer3(x5)
            feature = [x1, x2, x3, x4, x5]
            main_seg = self.main_decoder(feature)
            return main_seg

        # FeatureNoise
        feature = [x1, x2, x3, x4, x5]
        aux1_feature = [FeatureDropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)

        x5 = self.ConvMixer1(x5)
        feature = [x1, x2, x3, x4, x5]
        aux2_feature = [Dropout(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)

        x5 = self.ConvMixer2(x5)
        feature = [x1, x2, x3, x4, x5]
        aux3_feature = [FeatureNoise()(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)

        # main decoder
        x5 = self.ConvMixer3(x5)
        feature = [x1, x2, x3, x4, x5]
        main_seg = self.main_decoder(feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3
