import torch
import torch.nn as nn 
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import torchvision 
import torchvision.models as models
import torchvision.utils as vutils

import kornia
from collections import namedtuple


###########################################
##   Tools
#------------------------------------------

mean_std = namedtuple("mean_std", ['mean','std'])
vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
vgg_outputs_super = namedtuple("VggOutputs", ['map', 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])


###########################################
##   Layers and Blocks
#------------------------------------------


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):

        """ avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3 """
        
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class FilterPredictor(nn.Module):
    def __init__(self, vgg_channel=512, inner_channel=32):
        super(FilterPredictor, self).__init__()
        self.down_sample = nn.Sequential(nn.Conv2d(vgg_channel, inner_channel, kernel_size=3, padding=1))
        self.inner_channel = inner_channel
        self.FC = nn.Linear(inner_channel*2,inner_channel*inner_channel)

    def forward(self, content, style):
        content = self.down_sample(content)
        style = self.down_sample(style)

        content = torch.mean(content.view(content.size(0), content.size(1), -1), dim=2)
        style = torch.mean(style.view(style.size(0), style.size(1), -1), dim=2)

        filter = self.FC(torch.cat([content, style],1))
        filter = filter.view(-1, self.inner_channel, self.inner_channel).unsqueeze(3)
        return filter


class KernelFilter(nn.Module):
    def __init__(self, vgg_channel=512, inner_channel=32):
        super(KernelFilter, self).__init__()
        self.down_sample = nn.Sequential(
                        nn.Conv2d(vgg_channel, inner_channel, kernel_size=3, padding=1),
                    )

        self.upsample = nn.Sequential(
                        nn.Conv2d(inner_channel, vgg_channel, kernel_size=3, padding=1),
                    )

        self.F1 = FilterPredictor(vgg_channel, inner_channel)
        self.F2 = FilterPredictor(vgg_channel, inner_channel)

        self.relu = nn.LeakyReLU(0.2)

    def apply_filter(self, input_, filter_):
        ''' input_:  [B, inC, H, W]
            filter_: [B, inC, outC, 1] '''

        B = input_.shape[0]
        input_chunk = torch.chunk(input_, B, dim=0)
        filter_chunt = torch.chunk(filter_, B, dim=0)

        results = []

        for input, filter_ in zip(input_chunk, filter_chunt):
            input = F.conv2d(input, filter_.permute(1,2,0,3), groups=1)
            results.append(input)

        return torch.cat(results,0)

    def forward(self, content, style):
        content_ = self.down_sample(content)

        content_ = self.apply_filter(content_, self.F1(content, style))
        content_ = self.relu(content_)

        content_ = self.apply_filter(content_, self.F2(content, style))

        return content + self.upsample(content_)


class FilterPredictor_S(nn.Module):
    def __init__(self, vgg_channel=512, inner_channel=32):
        super(FilterPredictor_S, self).__init__()
        self.down_sample = nn.Sequential(
                        nn.Conv2d(vgg_channel, inner_channel, kernel_size=3, padding=1),
                    )
        self.inner_channel = inner_channel
        self.FC = nn.Linear(inner_channel,9*inner_channel*inner_channel)

    def forward(self, style):
        style = self.down_sample(style)
        style = torch.mean(style.view(style.size(0), style.size(1), -1), dim=2)

        filter = self.FC(style)
        filter = filter.view(-1, self.inner_channel, self.inner_channel, 3,3)
        return filter


class KernelFilter_S(nn.Module):
    def __init__(self, vgg_channel=512, inner_channel=32):
        super(KernelFilter_S, self).__init__()
        self.down_sample = nn.Sequential(
                        nn.Conv2d(vgg_channel, inner_channel, kernel_size=3, padding=1),
                    )

        self.upsample = nn.Sequential(
                        nn.Conv2d(inner_channel, vgg_channel, kernel_size=3, padding=1),
                    )

        self.F1 = FilterPredictor_S(vgg_channel, inner_channel)
        self.F2 = FilterPredictor_S(vgg_channel, inner_channel)

        self.relu = nn.LeakyReLU(0.2)

    def apply_filter(self, input_, filter_):
        ''' input_:  [B, inC, H, W]
            filter_: [B, inC, outC, 1] '''

        B = input_.shape[0]
        input_chunk = torch.chunk(input_, B, dim=0)
        filter_chunt = torch.chunk(filter_, B, dim=0)

        results = []

        for input, filter_ in zip(input_chunk, filter_chunt):
            input = F.conv2d(input, filter_.squeeze(0), groups=1, padding=1)
            results.append(input)

        return torch.cat(results,0)

    def forward(self, content, style):
        content_ = self.down_sample(content)

        content_ = self.apply_filter(content_, self.F1(style))
        content_ = self.relu(content_)

        content_ = self.apply_filter(content_, self.F2(style))

        return content + self.upsample(content_)


class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, upsample=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.conv_shortcut = nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.2)
        self.norm = InstanceNorm()
        self.upsample = upsample

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=2)
        x_s = self.conv_shortcut(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm(x)

        return x_s + x




###########################################
##   Networks
#------------------------------------------


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h
        out = vgg_outputs(h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # VGG
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice = nn.Sequential()
        for x in range(21):
            self.slice.add_module(str(x), vgg_pretrained_features[x])
            
    def forward(self, input_frame):
        return self.slice(input_frame)


class EncoderStyle(nn.Module):
    def __init__(self):
        super(EncoderStyle, self).__init__()
        # VGG
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

    def cal_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

        out = mean_std(feat_mean, feat_std)
        return out

    def forward(self, style):
        h = self.slice1(style)
        h_relu1_1 = self.cal_mean_std(h)

        h = self.slice2(h)
        h_relu2_1 = self.cal_mean_std(h)

        h = self.slice3(h)
        h_relu3_1 = self.cal_mean_std(h)

        h = self.slice4(h)
        h_relu4_1 = self.cal_mean_std(h)

        out = vgg_outputs_super(h, h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1)
        return out


class Decoder(nn.Module):
    def __init__(self, dynamic_filter=True, both_sty_con=True):
        super(Decoder, self).__init__()

        self.slice4 = ResidualBlock(512, 256)
        self.slice3 = ResidualBlock(256, 128)
        self.slice2 = ResidualBlock(128, 64)
        self.slice1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.norm = InstanceNorm()

        self.dynamic_filter = dynamic_filter
        if dynamic_filter:
            if both_sty_con:
                self.Filter1 = KernelFilter()
                self.Filter2 = KernelFilter()
                self.Filter3 = KernelFilter()
            else:
                self.Filter1 = KernelFilter_S()
                self.Filter2 = KernelFilter_S()
                self.Filter3 = KernelFilter_S()

    def AdaIN(self, content_feat, style_feat):
        size = content_feat.size()

        style_mean = style_feat.mean
        style_std = style_feat.std

        normalized_feat = self.norm(content_feat)

        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def AdaIN_filter(self, content_feat, style_feat, style_map):
        size = content_feat.size()

        style_mean = style_feat.mean
        style_std = style_feat.std

        normalized_content = self.norm(content_feat)
        normalized_style = (style_map - style_mean)/style_std

        results = self.Filter1(normalized_content, normalized_style)
        results = self.Filter2(results, normalized_style)
        results = self.Filter3(results, normalized_style)

        return results * style_std.expand(size) + style_mean.expand(size)

    def forward(self, x, style_features=None):
        if self.dynamic_filter:
            h = self.AdaIN_filter(x, style_features.relu4_1, style_features.map)
        else:
            h = self.AdaIN(x, style_features.relu4_1)
        
        h = self.slice4(h)

        h = self.AdaIN(h, style_features.relu3_1)
        h = self.slice3(h)

        h = self.AdaIN(h, style_features.relu2_1)
        h = self.slice2(h)

        h = self.AdaIN(h, style_features.relu1_1)
        h = self.slice1(h)

        return h


class TransformerNet(nn.Module):
    def __init__(self, dynamic_filter=True, both_sty_con=True):
        super(TransformerNet, self).__init__()

        # Sub-models
        self.Decoder = Decoder(dynamic_filter=dynamic_filter, both_sty_con=both_sty_con)
        self.Encoder = Encoder()
        self.EncoderStyle = EncoderStyle()
        self.Vgg19 = Vgg19()

        self.have_delete_vgg = False

    def generate_style_features(self, style):
        self.F_style = self.EncoderStyle(style)
        if not self.have_delete_vgg:
            del self.Vgg19
            self.have_delete_vgg = True

    def RGB2Gray(self, image):
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        image = (image*std+mean)

        gray = image[:,2:3,:,:]*0.299 + image[:,1:2,:,:]*0.587 + image[:,0:1,:,:]*0.114
        gray = gray.expand(image.size())

        gray = (gray-mean)/std

        return gray

    def forward(self, input_frame):
        F_content = self.Encoder(self.RGB2Gray(input_frame))
        return self.Decoder(F_content, self.F_style)

