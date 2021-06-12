import torch
import torchvision 
import torch.nn as nn 
from torch.nn import init

from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
from collections import namedtuple
import torchvision.utils as vutils
import time


###########################################
##   Layers and Blocks
#------------------------------------------

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8, style_num=1):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

        self.style_num = style_num

        self.x_max = [None for x in range(self.style_num)]
        self.x_min = [None for x in range(self.style_num)]

        self.saved_mean = [None for x in range(self.style_num)]
        self.saved_std = [None for x in range(self.style_num)]

    def forward(self, x, style_weight=[1.]):
        tmp_saved_mean = 0.
        tmp_saved_std = 0.
        tmp_x_min = 0.
        tmp_x_max = 0.

        for style_id in range(self.style_num):
            tmp_saved_mean += self.saved_mean[style_id] * style_weight[style_id]
            tmp_saved_std += self.saved_std[style_id] * style_weight[style_id]
            tmp_x_min += self.x_min[style_id] * style_weight[style_id]
            tmp_x_max += self.x_max[style_id] * style_weight[style_id]

        x = x - tmp_saved_mean
        x = x * tmp_saved_std

        x = torch.max(tmp_x_min, x)
        x = torch.min(tmp_x_max, x)

        return x

    def compute_norm(self, x, style_id):
        ## mean and std
        self.saved_mean[style_id] = torch.mean(x, (0, 2, 3), True)
        x = x - self.saved_mean[style_id]
        tmp = torch.mul(x, x)
        self.saved_std[style_id] = torch.rsqrt(torch.mean(tmp, (0, 2, 3), True) + self.epsilon)
        x = x * self.saved_std[style_id]

        ## max and min
        tmp_max, _ = torch.max(x, 2, True)
        tmp_max, _ = torch.max(tmp_max, 0, True)
        self.x_max[style_id], _ = torch.max(tmp_max, 3, True)

        tmp_min, _ = torch.min(x, 2, True)
        tmp_min, _ = torch.min(tmp_min, 0, True)
        self.x_min[style_id], _ = torch.min(tmp_min, 3, True)
        return x

    def clean(self):
        self.x_max = [None for x in range(self.style_num)]
        self.x_min = [None for x in range(self.style_num)]

        self.saved_mean = [None for x in range(self.style_num)]
        self.saved_std = [None for x in range(self.style_num)]


class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, upsample=True, style_num=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.conv_shortcut = nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.2)
        self.norm1 = InstanceNorm(style_num=style_num)
        self.norm2 = InstanceNorm(style_num=style_num)
        self.upsample = upsample
        self.style_num = style_num

    def forward(self, x, style_weight=[1.]):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=2)
        x_s = self.conv_shortcut(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm1(x, style_weight)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm2(x, style_weight)
        return x_s + x

    def compute_norm(self, x, style_id=1):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=2)
        x_s = self.conv_shortcut(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm1.compute_norm(x, style_id)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm2.compute_norm(x, style_id)
        return x_s + x

    def clean(self):
        self.norm1.clean()
        self.norm2.clean()


class FilterPredictor(nn.Module):
    def __init__(self, vgg_channel=512, inner_channel=32, style_num=1):
        super(FilterPredictor, self).__init__()
        self.down_sample = nn.Sequential(
                        nn.Conv2d(vgg_channel, inner_channel, kernel_size=3, padding=1),
                    )
        self.inner_channel = inner_channel
        self.FC = nn.Linear(inner_channel*2,inner_channel*inner_channel)
        self.style_num = style_num
        self.filter = [None for x in range(self.style_num)]

    def forward(self, style_weight=[1.]):
        tmp_filter = 0.
        for style_id in range(self.style_num):
            tmp_filter += self.filter[style_id] * style_weight[style_id]
        return tmp_filter

    def compute(self, content, style, style_id):
        content = self.down_sample(content)
        content = torch.mean(content.view(content.size(0),content.size(1), -1), dim=2)
        content = torch.mean(content, dim=0).unsqueeze(0)
        
        style = self.down_sample(style)
        style = torch.mean(style.view(style.size(0), style.size(1), -1), dim=2)

        filter = self.FC(torch.cat([content, style],1))
        filter = filter.view(-1, self.inner_channel, self.inner_channel).unsqueeze(3)
        self.filter[style_id] = filter
        return filter

    def clean(self):
        self.filter = [None for x in range(self.style_num)]


class KernelFilter(nn.Module):
    def __init__(self, vgg_channel=512, inner_channel=32, style_num=1):
        super(KernelFilter, self).__init__()
        self.down_sample = nn.Sequential(
                        nn.Conv2d(vgg_channel, inner_channel, kernel_size=3, padding=1),
                    )

        self.upsample = nn.Sequential(
                        nn.Conv2d(inner_channel, vgg_channel, kernel_size=3, padding=1),
                    )

        self.F1 = FilterPredictor(vgg_channel, inner_channel, style_num)
        self.F2 = FilterPredictor(vgg_channel, inner_channel, style_num)

        self.relu = nn.LeakyReLU(0.2)

        self.style_num = style_num

    def apply_filter(self, input_, filter_):
        ''' input_: [B,inC,H,W], filter_: [B,inC,outC,1]
        '''
        B = input_.shape[0]
        input_chunk = torch.chunk(input_, B, dim=0)
        filter_chunt = torch.chunk(filter_, B, dim=0)

        results = []

        for input, filter_ in zip(input_chunk, filter_chunt):
            input = F.conv2d(input, filter_.permute(1,2,0,3), groups=1)
            results.append(input)

        return torch.cat(results,0)

    def forward(self, content, style_weight=[1.]):
        content_ = self.down_sample(content)

        content_ = self.apply_filter(content_, self.F1(style_weight))
        content_ = self.relu(content_)
        content_ = self.apply_filter(content_, self.F2(style_weight))

        return content + self.upsample(content_)

    def clean(self):
        self.F1.clean()
        self.F2.clean()

    def compute(self, content, style, style_id):
        content_ = self.down_sample(content)

        content_ = self.apply_filter(content_, self.F1.compute(content, style, style_id))
        content_ = self.relu(content_)
        content_ = self.apply_filter(content_, self.F2.compute(content, style, style_id))

        return content + self.upsample(content_)


###########################################
##   Networks
#------------------------------------------

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=False).features
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
        vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
        out = vgg_outputs(h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        ## VGG
        vgg_pretrained_features = models.vgg19(pretrained=False).features
        self.slice = nn.Sequential()
        for x in range(21):
            self.slice.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, cur_frame):
        return self.slice(cur_frame)


class EncoderStyle(nn.Module):
    def __init__(self):
        super(EncoderStyle, self).__init__()
        ## VGG
        vgg_pretrained_features = models.vgg19(pretrained=False).features

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
        ## eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1) 
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1) 

        mean_std = namedtuple("mean_std", ['mean','std'])
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

        vgg_outputs = namedtuple("VggOutputs", ['map','relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
        out = vgg_outputs(h, h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1)

        return out


class Decoder(nn.Module):
    def __init__(self, style_num=1):
        super(Decoder, self).__init__()

        self.slice4 = ResidualBlock(512, 256, style_num=style_num)
        self.slice3 = ResidualBlock(256, 128, style_num=style_num)
        self.slice2 = ResidualBlock(128, 64, style_num=style_num)
        self.slice1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.Filter1 = KernelFilter(style_num=style_num)
        self.Filter2 = KernelFilter(style_num=style_num)
        self.Filter3 = KernelFilter(style_num=style_num)

        self.norm = [InstanceNorm(style_num=style_num),
                     InstanceNorm(style_num=style_num),
                     InstanceNorm(style_num=style_num),
                     InstanceNorm(style_num=style_num),
                     InstanceNorm(style_num=style_num)]

        self.style_num = style_num

        self.style_means = [None]
        self.style_stds = [None]

        for layer in range(1,5):
            self.style_means.append([None for i in range(style_num)])
            self.style_stds.append([None for i in range(style_num)])

    #############################
    # For forward Transfer
    # ---------------------------
    def AdaIN(self, content_feat, norm_id, style_weight=[1.]):
        normalized_feat = self.norm[norm_id](content_feat, style_weight)

        style_mean = 0.
        style_std = 0.

        for style_id in range(self.style_num):
            style_mean += self.style_means[norm_id][style_id] * style_weight[style_id]
            style_std += self.style_stds[norm_id][style_id] * style_weight[style_id]

        result =  normalized_feat * style_std + style_mean

        return result

    def AdaIN_filter(self, content_feat, style_weight=[1.]):
        normalized_content = self.norm[0](content_feat, style_weight)
        results = self.Filter1(normalized_content, style_weight)
        results = self.Filter2(results, style_weight)
        results = self.Filter3(results, style_weight)
        return results

    ##########################################
    # For memory-based pre-processing
    # ----------------------------------------

    def AdaIN_compute_norm(self, content_feat, style_feat, norm_id, style_id):
        style_mean = style_feat.mean
        style_std = style_feat.std

        self.style_means[norm_id][style_id] = style_mean
        self.style_stds[norm_id][style_id] = style_std

        normalized_feat = self.norm[norm_id].compute_norm(content_feat, style_id)
        return normalized_feat * style_std + style_mean

    def AdaIN_filter_compute_norm(self, content_feat, style_feat, style_map, style_id):
        style_mean = style_feat.mean
        style_std = style_feat.std

        normalized_content = self.norm[0].compute_norm(content_feat, style_id)
        normalized_style = (style_map - style_mean) / style_std

        results = self.Filter1.compute(normalized_content, normalized_style, style_id)
        results = self.Filter2.compute(results, normalized_style, style_id)
        results = self.Filter3.compute(results, normalized_style, style_id)
        return results

    #############
    # Others
    # -----------

    def clean(self):
        for norm in self.norm:
            norm.clean()

        self.slice4.clean()
        self.slice3.clean()
        self.slice2.clean()

        self.Filter1.clean()
        self.Filter2.clean()
        self.Filter3.clean()

    ##################
    # Processing
    # ----------------

    def compute_norm(self, x, style_features=None):
        for style_id, style_feature in enumerate(style_features):
            h = self.AdaIN_filter_compute_norm(x, style_feature.relu4_1, style_feature.map, style_id)

            h = self.AdaIN_compute_norm(h, style_feature.relu4_1, 1, style_id)
            h = self.slice4.compute_norm(h, style_id)

            h = self.AdaIN_compute_norm(h, style_feature.relu3_1, 2, style_id)
            h = self.slice3.compute_norm(h, style_id)

            h = self.AdaIN_compute_norm(h, style_feature.relu2_1, 3, style_id)
            h = self.slice2.compute_norm(h, style_id)

            h = self.AdaIN_compute_norm(h, style_feature.relu1_1, 4, style_id)

        del h

    def forward(self, x, style_weight=[1.]):
        start = time.time()
        h = self.AdaIN_filter(x, style_weight)

        start = time.time()
        h = self.AdaIN(h, 1, style_weight)

        start = time.time()
        h = self.slice4(h, style_weight)

        start = time.time()
        h = self.AdaIN(h, 2, style_weight)

        start = time.time()
        h = self.slice3(h, style_weight)

        start = time.time()
        h = self.AdaIN(h, 3, style_weight)

        start = time.time()
        h = self.slice2(h, style_weight)

        start = time.time()
        h = self.AdaIN(h, 4, style_weight)

        start = time.time()
        h = self.slice1(h)

        return h



class TransformerNet(nn.Module):
    def __init__(self, style_num=1):
        super(TransformerNet, self).__init__()
        self.Decoder = Decoder(style_num)
        self.Encoder = Encoder()
        self.EncoderStyle = EncoderStyle()
        self.Vgg19 = Vgg19()
        self.have_delete_vgg = False
        self.F_patches = []
        self.F_style = [None for x in range(style_num)]

    def generate_style_features(self, style, style_id):
        self.F_style[style_id] = self.EncoderStyle(style)

    def forward(self, F_content, style_weight=[1.]):
        ## -------------------------------
        ## Stylization

        start = time.time()
        styled_pre_frame = self.Decoder(F_content, style_weight)
        return styled_pre_frame

    def generate_content_features(self, content):
        return self.Encoder(self.RGB2Gray(content))

    def add_patch(self, F_patch):
        self.F_patches.append(F_patch)

    def compute_norm(self):
        self.Decoder.compute_norm(torch.cat(self.F_patches, dim=0), self.F_style)
        self.F_patches = []

    def clean(self):
        self.Decoder.clean()

    def RGB2Gray(self, image):
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        image = (image*std+mean)

        gray = image[:,2:3,:,:]*0.299 + image[:,1:2,:,:]*0.587 + image[:,0:1,:,:]*0.114
        gray = gray.expand(image.size())

        gray = (gray-mean)/std
        return gray




