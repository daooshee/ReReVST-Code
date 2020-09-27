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


def warp(x, flo, padding_mode='border'):
    B, C, H, W = x.size()

    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    grid = grid.to(x.device)
    vgrid = grid - flo
    
    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)
    output = F.grid_sample(x, vgrid, padding_mode=padding_mode, )
    return output


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def crop_2d(input, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0):
    assert input.dim() == 4, 'only support Input(B, C, W, H)'
    B, C, W, H = input.size()
    return input[:, :,
                 crop_left:(W-crop_right),
                 crop_bottom:(H-crop_top)]


class Crop2d(nn.Module):
    def __init__(self, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0):
        super(Crop2d, self).__init__()
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def forward(self, input):
        return crop_2d(input,
                       self.crop_left,
                       self.crop_right,
                       self.crop_top,
                       self.crop_bottom)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std




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
            
    def forward(self, cur_frame):
        return self.slice(cur_frame)


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

        init_weights(self)

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
    def __init__(self, dynamic_filter=True, both_sty_con=True, train_only_decoder=False,
                       style_content_loss=True, recon_loss=True, relax_style=True):
        
        super(TransformerNet, self).__init__()

        # Sub-models
        self.Decoder = Decoder(dynamic_filter=dynamic_filter, both_sty_con=both_sty_con)
        self.Encoder = Encoder()
        self.EncoderStyle = EncoderStyle()
        self.Vgg19 = Vgg19()

        if train_only_decoder:
            for param in self.Encoder.parameters():
                param.requires_grad = False

            for param in self.EncoderStyle.parameters():
                param.requires_grad = False

        # Other functions and tools
        self.MSE = nn.MSELoss()
        self.Padding = nn.ReflectionPad2d((32,32,32,32))
        self.Cropping = Crop2d(32,32,32,32)
        self.gauss = kornia.filters.GaussianBlur2d((101, 101), (50.5, 50.5))

        # Parameters
        self.flow_scale = 8
        self.flow_iter = 16
        self.flow_max = 20
        self.flow_lr = 16

        self.use_style_loss = style_content_loss
        self.use_content_loss = style_content_loss
        self.use_recon_loss = recon_loss
        self.relax_style = relax_style

    ## ---------------------------------------------------
    ##  Functions for setting the states
    ##  Useful in adversarial training

    def ParamStatic(self):
        for param in self.parameters():
            param.requires_grad = False

    def ParamActive(self):
        for param in self.Encoder.parameters():
            param.requires_grad = True

        for param in self.Decoder.parameters():
            param.requires_grad = True

        for param in self.EncoderStyle.parameters():
            param.requires_grad = True

    ## ---------------------------------------------------
    ##  Style loss, Content loss, Desaturation

    def style_loss(self, features_coded_Image, features_style):
        style_loss = 0.
        for ft_x, ft_s in zip(features_coded_Image, features_style):
            mean_x, var_x = calc_mean_std(ft_x)
            mean_style, var_style = calc_mean_std(ft_s)

            style_loss = style_loss + self.MSE(mean_x, mean_style)
            style_loss = style_loss + self.MSE(var_x, var_style)

        return style_loss

    def content_loss(self, features_coded_Image, features_content):
        content_loss = self.MSE(features_coded_Image.relu4_1, features_content.relu4_1)
        return content_loss

    def RGB2Gray(self, image):
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        image = (image*std+mean)

        gray = image[:,2:3,:,:]*0.299 + image[:,1:2,:,:]*0.587 + image[:,0:1,:,:]*0.114
        gray = gray.expand(image.size())

        gray = (gray-mean)/std

        return gray

    ## ---------------------------------------------------
    ##  Debug tool for saving results

    def save_figure(self, img, name):
        img = img.data.clone()
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img = (img*std+mean)
        vutils.save_image(img,'result.png')

    ## ---------------------------------------------------
    ##  Functions for the proposed Relaxed Style Loss

    def get_input_optimizer(self, input_img):
        return optim.SGD([input_img.requires_grad_()], lr=self.flow_lr, momentum=0.9)

    def smooth_flow(self, flow, H, W):
        flow = F.interpolate(flow, (H, W), mode='bilinear')
        flow = F.tanh(flow) * self.flow_max
        flow = self.gauss(flow)
        return flow

    ## ---------------------------------------------------
    ##  Inference main function

    def validation(self, cur_frame, style):
        F_cur = self.Encoder(cur_frame)
        F_style = self.EncoderStyle(style)
        return self.Decoder(F_cur, F_style)

    ## ---------------------------------------------------
    ##  Training main function

    def forward(self, content, style):

        ## Content image desaturation
        gray_content = self.RGB2Gray(content)

        ## Style transfer
        F_content = self.Encoder(content)
        F_style = self.EncoderStyle(style)
        styled_result = self.Decoder(F_content, F_style)

        ## Style loss and content loss

        # Get ground truth style/content features
        if self.use_content_loss or self.use_style_loss:
            F_styled = self.Vgg19(styled_result)

            if self.use_content_loss:
                F_content_gt = self.Vgg19(gray_content)

            if self.use_style_loss:
                F_style_gt = self.Vgg19(style)

        # Content loss
        if self.use_content_loss:
            content_loss = self.content_loss(F_styled, F_content_gt)
        else:
            content_loss = 0.

        # Style loss
        if self.use_style_loss:
            if self.relax_style:
                ori_style_loss = self.style_loss(F_styled, F_style_gt)
                
                ''' The proposed Relaxed Style Loss '''

                # Init flow
                B,C,H,W = style.shape
                Flow = torch.zeros([B,2,H//self.flow_scale,W//self.flow_scale]).to(style.device)

                # Optimizer
                optimizer = self.get_input_optimizer(Flow)

                # Records
                best_Bounded_Flow = None
                min_style_loss = ori_style_loss.item()
                min_iter = -1

                # Target loss
                static_F_style = vgg_outputs(F_styled.relu1_1.detach(), 
                                             F_styled.relu2_1.detach(), 
                                             F_styled.relu3_1.detach(), 
                                             F_styled.relu4_1.detach())
                
                tmp_style = style.detach()

                ''' We need to find the best <Flow> to minimize <style_loss>.
                    First, <Flow> is gaussian-smoothed by <self.smooth_flow>.
                    Then, the style image is warped by the flow.
                    Finally, we calculate the <style_loss> and do back-propagation. '''

                for i in range(self.flow_iter):
                    optimizer.zero_grad()

                    # Gaussian-smooth the flow
                    Bounded_Flow = self.smooth_flow(Flow, H, W)

                    # Warp the style image using the flow
                    warpped_tmp_style = warp(tmp_style, Bounded_Flow)

                    # Calculate style loss
                    tmp_F_style_gt = self.Vgg19(warpped_tmp_style)
                    style_loss = self.style_loss(static_F_style, tmp_F_style_gt)
                    
                    style_loss.backward()
                    optimizer.step()

                    if style_loss < min_style_loss:
                        min_style_loss = style_loss.item()
                        best_Bounded_Flow = Bounded_Flow.detach()
                        min_iter = i

                if min_iter != -1:
                    robust_tmp_style = warp(tmp_style, best_Bounded_Flow)
                    robust_F_style_gt = self.Vgg19(robust_tmp_style)
                    new_style_loss = self.style_loss(F_styled, robust_F_style_gt)

                    del best_Bounded_Flow

                else:
                    robust_tmp_style = style
                    new_style_loss = ori_style_loss
            else:
                new_style_loss = self.style_loss(F_styled, F_style_gt)
                ori_style_loss = 0.
                robust_tmp_style = None
        else:
            ori_style_loss = 0.
            new_style_loss = 0.
            robust_tmp_style = None

        ## Reconstruction loss
        if self.use_recon_loss:
            recon_content = self.Decoder(F_content, self.EncoderStyle(content))
            recon_style = self.Decoder(self.Encoder(self.RGB2Gray(style)), F_style)
            recon_loss = torch.mean(torch.abs(recon_content-content)) + torch.mean(torch.abs(recon_style-style))
        else:
            recon_loss = 0.
            recon_content = None
            recon_style = None

        return styled_result, robust_tmp_style, recon_content, recon_style, \
                    content_loss, new_style_loss, recon_loss, ori_style_loss



   