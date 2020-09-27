###############################################################################
# This script is modified from 
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
###############################################################################

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


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


def init_net(net, init_type='normal', init_gain=0.02):
    init_weights(net, init_type, gain=init_gain)
    return net

def define_G(input_nc, output_nc):
    norm_layer = get_norm_layer(norm_type='batch')
    netG = UnetGenerator(input_nc, output_nc, 8, 64, norm_layer=norm_layer, use_dropout=False)
    return netG
    # return init_net(netG, 'normal', 0.02)


def define_D(input_nc):
    norm_layer = get_norm_layer(norm_type='batch')
    netD = NLayerDiscriminator(input_nc, 64, n_layers=3, norm_layer=norm_layer, use_sigmoid=False)
    return netD
    # return init_net(netD, 'normal', 0.02)


##############################################################################
# Classes
##############################################################################


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

# Progressive growing U-net
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        self.InnerBlocks = unet_block

        self.down_32 = DownBlock(ngf * 4, ngf * 8)
        self.down_64 = DownBlock(ngf * 2, ngf * 4)
        self.down_128 = DownBlock(ngf, ngf * 2)
        self.down_256 = DownBlock(input_nc, ngf, outermost=True)

        self.up_32 = UpBlock(ngf * 16, ngf * 4)
        self.up_64 = UpBlock(ngf * 8, ngf * 2)
        self.up_128 = UpBlock(ngf * 4, ngf)
        self.up_256 = UpBlock(ngf * 2, output_nc, outermost=True)

        self.temp_down_32 = DownBlock(input_nc, ngf * 8, outermost=True)
        self.temp_down_64 = DownBlock(input_nc, ngf * 4, outermost=True)
        self.temp_down_128 = DownBlock(input_nc, ngf * 2, outermost=True)

        self.temp_up_32 = UpBlock(ngf * 16, output_nc, outermost=True)
        self.temp_up_64 = UpBlock(ngf * 4, output_nc, outermost=True)
        self.temp_up_128 = UpBlock(ngf * 2, output_nc, outermost=True)

    def forward(self, x, size):
        if size == 32:
            x = self.temp_down_32(x)
            x = self.InnerBlocks(x)
            return self.temp_up_32(x)
        elif size == 64:
            x = self.temp_down_64(x)
            x_32 = self.down_32(x)
            x_32 = self.up_32(self.InnerBlocks(x_32))
            return self.temp_up_64(x_32)
        elif size == 128:
            x_128 = self.temp_down_128(x)
            x_64 = self.down_64(x_128)
            x_32 = self.down_32(x_64)
            x_32 = self.up_32(self.InnerBlocks(x_32))
            x_64 = self.up_64(torch.cat([x_64, x_32], 1))
            return self.temp_up_128(x_64)
        elif size == 256:
            x_256 = self.down_256(x)
            x_128 = self.down_128(x_256)
            x_64 = self.down_64(x_128)
            x_32 = self.down_32(x_64)
            x_32 = self.up_32(self.InnerBlocks(x_32))
            x_64 = self.up_64(torch.cat([x_64, x_32], 1))
            x_128 = self.up_128(torch.cat([x_128, x_64], 1))
            return self.up_256(torch.cat([x_256, x_128], 1))
        else:
            assert(size,'only supports resolution: 32, 64, 128, 256.')


class DownBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(DownBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downconv = nn.Conv2d(input_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(outer_nc)

        if outermost:
            model = [downconv]
        elif innermost:
            model = [downrelu, downconv]
        else:
            model = [downrelu, downconv, downnorm]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class UpBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UpBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        upconv = nn.ConvTranspose2d(input_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            model = [uprelu, upconv, nn.Tanh()]
        elif innermost:
            model = [uprelu, upconv, upnorm]
        else:
            if use_dropout:
                model = [uprelu, upconv, upnorm] + [nn.Dropout(0.5)]
            else:
                model = [uprelu, upconv, upnorm]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# PatchGAN discriminator.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input) 