import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from math import floor, log2
from functools import partial
from linear_attention_transformer import ImageLinearAttention

###

from random import random


import numpy as np
import torch.nn.functional as F


###

from models.networks_SPADE.base_network import BaseNetwork
from models.networks_SPADE.architecture import ResnetBlock as ResnetBlock
from models.networks_SPADE.architecture import SPADEResnetBlock as SPADEResnetBlock


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def define_SPADE(opt,gpu_ids):
    if('spade8' in opt.netG):
        net = SPADE8Generator(input_nc=1, output_nc=1, num_downs = 8, ngf=1, norm_layer='abc', use_dropout=False, opt=opt)
    elif('spade6' in opt.netG):
        net = SPADE6Generator(input_nc=1, output_nc=1, num_downs = 8, ngf=1, norm_layer='abc', use_dropout=False, opt=opt)
    else:
        net = SPADEGenerator(input_nc=1, output_nc=1, num_downs = 8, ngf=1, norm_layer='abc', use_dropout=False, opt=opt)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        #net = torch.nn.DataParallel(net, gpu_ids) 
    net.init_weights()
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_9blocksup':
        net = ResnetGeneratorUp(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_768':
        net = UNet768(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_768_sigm':
        net = UNet768Sigm(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_spade':
        net = UNet768PIXSPADE(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_spade8sm':
        net = UNet768PIXSPADE8SM(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'conditional': #conditional patchGAN
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'unet':
        net = UnetDiscriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


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
    
class UnetGANLoss(nn.Module):
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
        super(UnetGANLoss, self).__init__()
        self.register_buffer('real_label_1', torch.tensor(target_real_label))
        self.register_buffer('real_label_2', torch.tensor(np.ones((1,256,256))))
        self.register_buffer('fake_label_1', torch.tensor(target_fake_label))
        self.register_buffer('fake_label_2', torch.tensor(np.zeros((1,256,256))))

        self.loss_1 = nn.BCEWithLogitsLoss()
        self.loss_2 = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction_1, prediction_2, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor_1 = self.real_label_1
            target_tensor_2 = self.real_label_2
        else:
            target_tensor_1 = self.fake_label_1
            target_tensor_2 = self.fake_label_2
        return target_tensor_1.expand_as(prediction_1), target_tensor_2.expand_as(prediction_2)

    def __call__(self, prediction_1, prediction_2, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """

        target_tensor_1, target_tensor_2 = self.get_target_tensor(prediction_1, prediction_2, target_is_real)
        loss_1 = self.loss_1(prediction_1, target_tensor_1)
        loss_2 = self.loss_2(prediction_2, target_tensor_2)

                
        loss = loss_1.mean()+loss_2.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    
class ResnetGeneratorUp(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGeneratorUp, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor = 2, mode='nearest'),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0),]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
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
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
        
#%%%      Unet from DeepMact
            
        
class ConvBnRelu2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, output_padding=1, dilation=1, stride=1, groups=1, is_bn=True, is_relu=True, is_decoder=False):
        super(ConvBnRelu2d, self).__init__()
        if is_decoder:
            self.transpConv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, output_padding=output_padding, stride=stride, dilation=dilation, groups=groups, bias=False)
            self.conv = None
        else:
            self.transpConv = None
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = torch.nn.ReLU(inplace=True)
        if is_bn is False: self.bn = None
        if is_relu is False: self.relu = None

    def forward(self, x):
        if self.conv is None:
            x = self.transpConv(x)
        elif self.transpConv is None:
            x = self.conv(x)
            
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

            
class StackEncoder(torch.nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3, stride=1):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = torch.nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
        )

    def forward(self, x):
        y = self.encode(x)
        y_small = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class StackDecoder(torch.nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, stride=1):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = torch.nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels,                  y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels,                  y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
        )

    def forward(self, x_big, x):
        N, C, H, W = x_big.size()
        y = torch.nn.functional.upsample(x, size=(H, W), mode='bilinear', align_corners=True)
        y = torch.cat([y, x_big], 1)
        y = self.decode(y)
        return y
# 768
class UNet768(torch.nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNet768, self).__init__()
        #    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        # C, H, W = in_shape
        # assert(C==3)
        self.output_nc = output_nc

        # 1024
        self.down1 = StackEncoder(input_nc, 24, kernel_size=3)    # Channels: 1 in,   24 out;  Image size: 300 in, 150 out
        self.down2 = StackEncoder(24, 64, kernel_size=3)   # Channels: 24 in,  64 out;  Image size: 150 in, 75 out
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # Channels: 64 in,  128 out; Image size: 75 in,  38 out
        self.down4 = StackEncoder(128, 256, kernel_size=3) # Channels: 128 in, 256 out; Image size: 38 in,  19 out
        self.down5 = StackEncoder(256, 512, kernel_size=3) # Channels: 256 in, 512 out; Image size: 19 in,  10 out
        self.down6 = StackEncoder(512, 768, kernel_size=3) # Channels: 512 in, 768 out; Image size: 10 in,  5 out

        self.center = torch.nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1), # Channels: 768 in, 768 out; Image size: 5 in,  5 out
        )

        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768, 768, 512, kernel_size=3) # Channels: 768+768 = 1536 in, 512 out; Image size: 5 in,   10 out
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3) # Channels: 512+512 = 1024 in, 256 out; Image size: 10 in,  19 out
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3) # Channels: 256+256 = 512  in, 128 out; Image size: 19 in,  38 out
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # Channels: 128+128 = 256  in, 64  out; Image size: 38 in,  75 out
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)    # Channels: 64+64   = 128  in, 24  out; Image size: 75 in,  150 out
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)    # Channels: 24+24   = 48   in, 24  out; Image size: 150 in, 300 out
        self.classify = torch.nn.Conv2d(24, output_nc, kernel_size=1, padding=0, stride=1, bias=True) # Channels: 24 in, 1 out; Image size: 300 in, 300 out
        self.final_out = torch.nn.Tanh()

    def _crop_concat(self, upsampled, bypass):
        """
         Crop y to the (h, w) of x and concat them.
         Used for the expansive path.
        Returns:
            The concatenated tensor
        """
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = torch.nn.functional.pad(bypass, (-c, -c, -c, -c))

        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        out = x  # ;print('x    ',x.size())
        #
        down1, out = self.down1(out)  ##;
        #print('down1',down1.shape)  #256
        down2, out = self.down2(out)  # ;
        #print('down2',down2.shape)  #128
        down3, out = self.down3(out)  # ;
        #print('down3',down3.shape)  #64
        down4, out = self.down4(out)  # ;
        #print('down4',down4.shape)  #32
        down5, out = self.down5(out)  # ;
        #print('down5',down5.shape)  #16
        down6, out = self.down6(out)  # ;
        #print('down6',down6.shape)  #8
        pass  # ;
        #print('out  ',out.shape)

        out = self.center(out)
        #print('0',out.shape)
        out = self.up6(down6, out)
        #print('1',out.shape)
        out = self.up5(down5, out)
        #print('2',out.shape)
        out = self.up4(down4, out)
        #print('3',out.shape)
        out = self.up3(down3, out)
        #print('4',out.shape)
        out = self.up2(down2, out)
        #print('5',out.shape)
        out = self.up1(down1, out)
        # 1024
        #print('6',out.shape)
        out = self.final_out(self.classify(out))
        out = torch.reshape(out,(-1, self.output_nc, x.shape[2],x.shape[3]))#, dim=1)
        return out
    
#%%Unet_spade_768_300
        
    
    
#%%sigm


class UNet768Sigm(torch.nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNet768Sigm, self).__init__()
        #    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        # C, H, W = in_shape
        # assert(C==3)
        self.output_nc = output_nc

        # 1024
        self.down1 = StackEncoder(input_nc, 24, kernel_size=3)    # Channels: 1 in,   24 out;  Image size: 300 in, 150 out
        self.down2 = StackEncoder(24, 64, kernel_size=3)   # Channels: 24 in,  64 out;  Image size: 150 in, 75 out
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # Channels: 64 in,  128 out; Image size: 75 in,  38 out
        self.down4 = StackEncoder(128, 256, kernel_size=3) # Channels: 128 in, 256 out; Image size: 38 in,  19 out
        self.down5 = StackEncoder(256, 512, kernel_size=3) # Channels: 256 in, 512 out; Image size: 19 in,  10 out
        self.down6 = StackEncoder(512, 768, kernel_size=3) # Channels: 512 in, 768 out; Image size: 10 in,  5 out

        self.center = torch.nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1), # Channels: 768 in, 768 out; Image size: 5 in,  5 out
        )

        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768, 768, 512, kernel_size=3) # Channels: 768+768 = 1536 in, 512 out; Image size: 5 in,   10 out
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3) # Channels: 512+512 = 1024 in, 256 out; Image size: 10 in,  19 out
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3) # Channels: 256+256 = 512  in, 128 out; Image size: 19 in,  38 out
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # Channels: 128+128 = 256  in, 64  out; Image size: 38 in,  75 out
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)    # Channels: 64+64   = 128  in, 24  out; Image size: 75 in,  150 out
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)    # Channels: 24+24   = 48   in, 24  out; Image size: 150 in, 300 out
        self.classify = torch.nn.Conv2d(24, output_nc, kernel_size=1, padding=0, stride=1, bias=True) # Channels: 24 in, 1 out; Image size: 300 in, 300 out
        self.final_out = torch.nn.Sigmoid()

    def _crop_concat(self, upsampled, bypass):
        """
         Crop y to the (h, w) of x and concat them.
         Used for the expansive path.
        Returns:
            The concatenated tensor
        """
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = torch.nn.functional.pad(bypass, (-c, -c, -c, -c))

        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        out = x  # ;print('x    ',x.size())
        #
        down1, out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2, out = self.down2(out)  # ;print('down2',down2.size())  #128
        down3, out = self.down3(out)  # ;print('down3',down3.size())  #64
        down4, out = self.down4(out)  # ;print('down4',down4.size())  #32
        down5, out = self.down5(out)  # ;print('down5',down5.size())  #16
        down6, out = self.down6(out)  # ;print('down6',down6.size())  #8
        pass  # ;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        # 1024

        out = self.final_out(self.classify(out))
        out = torch.reshape(out,(1, self.output_nc, 256,256))#, dim=1)
        return out





class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


#%% Unet as Disdef random_hflip(tensor, prob):
        

def DiffAugment(x, types=[]):
    for p in types:
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous(memory_format = torch.contiguous_format)

def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x

def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x

def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x

def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous(memory_format = torch.contiguous_format)
    return x

def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
    
def random_float(lo, hi):
    return lo + (hi - lo) * random()

def random_crop_and_resize(tensor, scale):
    b, c, h, _ = tensor.shape
    new_width = int(h * scale)
    delta = h - new_width
    h_delta = int(random() * delta)
    w_delta = int(random() * delta)
    cropped = tensor[:, :, h_delta:(h_delta + new_width), w_delta:(w_delta + new_width)].clone()
    return F.interpolate(cropped, size=(h, h), mode='bilinear')

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size, types):
        super().__init__()
        self.D = D
        self.types = types

    def forward(self, images, prob = 0., detach = False):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=self.types)

        if detach:
            images.detach_()

        return self.D(images), images
    
    
def leaky_relu(p=0.2):
    return nn.LeakyReLU(p)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class Flatten(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, x):
        return x.flatten(self.index)

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.fn(x) * self.g        
    
def double_conv(chan_in, chan_out):
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, 3, padding=1),
        leaky_relu(),
        nn.Conv2d(chan_out, chan_out, 3, padding=1),
        leaky_relu()
    )
        
class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = double_conv(input_channels, filters)
        self.down = nn.Conv2d(filters, filters, 3, padding = 1, stride = 2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        x = x + res
        return x, unet_res
    

# one layer of self-attention and feedforward, for images

attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries = True))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])
            
class UpBlock(nn.Module):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.conv_res = nn.ConvTranspose2d(input_channels // 2, filters, 1, stride = 2)
        self.net = double_conv(input_channels, filters)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)
        self.input_channels = input_channels
        self.filters = filters

    def forward(self, x, res):
        *_, h, w = x.shape
        conv_res = self.conv_res(x, output_size = (h * 2, w * 2))
        x = self.up(x)
        x = torch.cat((x, res), dim=1)
        x = self.net(x)
        x = x + conv_res
        return x
        
class UnetDiscriminator(nn.Module):
    def __init__(self, image_size=256, network_capacity = 16, transparent = False, fmap_max = 256):
        super().__init__()
        num_layers = int(log2(image_size) - 3)
        num_init_filters = 2# if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        filters[-1] = filters[-2]

        chan_in_out = list(zip(filters[:-1], filters[1:]))
        chan_in_out = list(map(list, chan_in_out))

        print('Channels',chan_in_out)
        down_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DownBlock(in_chan, out_chan, downsample = is_not_last)
            down_blocks.append(block)

            attn_fn = attn_and_ff(out_chan)
            attn_blocks.append(attn_fn)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        last_chan = filters[-1]

        self.to_logit = nn.Sequential(
            leaky_relu(),
            nn.AvgPool2d(image_size // (2 ** num_layers)),
            Flatten(1),
            nn.Linear(last_chan, 1)
        )

        self.conv = double_conv(last_chan, last_chan)

        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), dec_chan_in_out)))
        self.conv_out = nn.Conv2d(2, 1, 1)

    def forward(self, x):
        
        #print('Input shape:', x.shape)
        b, *_ = x.shape

        residuals = []
        i=0
        for (down_block, attn_block) in zip(self.down_blocks, self.attn_blocks):
            #print('Step', i, x.shape)
            i=i+1
            x, unet_res = down_block(x)
            residuals.append(unet_res)

            if attn_block is not None:
                x = attn_block(x)

        x = self.conv(x) + x
        enc_out = self.to_logit(x)

        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            #print('in up blocK', x.shape)
            x = up_block(x, res)

        dec_out = self.conv_out(x)
        return enc_out.squeeze(), dec_out


#%%  SPADE RESNET
        
    
class SPADEGenerator(BaseNetwork):

    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_layer=nn.BatchNorm2d, use_dropout=False,opt=None):
        super(SPADEGenerator, self).__init__()
        self.opt = opt
        self.opt.num_upsampling_layers = 'normal'
        self.opt.norm_G = 'spectralspadesyncbatch3x3'
        self.opt.ngf = 64
        self.opt.semantic_nc = 2
        self.opt.use_vae = False
        self.opt.crop_size = 256
        self.opt.normG = 'spectralinstance'
        self.opt.aspect_ratio = 1.0
        nf = self.opt.ngf
        opt = self.opt

        self.sw, self.sh = self.compute_latent_vector_size(opt)


        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = self.opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        #print('0,', x.shape)
        x = self.head_0(x, seg)
        #print('1,', x.shape)
        x = self.up(x)
        #print('2', x.shape)
        x = self.G_middle_0(x, seg)
        #print('3,', x.shape)
        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
        #print('4,', x.shape)
        #x = self.G_middle_1(x, seg)
        output_5 = x
        #print('5,', x.shape)
        x = self.up(x)
        output_6 = x
        #print('6,', x.shape)
        x = self.up_0(x, seg)
        #print('7,', x.shape)
        x = self.up(x)
        #print('8,', x.shape)
        x = self.up_1(x, seg)
        output_9 = x
        #print('9,', x.shape)
        x = self.up(x)
        #print('10,', x.shape)
        x = self.up_2(x, seg)
        #print('11,', x.shape)
        output_11 = x
        x = self.up(x)
       # print('12,', x.shape)
        x = self.up_3(x, seg)
        #print('13,', x.shape)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
        #print('14,', x.shape)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
       # print('15,', x.shape)
        output_15 = x
        #x = F.tanh(x)
        #print('16,', x.shape)

        return output_5,output_6,output_9,output_11,output_15
    
#%% spade8
        
class SPADE8Generator(BaseNetwork):

    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_layer=nn.BatchNorm2d, use_dropout=False,opt=None):
        super(SPADE8Generator, self).__init__()
        self.opt = opt
        self.opt.num_upsampling_layers = 'normal'
        self.opt.norm_G = 'spectralspadesyncbatch3x3'
        self.opt.ngf = 8
        self.opt.semantic_nc = 2
        self.opt.use_vae = False
        self.opt.crop_size = 256
        self.opt.normG = 'spectralinstance'
        self.opt.aspect_ratio = 1.0
        nf = self.opt.ngf
        opt = self.opt

        self.sw, self.sh = self.compute_latent_vector_size(opt)


        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = self.opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        #print('0,', x.shape)
        x = self.head_0(x, seg)
        #print('1,', x.shape)
        x = self.up(x)
        #print('2', x.shape)
        x = self.G_middle_0(x, seg)
        #print('3,', x.shape)
        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
        #print('4,', x.shape)
        x = self.G_middle_1(x, seg)
        output_5 = x
        #print('5,', x.shape)
        x = self.up(x)
        output_6 = x
        #print('6,', x.shape)
        x = self.up_0(x, seg)
        #print('7,', x.shape)
        x = self.up(x)
        #print('8,', x.shape)
        x = self.up_1(x, seg)
        output_9 = x
        #print('9,', x.shape)
        x = self.up(x)
        #print('10,', x.shape)
        x = self.up_2(x, seg)
        #print('11,', x.shape)
        output_11 = x
        '''this can be removed'''
        x = self.up(x)
        #print('12,', x.shape)
        x = self.up_3(x, seg)
        #print('13,', x.shape)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
        #print('14,', x.shape)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        #print('15,', x.shape)
        output_15 = x
        #x = F.tanh(x)
        #print('16,', x.shape)
        '''til here'''
        return output_5,output_6,output_9,output_11,output_15
    
#%%
class SPADE6Generator(BaseNetwork):

    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_layer=nn.BatchNorm2d, use_dropout=False,opt=None):
        super(SPADE6Generator, self).__init__()
        self.opt = opt
        self.opt.num_upsampling_layers = 'normal'
        self.opt.norm_G = 'spectralspadesyncbatch3x3'
        self.opt.ngf = 6
        self.opt.semantic_nc = 2
        self.opt.use_vae = False
        self.opt.crop_size = 300
        self.opt.normG = 'spectralinstance'
        self.opt.aspect_ratio = 1.0
        nf = self.opt.ngf
        opt = self.opt

        self.sw, self.sh = self.compute_latent_vector_size(opt)


        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = 10#self.opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        print('0,', x.shape)
        x = self.head_0(x, seg)
        print('1,', x.shape)
        x = self.up(x)
        print('2', x.shape)
        x = self.G_middle_0(x, seg)
        print('3,', x.shape)
        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
        print('4,', x.shape)
        x = self.G_middle_1(x, seg)
        output_5 = x
        print('5,', x.shape)
        x = self.up(x)
        output_6 = x
        print('6,', x.shape)
        x = self.up_0(x, seg)
        print('7,', x.shape)
        x = self.up(x)
        print('8,', x.shape)
        x = self.up_1(x, seg)
        output_9 = x
        print('9,', x.shape)
        x = self.up(x)
        print('10,', x.shape)
        x = self.up_2(x, seg)
        print('11,', x.shape)
        output_11 = x
        x = self.up(x)
        print('12,', x.shape)
        x = self.up_3(x, seg)
        print('13,', x.shape)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
        print('14,', x.shape)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        print('15,', x.shape)
        output_15 = x
        #x = F.tanh(x)
        print('16,', x.shape)

        return output_5,output_6,output_9,output_11,output_15

#%% For the PIX2SPADE
        
class UNet768PIXSPADE(torch.nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNet768PIXSPADE, self).__init__()
        #    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        # C, H, W = in_shape
        # assert(C==3)
        print('UNET 768 SPADE')
        self.output_nc = output_nc

        # 1024
        self.down1 = StackEncoder(1, 24, kernel_size=3)    # Channels: 1 in,   24 out;  Image size: 300 in, 150 out
        self.down2 = StackEncoder(24, 64, kernel_size=3)   # Channels: 24 in,  64 out;  Image size: 150 in, 75 out
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # Channels: 64 in,  128 out; Image size: 75 in,  38 out
        self.down4 = StackEncoder(128, 256, kernel_size=3) # Channels: 128 in, 256 out; Image size: 38 in,  19 out
        self.down5 = StackEncoder(256, 512, kernel_size=3) # Channels: 256 in, 512 out; Image size: 19 in,  10 out
        self.down6 = StackEncoder(512, 768, kernel_size=3) # Channels: 512 in, 768 out; Image size: 10 in,  5 out

        self.center = torch.nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1), # Channels: 768 in, 768 out; Image size: 5 in,  5 out
        )

        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768, 768, 512, kernel_size=3) # Channels: 768+768 = 1536 in, 512 out; Image size: 5 in,   10 out
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3) # Channels: 512+512 = 1024 in, 256 out; Image size: 10 in,  19 out
        self.up4 = StackDecoder(256+1024, 256, 128, kernel_size=3) # Channels: 256+256 = 512  in, 128 out; Image size: 19 in,  38 out
        self.up3 = StackDecoder(128+1024, 128, 64, kernel_size=3)  # Channels: 128+128 = 256  in, 64  out; Image size: 38 in,  75 out
        self.up2 = StackDecoder(64+256, 64, 24, kernel_size=3)    # Channels: 64+64   = 128  in, 24  out; Image size: 75 in,  150 out
        self.up1 = StackDecoder(24+128, 24, 24, kernel_size=3)    # Channels: 24+24   = 48   in, 24  out; Image size: 150 in, 300 out
        self.classify = torch.nn.Conv2d(24+3, output_nc, kernel_size=1, padding=0, stride=1, bias=True) # Channels: 24 in, 1 out; Image size: 300 in, 300 out
        self.final_out = torch.nn.Tanh()

    def _crop_concat(self, upsampled, bypass):
        """
         Crop y to the (h, w) of x and concat them.
         Used for the expansive path.
        Returns:
            The concatenated tensor
        """
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = torch.nn.functional.pad(bypass, (-c, -c, -c, -c))

        return torch.cat((upsampled, bypass), 1)

    def forward(self,x, input_to_net):
        #print(input_to_net.shape)
        output_5,output_6,output_9,output_11,output_15 = input_to_net
        
        #print(x.shape)
        
        out = x  # ;print('x    ',x.size())
        #
        down1, out = self.down1(out)  ##;
        #print('down1',down1.shape)  #256
        down2, out = self.down2(out)  # ;
        #print('down2',down2.shape)  #128
        down3, out = self.down3(out)  # ;
        #print('down3',down3.shape)  #64
        down4, out = self.down4(out)  # ;
        #print('down4',down4.shape)  #32
        down5, out = self.down5(out)  # ;
        #print('down5',down5.shape)  #16
        down6, out = self.down6(out)  # ;
        #print('down6',down6.shape)  #8
        pass  # ;
        #print('out  ',out.shape)

        out = self.center(out)
        #print('0',out.shape)
        out = self.up6(down6, out)
        #print('1',out.shape)
        out = self.up5(down5, out)
        out = torch.cat((out,output_5 ),1 )
        #print('2',out.shape)
        out = self.up4(down4, out)
        out = torch.cat((out,output_6 ),1 )
        #print('3',out.shape)
        out = self.up3(down3, out)
        out = torch.cat((out,output_9 ),1 )
        #print('4',out.shape)
        out = self.up2(down2, out)
        out = torch.cat((out,output_11 ),1 )
        #print('5',out.shape)
        out = self.up1(down1, out)
        # 1024
        out = torch.cat((out,output_15 ),1 )
        #print('6',out.shape)
        out = self.final_out(self.classify(out))
        out = torch.reshape(out,(-1, self.output_nc, 256,256))#, dim=1)
        return out

#%%Unet for spade8
    
class UNet768PIXSPADE8SM(torch.nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNet768PIXSPADE8SM, self).__init__()
        #    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        # C, H, W = in_shape
        # assert(C==3)
        print('UNET 768 SPADE')
        self.output_nc = output_nc

        # 1024
        self.down1 = StackEncoder(1, 24, kernel_size=3)    # Channels: 1 in,   24 out;  Image size: 300 in, 150 out
        self.down2 = StackEncoder(24, 64, kernel_size=3)   # Channels: 24 in,  64 out;  Image size: 150 in, 75 out
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # Channels: 64 in,  128 out; Image size: 75 in,  38 out
        self.down4 = StackEncoder(128, 256, kernel_size=3) # Channels: 128 in, 256 out; Image size: 38 in,  19 out
        self.down5 = StackEncoder(256, 512, kernel_size=3) # Channels: 256 in, 512 out; Image size: 19 in,  10 out
        self.down6 = StackEncoder(512, 768, kernel_size=3) # Channels: 512 in, 768 out; Image size: 10 in,  5 out

        self.center = torch.nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1), # Channels: 768 in, 768 out; Image size: 5 in,  5 out
        )

        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768, 768, 512, kernel_size=3) # Channels: 768+768 = 1536 in, 512 out; Image size: 5 in,   10 out
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3) # Channels: 512+512 = 1024 in, 256 out; Image size: 10 in,  19 out
        self.up4 = StackDecoder(256+128, 256, 128, kernel_size=3) # Channels: 256+256 = 512  in, 128 out; Image size: 19 in,  38 out
        self.up3 = StackDecoder(128+128, 128, 64, kernel_size=3)  # Channels: 128+128 = 256  in, 64  out; Image size: 38 in,  75 out
        self.up2 = StackDecoder(64+32, 64, 24, kernel_size=3)    # Channels: 64+64   = 128  in, 24  out; Image size: 75 in,  150 out
        self.up1 = StackDecoder(24+16, 24, 24, kernel_size=3)    # Channels: 24+24   = 48   in, 24  out; Image size: 150 in, 300 out
        self.classify = torch.nn.Conv2d(24, output_nc, kernel_size=1, padding=0, stride=1, bias=True) # Channels: 24 in, 1 out; Image size: 300 in, 300 out
        self.final_out = torch.nn.Tanh()

    def _crop_concat(self, upsampled, bypass):
        """
         Crop y to the (h, w) of x and concat them.
         Used for the expansive path.
        Returns:
            The concatenated tensor
        """
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = torch.nn.functional.pad(bypass, (-c, -c, -c, -c))

        return torch.cat((upsampled, bypass), 1)

    def forward(self,x, input_to_net):
        #print(input_to_net.shape)
        output_5,output_6,output_9,output_11,output_15 = input_to_net
        
        #print(x.shape)
        
        out = x  # ;print('x    ',x.size())
        #
        down1, out = self.down1(out)  ##;
        #print('down1',down1.shape)  #256
        down2, out = self.down2(out)  # ;
        #print('down2',down2.shape)  #128
        down3, out = self.down3(out)  # ;
        #print('down3',down3.shape)  #64
        down4, out = self.down4(out)  # ;
        #print('down4',down4.shape)  #32
        down5, out = self.down5(out)  # ;
        #print('down5',down5.shape)  #16
        down6, out = self.down6(out)  # ;
        #print('down6',down6.shape)  #8
        pass  # ;
        #print('out  ',out.shape)

        out = self.center(out)
        #print('0',out.shape)
        out = self.up6(down6, out)
        #print('1',out.shape)
        out = self.up5(down5, out)
        out = torch.cat((out,output_5 ),1 )
        #print('2',out.shape)
        out = self.up4(down4, out)
        out = torch.cat((out,output_6 ),1 )
        #print('3',out.shape)
        out = self.up3(down3, out)
        out = torch.cat((out,output_9 ),1 )
        #print('4',out.shape)
        out = self.up2(down2, out)
        out = torch.cat((out,output_11 ),1 )
        #print('5',out.shape)
        out = self.up1(down1, out)
        # 1024
        #out = torch.cat((out,output_15 ),1 )
        #print('6',out.shape)
        out = self.final_out(self.classify(out))
        out = torch.reshape(out,(-1, self.output_nc, 256,256))#, dim=1)
        return out

        

    

