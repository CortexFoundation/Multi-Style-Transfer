import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
from mxnet.gluon import nn, Block, HybridBlock, Parameter
from mxnet.base import numeric_types
from mxnet.initializer import Xavier


class InstanceNorm(HybridBlock):
    def __init__(self, axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=False,
                 beta_initializer='zeros', gamma_initializer='ones',
                 in_channels=0, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self._kwargs = {'eps': epsilon}
        if in_channels != 0:
            self.in_channels = in_channels
        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True)

    def hybrid_forward(self, F, x, gamma, beta):
        return F.InstanceNorm(x, gamma, beta,
                           name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{name}({content}'
        if hasattr(self, 'in_channels'):
            s += ', in_channels={0}'.format(self.in_channels)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))


class ReflectancePadding(HybridBlock):
    def __init__(self, pad_width=None, **kwargs):
        super(ReflectancePadding, self).__init__(**kwargs)
        self.pad_width = pad_width
        
    def hybrid_forward(self, F, x):
        return F.pad(x, mode='reflect', pad_width=self.pad_width)


class Bottleneck(HybridBlock):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=InstanceNorm):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2D(in_channels=inplanes, 
                                            channels=planes * self.expansion,
                                            kernel_size=1, strides=(stride, stride))
        self.conv_block = nn.HybridSequential()
        with self.conv_block.name_scope():
            self.conv_block.add(norm_layer(in_channels=inplanes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(nn.Conv2D(in_channels=inplanes, channels=planes, 
                                 kernel_size=1))
            self.conv_block.add(norm_layer(in_channels=planes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(ConvLayer(planes, planes, kernel_size=3, 
                stride=stride))
            self.conv_block.add(norm_layer(in_channels=planes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(nn.Conv2D(in_channels=planes, 
                                 channels=planes * self.expansion, 
                                 kernel_size=1))
        
    def hybrid_forward(self, F, x):
        if self.downsample is not None:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)



class UpBottleneck(HybridBlock):
    """ Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """
    def __init__(self, inplanes, planes, stride=2, norm_layer=InstanceNorm):
        super(UpBottleneck, self).__init__()
        self.expansion = 4
        self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion,
                                                      kernel_size=1, stride=1, upsample=stride)
        self.conv_block = nn.HybridSequential()
        with self.conv_block.name_scope():
            self.conv_block.add(norm_layer(in_channels=inplanes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(nn.Conv2D(in_channels=inplanes, channels=planes, 
                                kernel_size=1))
            self.conv_block.add(norm_layer(in_channels=planes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride))
            self.conv_block.add(norm_layer(in_channels=planes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(nn.Conv2D(in_channels=planes, 
                                channels=planes * self.expansion, 
                                kernel_size=1))

    def hybrid_forward(self, F, x):
        return  self.residual_layer(x) + self.conv_block(x)


class ConvLayer(HybridBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = int(np.floor(kernel_size / 2))
        self.pad = ReflectancePadding(pad_width=(0,0,0,0,padding,padding,padding,padding))
        self.conv2d = nn.Conv2D(in_channels=in_channels, channels=out_channels, 
                                kernel_size=kernel_size, strides=(stride,stride),
                                padding=0)

    def hybrid_forward(self, F, x):
        x = self.pad(x)
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(HybridBlock):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
            stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        """
        if upsample:
            self.upsample_layer = torch.nn.UpsamplingNearest2d(scale_factor=upsample)
        """
        self.reflection_padding = int(np.floor(kernel_size / 2))
        self.conv2d = nn.Conv2D(in_channels=in_channels, 
                                channels=out_channels, 
                                kernel_size=kernel_size, strides=(stride,stride),
                                padding=self.reflection_padding)

    def hybrid_forward(self, F, x):
        if self.upsample:
            x = F.UpSampling(x, scale=self.upsample, sample_type='nearest')
        """
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        """
        out = self.conv2d(x)
        return out


class GramMatrix(HybridBlock):
    def hybrid_forward(self, F, x):
        gram = self.gram_matrix(F, x)
        return gram

    def gram_matrix(self, F, y): # out B, C, C
        #(b, ch, h, w) = y.shape
        weight = F.ones_like(y)
        weight = F.sum(weight, axis=(1,2,3))
        weight = F.reshape(weight, (-1, 1, 1))
        features = y.reshape((0, 0, -1)) # (B, C, H*W)
        gram = F.batch_dot(features, features, transpose_b=True) # (B, C, C)
        gram = F.broadcast_div(gram, weight)
        return gram

class Inspiration(HybridBlock):
    """ Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953
    """
    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        # B is equal to 1 or input mini_batch
        self.C = C
        # self.gram = self.params.get('gram', allow_deferred_init=True)
        self.weight = self.params.get('weight', shape=(1,C,C),
                                      init=mx.initializer.Uniform(),
                                      allow_deferred_init=True)
        # self.gram = None

    # def set_target(self, target):
        # self.gram = target

    def hybrid_forward(self, F, X, gram, weight):

        # assert self.gram is not None, 'InputError: please set target'
        # input X is a 3D feature map
        b = F.zeros_like(X)
        b = F.sum(b, axis=(1,2,3)) # B
        b = F.reshape(b, (-1, 1, 1), name="BBBB") # B, 1, 1

        P = F.batch_dot(F.broadcast_like(weight, gram), gram) # B, C, C
        P1 = F.SwapAxis(P, 1, 2) # B, C, C
        P2 = F.batch_dot(F.broadcast_add(P1, b), X.reshape((0,0,-1)))
        return P2.reshape_like(X)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.C) + ')'




class Net(HybridBlock):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, 
                 norm_layer=InstanceNorm, n_blocks=6, gpu_ids=[]):
        super(Net, self).__init__()
        self.gpu_ids = gpu_ids
        self.gram = GramMatrix()

        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4

        with self.name_scope():
            self.model1 = nn.HybridSequential()
            self.ins = Inspiration(ngf*expansion)
            self.model = nn.HybridSequential()

            self.model1.add(ConvLayer(input_nc, 64, kernel_size=7, stride=1))
            self.model1.add(norm_layer(in_channels=64))
            self.model1.add(nn.Activation('relu'))
            self.model1.add(block(64, 32, 2, 1, norm_layer))
            self.model1.add(block(32*expansion, ngf, 2, 1, norm_layer))

            for i in range(n_blocks):
                self.model.add(block(ngf*expansion, ngf, 1, None, norm_layer))
        
            self.model.add(upblock(ngf*expansion, 32, 2, norm_layer))
            self.model.add(upblock(32*expansion, 16, 2, norm_layer))
            self.model.add(norm_layer(in_channels=16*expansion))
            self.model.add(nn.Activation('relu'))
            self.model.add(ConvLayer(16*expansion, output_nc, kernel_size=7, stride=1))


    def set_target(self, Xs):
        # F = self.model1(Xs)
        # G = self.gram(F)
        # self.ins.set_target(G)
        pass

    def hybrid_forward(self, F, content_image, style_image):
        X = self.model1(content_image)

        F = self.model1(style_image)
        G = self.gram(F)

        features = self.ins(X, G)

        return self.model(features)

    def _save_params(self, path):
        self.collect_params().save(path)