import mxnet as mx
from mxnet.gluon import nn, HybridBlock, Parameter
from mxnet.initializer import Xavier


class Vgg16(HybridBlock):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2D(in_channels=3, channels=64, kernel_size=3, strides=1, padding=1)
        self.conv1_2 = nn.Conv2D(in_channels=64, channels=64, kernel_size=3, strides=1, padding=1)

        self.conv2_1 = nn.Conv2D(in_channels=64, channels=128, kernel_size=3, strides=1, padding=1)
        self.conv2_2 = nn.Conv2D(in_channels=128, channels=128, kernel_size=3, strides=1, padding=1)

        self.conv3_1 = nn.Conv2D(in_channels=128, channels=256, kernel_size=3, strides=1, padding=1)
        self.conv3_2 = nn.Conv2D(in_channels=256, channels=256, kernel_size=3, strides=1, padding=1)
        self.conv3_3 = nn.Conv2D(in_channels=256, channels=256, kernel_size=3, strides=1, padding=1)

        self.conv4_1 = nn.Conv2D(in_channels=256, channels=512, kernel_size=3, strides=1, padding=1)
        self.conv4_2 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)
        self.conv4_3 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)

        self.conv5_1 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)
        self.conv5_2 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)
        self.conv5_3 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)

    def hybrid_forward(self,F, X):
        h = F.Activation(self.conv1_1(X), act_type='relu')
        h = F.Activation(self.conv1_2(h), act_type='relu')
        relu1_2 = h
        h = F.Pooling(h, pool_type='max', kernel=(2, 2), stride=(2, 2))

        h = F.Activation(self.conv2_1(h), act_type='relu')
        h = F.Activation(self.conv2_2(h), act_type='relu')
        relu2_2 = h
        h = F.Pooling(h, pool_type='max', kernel=(2, 2), stride=(2, 2))

        h = F.Activation(self.conv3_1(h), act_type='relu')
        h = F.Activation(self.conv3_2(h), act_type='relu')
        h = F.Activation(self.conv3_3(h), act_type='relu')
        relu3_3 = h
        h = F.Pooling(h, pool_type='max', kernel=(2, 2), stride=(2, 2))

        h = F.Activation(self.conv4_1(h), act_type='relu')
        h = F.Activation(self.conv4_2(h), act_type='relu')
        h = F.Activation(self.conv4_3(h), act_type='relu')
        relu4_3 = h

        return [relu1_2, relu2_2, relu3_3, relu4_3]

    def _init_weights(self, fixed=False, pretrain_path=None, ctx=None):
        if pretrain_path is not None:
            print('Loading parameters from {} ...'.format(pretrain_path))
            self.collect_params().load(pretrain_path, ctx=ctx)
            if fixed:
                print('Setting parameters of VGG16 to fixed ...')
                for param in self.collect_params().values():
                    param.grad_req = 'null'
        else:
            self.initialize(mx.initializer.Xavier(), ctx=ctx)



return_layers_id = {
    11: [6, 13, 20, 27],
    16: [5, 12, 22, 42]
}

vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
            13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
            16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
            19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}



class VGG(HybridBlock):
    r"""VGG model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each feature block.
    filters : list of int
        Numbers of filters in each feature block. List length should match the layers.
    classes : int, default 1000
        Number of classification classes.
    batch_norm : bool, default False
        Use batch normalization.
    """
    def __init__(self, num_layers, batch_norm=True, pretrain_path=None, ctx=None, **kwargs):
        super(VGG, self).__init__(**kwargs)
        layers, filters = vgg_spec[num_layers]
        self.features = self._make_features(layers, filters, batch_norm)
        self.features.add(nn.Dense(4096, activation='relu',
                                   weight_initializer='normal',
                                   bias_initializer='zeros'))
        self.features.add(nn.Dropout(rate=0.5))
        self.features.add(nn.Dense(4096, activation='relu',
                                   weight_initializer='normal',
                                   bias_initializer='zeros'))
        self.features.add(nn.Dropout(rate=0.5))
        self.output = nn.Dense(1000,
                               weight_initializer='normal',
                               bias_initializer='zeros')
        self.return_id_list = return_layers_id[num_layers]
        if pretrain_path is not None and os.path.isfile(pretrain_path):
            self.pretrained = True
            self.load_pretrained_param(pretrain_path, ctx)

    def _make_features(self, layers, filters, batch_norm):
        featurizer = nn.HybridSequential()
        for i, num in enumerate(layers):
            for _ in range(num):
                featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1,
                                         weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                         bias_initializer='zeros'))
                if batch_norm:
                    featurizer.add(nn.BatchNorm())
                featurizer.add(nn.Activation('relu'))
            featurizer.add(nn.MaxPool2D(strides=2))
        return featurizer

    def hybrid_forward(self, F, x):
        return_ = []
        for id, layer in enumerate(self.features):
            if isinstance(layer, nn.basic_layers.Dense):
                break
            x = layer(x)
            if id in self.return_id_list:
                return_.append(x)
        #x = self.features(x)
        #x = self.output(x)
        return return_

    def load_pretrained_param(self, pretrain_path, ctx):
        print('Loading Parameters from {}'.format(pretrain_path))
        self.load_parameters(pretrain_path, ctx=ctx)