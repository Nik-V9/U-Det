from functools import reduce
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models

class BatchNormalization(layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
    """

    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, inputs, training=None, **kwargs):
        # return super.call, but set training
        if not training:
            return super(BatchNormalization, self).call(inputs, training=False)
        else:
            return super(BatchNormalization, self).call(inputs, training=(not self.freeze))

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config
    
def DepthwiseConvBlock(kernel_size, strides, name, freeze_bn=False):
    f1 = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=False, name='{}_dconv'.format(name))
    f2 = layers.BatchNormalization( name='{}_bn'.format(name))
    f3 = layers.ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))


def ConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = layers.Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                       use_bias=False, name='{}_conv'.format(name))
    f2 = layers.BatchNormalization(name='{}_bn'.format(name))
    f3 = layers.ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))


def build_BiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P3'.format(id))(
            C3)
        P4_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P4'.format(id))(
            C4)
        P5_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P5'.format(id))(
            C5)
        P6_in = ConvBlock(num_channels, kernel_size=3, strides=2, freeze_bn=freeze_bn, name='BiFPN_{}_P6'.format(id))(
            C5)
        P7_in = ConvBlock(num_channels, kernel_size=3, strides=2, freeze_bn=freeze_bn, name='BiFPN_{}_P7'.format(id))(
            P6_in)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P3_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P3'.format(id))(
            P3_in)
        P4_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P4'.format(id))(
            P4_in)
        P5_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P5'.format(id))(
            P5_in)
        P6_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P6'.format(id))(
            P6_in)
        P7_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P7'.format(id))(
            P7_in)

    # upsample
    P7_U = layers.UpSampling2D()(P7_in)
    P6_td = layers.Add()([P7_U, P6_in])
    P6_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P6'.format(id))(P6_td)
    P6_U = layers.UpSampling2D()(P6_td)
    P5_td = layers.Add()([P6_U, P5_in])
    P5_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P5'.format(id))(P5_td)
    P5_U = layers.UpSampling2D()(P5_td)
    P4_td = layers.Add()([P5_U, P4_in])
    P4_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P4'.format(id))(P4_td)
    P4_U = layers.UpSampling2D()(P4_td)
    P3_out = layers.Add()([P4_U, P3_in])
    P3_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P3'.format(id))(P3_out)
    # downsample
    P3_D = layers.MaxPooling2D(strides=(2, 2))(P3_out)
    P4_out = layers.Add()([P3_D, P4_td, P4_in])
    P4_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P4'.format(id))(P4_out)
    P4_D = layers.MaxPooling2D(strides=(2, 2))(P4_out)
    P5_out = layers.Add()([P4_D, P5_td, P5_in])
    P5_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P5'.format(id))(P5_out)
    P5_D = layers.MaxPooling2D(strides=(2, 2))(P5_out)
    P6_out = layers.Add()([P5_D, P6_td, P6_in])
    P6_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P6'.format(id))(P6_out)
    P6_D = layers.MaxPooling2D(strides=(2, 2))(P6_out)
    P7_out = layers.Add()([P6_D, P7_in])
    P7_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P7'.format(id))(P7_out)

    return P3_out, P4_out, P5_out, P6_out, P7_out
