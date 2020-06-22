from bifpn import build_BiFPN
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.activations import mish
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout

def UDet(input_shape=(512,512,1)):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, (3, 3), activation=tfa.activations.mish, padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation=tfa.activations.mish, padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation=tfa.activations.mish, padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation=tfa.activations.mish, padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation=tfa.activations.mish, padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation=tfa.activations.mish, padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation=tfa.activations.mish, padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation=tfa.activations.mish, padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation=tfa.activations.mish, padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation=tfa.activations.mish, padding='same')(conv5)
    
    features = [conv1, conv2, conv3,conv4,conv5]
    channels = [64,128,256,512,1024]
    outs = build_BiFPN(features,channels[0],1)
    
    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), outs[3]], axis=3)
    conv6 = Conv2D(512, (3, 3), activation=tfa.activations.mish, padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation=tfa.activations.mish, padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), outs[2]], axis=3)
    conv7 = Conv2D(256, (3, 3), activation=tfa.activations.mish, padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation=tfa.activations.mish, padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), outs[1]], axis=3)
    conv8 = Conv2D(128, (3, 3), activation=tfa.activations.mish, padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation=tfa.activations.mish, padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), outs[0]], axis=3)
    conv9 = Conv2D(64, (3, 3), activation=tfa.activations.mish, padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation=tfa.activations.mish, padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    model = Model(inputs=[inputs], outputs=[conv10])

    return model
