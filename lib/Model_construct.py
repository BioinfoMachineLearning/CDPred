# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:41:28 2022

@author: Zhiye
"""

from six.moves import range
import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Dense, Reshape, Activation, Flatten, Dropout, Lambda, add, concatenate, Concatenate, Add, average, multiply
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Permute
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.activations import tanh, softmax
from keras import metrics, initializers, utils, regularizers
import numpy as np

import tensorflow as tf
import sys
sys.setrecursionlimit(10000)

# Helper to build a conv -> BN -> relu block
def _bn_relu(input):
    norm = BatchNormalization(axis=-1)(input)
    return Activation("relu")(norm)

def _in_relu(input):
    norm = InstanceNormalization(axis=-1)(input)
    return Activation("relu")(norm)

def _in_elu(input):
    norm = InstanceNormalization(axis=-1)(input)
    return Activation("elu")(norm)

def _in_elu_conv2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None, dilation_rate=(1,1)):
    def f(input):
        act = _in_elu(input)
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer, dilation_rate = dilation_rate)(act)
        return conv
    return f

def _conv_bn_relu2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None, dilation_rate=(1,1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer, dilation_rate = dilation_rate)(input)
        # norm = BatchNormalization(axis=-1)(conv)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _conv_in_relu2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None, dilation_rate=(1,1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer, dilation_rate = dilation_rate)(input)
        norm = InstanceNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _in_sigmoid(input):
    norm = InstanceNormalization(axis=-1)(input)
    return Activation("sigmoid")(norm)

def _conv_in_sigmoid2D(filters, nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal", dilation_rate=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
        norm = InstanceNormalization(axis=-1)(conv)
        return Activation("sigmoid")(conv)
    
    return f

def _weighted_mean_squared_error(weight):

    def loss(y_true, y_pred):
        #set 20A as thresold
        # y_bool = Lambda(lambda x: x <= 20.0)(y_pred)
        y_bool = K.cast((y_true <= 10.0), dtype='float32') # 16.0
        y_bool_invert = K.cast((y_true > 10.0), dtype='float32')
        y_mean = K.mean(y_true)
        y_pred_below = y_pred * y_bool 
        y_pred_upper = y_pred * y_bool_invert 
        y_true_below = y_true * y_bool 
        y_true_upper = y_true * y_bool_invert 
        weights1 = weight
        # weights2 = 0# here need confirm whether use mean or constant
        weights2 = 1/(1 + K.square(y_pred_upper/y_mean))
        return K.mean(K.square((y_pred_below-y_true_below))*weights1) + K.mean(K.square((y_pred_upper-y_true_upper))*weights2)
        # return add([K.mean(K.square((y_pred_below-y_true_below))*weights1), K.mean(K.square((y_pred_upper-y_true_upper))*weights2)], axis= -1)
    return loss

def MaxoutAct(input, filters, kernel_size, output_dim, padding='same', activation = "relu"):
    output = None
    for _ in range(output_dim):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(input)
        activa = Activation(activation)(conv)
        maxout_out = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(activa)
        if output is not None:
            output = concatenate([output, maxout_out], axis=-1)
        else:
            output = maxout_out
    return output

class InstanceNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis '+str(self.axis)+' of input tensor should have a defined dimension but the layer received an input with shape '+str(input_shape)+ '.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', initializer=initializers.random_normal(1.0, 0.02))
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)

class RowNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super(RowNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis '+str(self.axis)+' of input tensor should have a defined dimension but the layer received an input with shape '+str(input_shape)+ '.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', initializer=initializers.random_normal(1.0, 0.02))
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[1], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)

class ColumNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super(ColumNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis '+str(self.axis)+' of input tensor should have a defined dimension but the layer received an input with shape '+str(input_shape)+ '.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', initializer=initializers.random_normal(1.0, 0.02))
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[2], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

def _shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = 1
    stride_height = 1
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    if not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal")(input)
    return add([shortcut, residual])


def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)
    x = multiply([init, se])
    return x

def spatial_attention(input_feature):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat) 
    assert cbam_feature._keras_shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])

# can't directly shuffle the tensor
def gather_index(x):
    return tf.gather(x, tf.random_shuffle(tf.range(tf.shape(x)[0])))

def channel_shuffle(x):
    x = Permute((3,1,2))(x)
    x = Lambda(gather_index)(x)
    x = Permute((2,3,1))(x)
    return x

def SA_layer(input_feature):
    x_0, x_1 = Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=3))(input_feature)
    # x_0, x_1 = Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 2})(input_feature)
    x_c = squeeze_excite_block(x_0)
    x_s = spatial_attention(x_1)
    out = concatenate([x_c, x_s], axis=-1)
    # out = channel_shuffle(out)
    return out

def _in_relu_K(x, bn_name=None, relu_name=None):
    # norm = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
    norm = InstanceNormalization(axis=-1, name=bn_name)(x)
    return Activation("relu", name=relu_name)(norm)

def _rcin_relu_K(x, bn_name=None, relu_name=None, activation='relu'):
    norm1 = InstanceNormalization(axis=-1, name=bn_name)(x)
    norm2 = RowNormalization(axis=-1, name=bn_name)(x)
    norm3 = ColumNormalization(axis=-1, name=bn_name)(x)
    norm  = concatenate([norm1, norm2, norm3])
    return Activation(activation, name=relu_name)(norm)

def _dilated_residual_block(block_function, filters, repetitions, is_first_layer=False, dilation_rate=(1,1), use_SE = False):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                # init_strides = (2, 2)
                init_strides = (1, 1)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0), use_SE = use_SE)(input)
        return input

    return f
    
def dilated_bottleneck_rc(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, use_SE = False):
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=regularizers.l2(1e-4))(input)
        else:
            # conv_1_1 = _rcin_relu_K(input) 
            conv_1_1 = _rcin_relu_K(input) 
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=init_strides,padding="same",kernel_initializer="he_normal")(conv_1_1)

        conv_3_3 = _rcin_relu_K(conv_1_1) 
        conv_3_3 = Conv2D(filters=filters, kernel_size=(3, 3), strides=init_strides, padding="same", kernel_initializer="he_normal")(conv_3_3)
        conv_7_1 = Conv2D(filters=filters, kernel_size=(7, 1), strides=init_strides, padding="same", kernel_initializer="he_normal")(conv_3_3)
        conv_1_7 = Conv2D(filters=filters, kernel_size=(1, 7), strides=init_strides, padding="same", kernel_initializer="he_normal")(conv_3_3)
        conv_3_3 = concatenate([conv_3_3, conv_7_1, conv_1_7])
        residual = _rcin_relu_K(conv_3_3) 
        residual = Conv2D(filters=filters, kernel_size=(1, 1), strides=init_strides, padding="same", kernel_initializer="he_normal")(residual)
        if use_SE == True:
            residual = squeeze_excite_block(residual)
        return _shortcut(input, residual)
    return f

def SA_bottleneck_rc(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, use_SE = False):
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=regularizers.l2(1e-4))(input)
        else:
            # conv_1_1 = _rcin_relu_K(input) 
            conv_1_1 = _rcin_relu_K(input, activation='elu') 
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=init_strides,padding="same",kernel_initializer="he_normal")(conv_1_1)

        conv_3_3 = _rcin_relu_K(conv_1_1, activation='elu') 
        conv_3_3 = Conv2D(filters=filters, kernel_size=(3, 3), strides=init_strides, padding="same", kernel_initializer="he_normal")(conv_3_3)
        conv_7_1 = Conv2D(filters=filters, kernel_size=(7, 1), strides=init_strides, padding="same", kernel_initializer="he_normal")(conv_3_3)
        conv_1_7 = Conv2D(filters=filters, kernel_size=(1, 7), strides=init_strides, padding="same", kernel_initializer="he_normal")(conv_3_3)
        conv_3_3 = concatenate([conv_3_3, conv_7_1, conv_1_7])
        residual = _rcin_relu_K(conv_3_3, activation='elu') 
        residual = Conv2D(filters=filters, kernel_size=(1, 1), strides=init_strides, padding="same", kernel_initializer="he_normal")(residual)
        if use_SE == True:
            residual = SA_layer(residual)
        return Activation("elu")(_shortcut(input, residual))
    return f


def HomoPredRes_with_paras_2D(kernel_size, feature_2D_num, filters, nb_layers, initializer = "he_normal", predict_method = "categorical_crossentropy"):
    _handle_dim_ordering()
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(None,None,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)
    
    HomoPred_2D_input = contact_input
    HomoPred_2D_conv = HomoPred_2D_input
    HomoPred_2D_conv = InstanceNormalization(axis=-1)(HomoPred_2D_conv)
    HomoPred_2D_conv = Conv2D(128, 1, padding = 'same')(HomoPred_2D_conv)
    HomoPred_2D_conv = MaxoutAct(HomoPred_2D_conv, filters=4, kernel_size=(1,1), output_dim=64, padding='same', activation = "elu")

    block = HomoPred_2D_conv
    dilated_num = [1, 2, 4, 8, 1] * 4
    repetitions = [nb_layers]
    for i, r in enumerate(repetitions):
        block = _dilated_residual_block(SA_bottleneck_rc, filters=filters, repetitions=r, is_first_layer=(i == 0), dilation_rate =dilated_num, use_SE = False)(block)
        block = Dropout(0.2)(block)
    # Last activation
    block = _rcin_relu_K(block)
    HomoPred_2D_conv = block
    # loss_weight = {'intradist':1.0, 'interdist':1.0, 'interhdist':1.0}
    if predict_method == 'realdist_hdist':
        HomoPred_2D_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),use_bias=True,
                         kernel_initializer=initializer, padding="same", kernel_regularizer=None)(HomoPred_2D_conv)
        HomoPred_2D_conv = InstanceNormalization(axis=-1)(HomoPred_2D_conv)
        intradist = Dense(42, activation='softmax', name='intradist')(HomoPred_2D_conv) 
        interdist = Dense(42, activation='softmax', name='interdist')(HomoPred_2D_conv) 
        interhdist = Dense(42, activation='softmax', name='interhdist')(HomoPred_2D_conv) 
        # loss = {'intradist':'categorical_crossentropy', 'interdist':'categorical_crossentropy', 'interhdist':'categorical_crossentropy'}
        HomoPred_RES = Model(inputs=contact_input, outputs=[intradist, interdist, interhdist])
    elif predict_method == 'realdist_hdist_nointra' or predict_method == 'realdist_hdist_whole':
        HomoPred_2D_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),use_bias=True,
                         kernel_initializer=initializer, padding="same", kernel_regularizer=None)(HomoPred_2D_conv)
        HomoPred_2D_conv = InstanceNormalization(axis=-1)(HomoPred_2D_conv)
        interdist = Dense(42, activation='softmax', name='interdist')(HomoPred_2D_conv) 
        interhdist = Dense(42, activation='softmax', name='interhdist')(HomoPred_2D_conv) 

        HomoPred_RES = Model(inputs=contact_input, outputs=[interdist, interhdist])
    else:
        HomoPred_2D_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),use_bias=True,
                         kernel_initializer=initializer, padding="same", kernel_regularizer=None)(HomoPred_2D_conv)
        HomoPred_2D_conv = InstanceNormalization(axis=-1)(HomoPred_2D_conv)
        intradist = Dense(42, activation='softmax', name='intradist')(HomoPred_2D_conv) 

        HomoPred_RES = Model(inputs=contact_input, outputs=intradist)
    # HomoPred_RES.compile(loss=loss, metrics=['accuracy'], optimizer=opt)
    # HomoPred_RES.summary(line_length=120)
    return HomoPred_RES
