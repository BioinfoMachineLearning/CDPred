
# -*- coding: utf-8 -*-
"""
Created on Wed June 30 21:47:26 2021

@author: Zhiye
"""
import os, sys, glob, re, platform
import time
sys.path.insert(0, sys.path[0])
from generate_feature import *
from keras.models import model_from_json,load_model, Sequential, Model
from keras.utils import CustomObjectScope
from keras.engine.topology import Layer
from keras import initializers
from random import randint
import keras.backend as K
import tensorflow as tf
import numpy as np
import argparse

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


def getFileName(path, filetype):
    f_list = os.listdir(path)
    all_file = []
    for i in f_list:
        if os.path.splitext(i)[1] == filetype:
            all_file.append(i)
        elif filetype in i:
            all_file.append(i)
    return all_file

def get_model_info(model_path):
    os.chdir(model_path + '/')
    model_out = getFileName(model_path, '.json')[0]
    model_weight_out_best = getFileName(model_path, '.h5')
    reject_fea_file = getFileName(model_path, '.txt')[0] 
    HomoPred = []
    for model_weight in model_weight_out_best:
        with CustomObjectScope({'InstanceNormalization': InstanceNormalization, 'RowNormalization': RowNormalization, 'ColumNormalization': ColumNormalization, 'tf':tf}):
            json_string = open(model_out).read()
            temp_model = model_from_json(json_string)
            temp_model.load_weights(model_weight)
            HomoPred.append(temp_model)
    accept_list = []
    if not os.path.exists(reject_fea_file):
        print('%s not exists, plese check!'%reject_fea_file)
        sys.exit(1)
    else:
        with open(reject_fea_file) as f:
            for line in f:
                if line.startswith('#'):
                    feature_name = line.strip()
                    feature_name = feature_name[0:]
                    accept_list.append(feature_name)
    return HomoPred, accept_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model evalute for DeepComplex')
    parser.add_argument('-o', '--out_path', help="output folder", type=str, required=True)
    parser.add_argument('-m', '--model_path', help="model file in pytorch format, end with '.ckpt'", type=str, required=True)
    parser.add_argument('-f', '--feature_path', help="folder of feature of test dataset", type=str, required=True)
    parser.add_argument('-l', '--test_list', help="list file of test dataset", type=str, required=True)

    args = parser.parse_args()
    name = args.name
    out_path = args.out_path
    model_path = args.model_path
    feature_path = args.feature_path

    predict_method = 'realdist_hdist_nointra'
    HomoPred, accept_list = get_model_info(model_path)
    
    a3m_file = f'{feature_path}/{name}.a3m'
    pssm_file = f'{feature_path}/{name}_pssm.txt'
    rowatt_file = f'{feature_path}/{name}.npy'
    ccmpred_file = f'{feature_path}/{name}.mat'
    pred_dist_file = f'{feature_path}/{name}.dist'
  
    selected_list_2D = get2d_feature_by_list(name, accept_list, a3m_file, rowatt_file = rowatt_file, ccmpred_file = ccmpred_file, pssm_file = pssm_file, pred_dist_file_cb=pred_dist_file_cb)
    if type(selected_list_2D) == bool:
        print('Fareture shape error.', selected_list_2D.shape)
        sys.exit(1)
    selected_list_2D = selected_list_2D[np.newaxis,:,:,:]
    HomoPred_prediction = HomoPred.predict([selected_list_2D], batch_size= 1)

    if predict_method == 'realdist_hdist_nointra':
        Y_hat_rdist_inter = HomoPred_prediction[0][:,:,:,0:13].sum(axis=-1).squeeze()
        Y_hat_hdist_inter = HomoPred_prediction[1][:,:,:,0:13].sum(axis=-1).squeeze() #12 A 21
        # Y_hat_rdist_npy = HomoPred_prediction[0].squeeze()
        # Y_hat_hdist_npy = HomoPred_prediction[1].squeeze()
        pred_inter_hdist = f'{out_path}/{name}.htxt' 
        np.savetxt(pred_inter_hdist, Y_hat_hdist_inter, fmt='%.4f')