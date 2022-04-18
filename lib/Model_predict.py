
# -*- coding: utf-8 -*-
"""
Created on Thru March 10 21:47:26 2022

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
import warnings
from util import *
from pdb_process import process_pdbfile, get_sequence_from_pdb

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
    CDPred = []
    for model_weight in model_weight_out_best:
        with CustomObjectScope({'InstanceNormalization': InstanceNormalization, 'RowNormalization': RowNormalization, 'ColumNormalization': ColumNormalization, 'tf':tf}):
            json_string = open(model_out).read()
            temp_model = model_from_json(json_string)
            temp_model.load_weights(model_weight)
            CDPred.append(temp_model)
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
    return CDPred, accept_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model evalute for DeepComplex')
    parser.add_argument('-n', '--name', help="protein id, format in A_B", type=str, required=True)
    parser.add_argument('-p', '--pdb_file_list', help="pdb file or pdb file list", nargs='+', type=str, required=True)
    parser.add_argument('-a', '--a3m_file', help="mltiple sequence alignment file end in .a3m", type=str, required=True)
    parser.add_argument('-m', '--model_option', help="model option, i.e: homodimer, heterodimer", type=str, required=False, default='homodimer')
    parser.add_argument('-o', '--out_path', help="output folder", type=str, required=True)

    os.environ['CUDA_VISIBLE_DEVICES']="-1"
    warnings.filterwarnings('ignore')
    args = parser.parse_args()
    name = args.name
    in_a3m_file = os.path.abspath(args.a3m_file)
    out_path = os.path.abspath(args.out_path)
    model_option = args.model_option

    pdb_file_list = []
    for pdb_file in args.pdb_file_list:
        pdb_file = os.path.abspath(pdb_file)
        pdb_file_list.append(pdb_file)

    chkdirs(out_path)
    GLOABL_Path = os.path.dirname(os.path.split(os.path.realpath(__file__))[0])
    print("### Find gloabl path :", GLOABL_Path)

    if model_option == 'homodimer':    
        model_path = f'{GLOABL_Path}/model/homo/' 
    elif model_option == 'heterodimer':
        model_path = f'{GLOABL_Path}/model/hetero/' 

    print("### Find model path :", model_path)
    CDPred, accept_list = get_model_info(model_path)
    
    #process pdb
    pdb_name_list = []
    length_list = []
    intra_dist_list = []
    sequence_list = []
    complex_length = 0
    for pdb_file in pdb_file_list:
        pdb_name = os.path.basename(pdb_file).split('.')[0]
        pdb_name_list.append(pdb_name)
        out_file = f'{out_path}/{pdb_name}.pdb'
        process_pdbfile(pdb_file, out_file)
        sequence = get_sequence_from_pdb(pdb_file)
        if len(sequence) > 1:
            print("Please make sure the pdb file is monomer!")
        sequence_list.append(sequence[0])
        length = len(sequence[0])
        length_list.append(length)
        complex_length += length
        #get distance map from pdb file
        intra_dist = get_cb_dist_from_pdbfile(out_file, length)
        intra_dist_list.append(intra_dist)

    print('### Generate features')
    feature_path = f'{out_path}/feature/{name}'
    chkdirs(feature_path)
    a3m_file = f'{feature_path}/{name}.a3m'
    aln_file = f'{feature_path}/{name}.aln'
    os.system(f'cp {in_a3m_file} {a3m_file}')
    os.system("grep -v '^>' %s | sed 's/[a-z]//g' >  %s" % (a3m_file, aln_file))

    fasta_file = f'{feature_path}/{name}.fasta'
    if model_option == 'homodimer':
        open(fasta_file, 'w').write(f'>{name}\n{sequence_list[0]}\n')
    elif model_option == 'heterodimer':
        open(fasta_file, 'w').write(f'>{name}\n{"".join(sequence_list)}\n')
    ccmpred_file = f'{feature_path}/{name}.mat'
    if not os.path.exists(ccmpred_file):
        compute_ccmpred(name, aln_file, save_ccmpred_path=feature_path)
    rowatt_file = f'{feature_path}/{name}.npy'
    if not os.path.exists(rowatt_file):
        computerowatt_over1024(name, a3m_file, outdir=feature_path, depth=128)
    pssm_file = f'{feature_path}/{name}_pssm.txt'
    if not os.path.exists(pssm_file):
        computepssm(name, fasta_file, feature_path, unirefdb)
    pred_dist_file = f'{feature_path}/{name}.dist'
    if model_option == 'homodimer':
        np.savetxt(pred_dist_file, intra_dist_list[0], fmt='%.3f')
    elif model_option == 'heterodimer':
        intra_dist = np.zeros((complex_length, complex_length))
        lenA = length_list[0]
        intra_dist[:lenA, :lenA] = intra_dist_list[0]
        intra_dist[lenA:, lenA:] = intra_dist_list[1]
        np.savetxt(pred_dist_file, intra_dist, fmt='%.3f')
  
    selected_list_2D = get2d_feature_by_list(name, accept_list, a3m_file, rowatt_file = rowatt_file, ccmpred_file = ccmpred_file, pssm_file = pssm_file, pred_dist_file_cb=pred_dist_file)
    if type(selected_list_2D) == bool:
        print('Fareture shape error.', selected_list_2D.shape)
        sys.exit(1)
    selected_list_2D = selected_list_2D[np.newaxis,:,:,:]

    Y_hat_hdist_npy = 0
    for temp in CDPred:
        CDPred_prediction = temp.predict([selected_list_2D], batch_size= 1)
        Y_hat_hdist_npy += CDPred_prediction[1].squeeze()
    Y_hat_hdist_npy /= len(CDPred)
    hv_con = Y_hat_hdist_npy[:,:,0:13].sum(axis=-1).squeeze()
    hv_real_dist = npy2distmap(Y_hat_hdist_npy)
    if model_option == 'homodimer':
        hcon_inter = hv_con
        hdist_inter = hv_real_dist
    elif model_option == 'heterodimer':
        value = lenA
        hcon_inter = hv_con[:value,value:]
        hdist_inter = hv_real_dist[:value,value:]

    # save the output file
    predmap_dir = f'{out_path}/predmap/'
    print('### save prediction results %s'%predmap_dir)
    chkdirs(predmap_dir)
    hdist_inter_file = f'{predmap_dir}/{name}.dist'
    np.savetxt(hdist_inter_file, hdist_inter, fmt='%.4f')
    hcon_inter_file = f'{predmap_dir}/{name}.htxt'
    np.savetxt(hcon_inter_file, hcon_inter, fmt='%.4f')
    hdist_rr_file = f'{predmap_dir}/{name}_dist.rr'
    gen_rr_file(hdist_inter, hdist_rr_file,  option='distance')
    hcon_rr_file = f'{predmap_dir}/{name}_con.rr'
    gen_rr_file(hcon_inter, hcon_rr_file,  option='contact')