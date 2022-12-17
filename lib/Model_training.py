# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:47:26 2019

@author: Zhiye
"""
from Model_construct import *
from generate_feature import *

import numpy as np
import os, sys, shutil, platform, time, pickle
from collections import defaultdict
from six.moves import range

import keras.backend as K
import tensorflow as tf
from keras.models import model_from_json,load_model, Sequential, Model
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adagrad, Adamax, Nadam
from keras.utils import multi_gpu_model, Sequence, CustomObjectScope
from keras.callbacks import ReduceLROnPlateau
from random import randint
from sklearn import metrics

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

def build_dataset_dictionaries(data_file, Maximum_length=500, shuffle=False):
    shuffle = shuffle
    line_list = open(data_file).readlines()
    if shuffle==True:
        random.shuffle(line_list)
    data_dict = {}
    for line in line_list:
        line = line.strip('\n').split(' ')
        if len(line) < 3:
            tar_name = line[0]
            tar_length = line[1]
            if int(tar_length) > Maximum_length or int(tar_length) < 30:
                continue
            data_dict[tar_name] = tar_length
        else:
            tar_name = line[0]
            lenA = int(line[1])
            lenB = int(line[2])
            if lenA == lenB: #homo
                tar_length = line[1]            
                if int(tar_length) > Maximum_length or int(tar_length) < 30:
                    continue
                data_dict[tar_name] = tar_length
            else:
                tar_length = lenA + lenB
                if (tar_length) > Maximum_length or int(tar_length) < 30:
                    continue
                data_dict[tar_name] = f'{lenA}_{lenB}'
    return data_dict
   
#covert real distance map into multi-class distance
#G: 0-2,   2-22:0.5,   22-i  42
#C: 0-4,   4-20:2,     20-i  10
#D: 0-3.5, 3.5-19:0.5, 20-i  33
#T: 0-2,   2-20:0.5,   20-i  37
def real_value2mul_class(input_mat, if_onehot=False, option='G'):
    length = input_mat.shape[0]
    output_mat = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            if option == 'G':
                output_mat[i,j] = math.ceil(input_mat[i,j]/0.5 - 4)
                if output_mat[i,j] < -3:
                    output_mat[i, j] = 41 #gap in the map, when training, gap should treat as infinte number
                elif output_mat[i,j] < 0:
                    output_mat[i, j] = 0
                elif output_mat[i,j] > 41:
                    output_mat[i, j] = 41
            elif option == 'C':
                output_mat[i,j] = math.ceil(input_mat[i,j]/2 - 2)
                if output_mat[i,j] < -1:
                    output_mat[i, j] = 9
                elif output_mat[i,j] < 0:
                    output_mat[i, j] = 0
                elif output_mat[i,j] > 9:
                    output_mat[i, j] = 9
            elif option == 'D':
                output_mat[i,j] = math.ceil(input_mat[i,j]/0.5 - 7)
                if output_mat[i,j] < -6:
                    output_mat[i, j] = 32
                elif output_mat[i,j] < 0:
                    output_mat[i, j] = 0
                elif output_mat[i,j] > 32:
                    output_mat[i, j] = 32
            elif option == 'T':
                output_mat[i,j] = math.ceil(input_mat[i,j]/0.5 - 4)
                if output_mat[i,j] < -3:
                    output_mat[i, j] = 36
                elif output_mat[i,j] < 0:
                    output_mat[i, j] = 0
                elif output_mat[i,j] > 36:
                    output_mat[i, j] = 36
    if if_onehot:
        if option == 'G':
            output_mat= (np.arange(42) == output_mat[...,None]).astype(int)
        elif option == 'C':
            output_mat= (np.arange(10) == output_mat[...,None]).astype(int)
        elif option == 'D':
            output_mat= (np.arange(33) == output_mat[...,None]).astype(int)
        elif option == 'T':
            output_mat= (np.arange(37) == output_mat[...,None]).astype(int)
    else:
        output_mat = output_mat.astype(int)
    return output_mat

def real_value2mul_class_orientation(input_mat, if_onehot=False, option='dih'):
    length = input_mat.shape[0]
    output_mat = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            if option == 'dih':
                output_mat[i,j] = int((input_mat[i,j]+180)/15)
            else:
                output_mat[i,j] = int((input_mat[i,j])/15)
            if output_mat[i,j] > 23:
                output_mat[i, j] = 23
    if if_onehot:
        if option == 'dih':
            output_mat= (np.arange(24) == output_mat[...,None]).astype(int)
        else:
            output_mat= (np.arange(12) == output_mat[...,None]).astype(int)
    return output_mat

# short[6-11], medium[12-23], long[24,~]  less 16A
def get_upright(input_map, tar_range = 'long'):
    L = int(input_map.shape[0])
    vector = []
    if tar_range == 'long':
        for i in range(0, L):
            for j in range(i+24, L):
                vector.append(input_map[i,j])
    elif tar_range == 'medium':
        for i in range(0, L):
            for j in range(i+12, i+24):
                if j >= L:
                    continue
                vector.append(input_map[i,j])
    elif tar_range == 'short':
        for i in range(0, L):
            for j in range(i+6, i+12):
                if j >= L:
                    continue
                vector.append(input_map[i,j])
    elif type(tar_range) == int:
        for i in range(0, L):
            for j in range(i+tar_range, L):
                if j >= L:
                    continue
                vector.append(input_map[i,j])
    return np.array(vector)

def ceil_topxL_to_one(Y_hat, x):
    Y_ceiled = np.copy(Y_hat)
    xL = int(x)
    Y_ceiled[:] = np.zeros(len(Y_hat[:]))
    Y_ceiled[np.argpartition(Y_hat[:], -xL)[-xL:]] = 1
    return Y_ceiled.astype(int)

def CheckFileAvaliablity(accept_list, name, feature_path):
    pssm_file = None
    a3m_file = None
    plm_file = None   
    pred_dist_file = None
    if '# pssm' in accept_list:  
        pssm_file = feature_path + '/pssm/' + name + '_pssm.txt'
        if not os.path.exists(pssm_file):
            print("Row attention file not exists: ",pssm_file, " pass!")
            # continue 
    else:
        pssm_file = None   
    if '# rowatt' in accept_list:  
        rowatt_file = feature_path + '/rowatt/' + name + '.npy'
        if not os.path.exists(rowatt_file):
            rowatt_file = None
            a3m_file = feature_path + '/a3m/' + name + '.a3m'
            if not os.path.exists(a3m_file):
                print("a3m file not exists: ",a3m_file, " pass!")
                # continue 
    else:
        rowatt_file = None   
    if '# plm' in accept_list:  
        plm_file = feature_path + '/plm/' + name + '.plm' #if not exists plm file, can generate from aln
        if not os.path.exists(plm_file):
            plm_file = None   
    else:
        plm_file = None   
    if '# ccmpred' in accept_list:  
        ccmpred_file = feature_path + '/plm/' + name + '.mat'
        if not os.path.exists(ccmpred_file):
            ccmpred_file = None   
    else:
        ccmpred_file = None   
    if '# intradist' in accept_list:  
        pred_dist_file = feature_path + '/pred_dist/' + name + '.dist'
        if not os.path.exists(pred_dist_file):
            pred_dist_file = None   
    else:
        pred_dist_file = None   
    return pssm_file, rowatt_file, plm_file, ccmpred_file, pred_dist_file

def generate_data_from_aln(list_file, path_of_X, path_of_Y, batch_size, reject_fea_file='None', feature_2D_num = 441, 
    if_use_binsize=False, predict_method='bin_class', Maximum_length = 500, droprate=1):
    accept_list = []
    if reject_fea_file != 'None':
        with open(reject_fea_file) as f:
            for line in f:
                if line.startswith('#'):
                    feature_name = line.strip()
                    feature_name = feature_name[0:]
                    accept_list.append(feature_name)

    training_dict = build_dataset_dictionaries(list_file, Maximum_length, shuffle = True)
    training_list = list(training_dict.keys())
    training_lens = list(training_dict.values())
    all_data_num = len(training_dict)
    loopcount = all_data_num // int(batch_size)
    index = 0
    while(True):
        if index >= loopcount:
            training_dict = build_dataset_dictionaries(list_file, Maximum_length, shuffle = True)   
            training_list = list(training_dict.keys())
            training_lens = list(training_dict.values())
            index = 0
        batch_list = training_list[index * batch_size:(index + 1) * batch_size]
        batch_list_len = training_lens[index * batch_size:(index + 1) * batch_size]
        index += 1
        # if if_use_binsize:
        #     max_pdb_lens = int(Maximum_length)
        # else:
        #     max_pdb_lens = int(max(batch_list_len))
        batch_X  = []
        batch_X1  = []
        batch_X2  = []
        batch_Y  = []
        batch_Y1 = []
        batch_Y2 = []
        batch_Y3 = []
        for i in range(0, len(batch_list)):
            pdb_name = batch_list[i]
            if '_' in batch_list_len[i]:
                pdb_len = int(batch_list_len[i].split('_')[0]) + int(batch_list_len[i].split('_')[1])
            else:
                pdb_len = int(batch_list_len[i])
            aln = path_of_X + '/alignment/' + pdb_name + '.aln'
            # if not os.path.exists(aln):
            #     print("Alignment file not exists: ",aln, " pass!")
            #     continue   
            pssm_file = None
            a3m_file = None
            plm_file = None   
            pred_dist_file = None
            if '# pssm' in accept_list:  
                pssm_file = path_of_X + '/pssm/' + pdb_name + '_pssm.txt'
                if not os.path.exists(pssm_file):
                    print("Row attention file not exists: ",pssm_file, " pass!")
                    continue 
            else:
                pssm_file = None   
            if '# rowatt' in accept_list:  
                rowatt_file = path_of_X + '/rowatt/' + pdb_name + '.npy'
                if not os.path.exists(rowatt_file):
                    rowatt_file = None
                    a3m_file = path_of_X + '/a3m/' + pdb_name + '.a3m'
                    if not os.path.exists(a3m_file):
                        print("a3m file not exists: ",a3m_file, " pass!")
                        continue 
            else:
                rowatt_file = None  
            if '# rowatt_diff' in accept_list:  
                rowatt_diff_file = path_of_X + '/rowatt_diff/' + pdb_name + '.npy'
                if not os.path.exists(rowatt_diff_file):
                    print("rowatt_diff_file not exists: ",rowatt_diff_file, " pass!")
                    continue 
            else:
                rowatt_diff_file = None    
            if '# rowatt_inter' in accept_list:  
                rowatt_inter_file = path_of_X + '/rowatt_inter/' + pdb_name + '.npy'
                if not os.path.exists(rowatt_inter_file):
                    print("rowatt_inter_file not exists: ",rowatt_inter_file, " pass!")
                    continue 
            else:
                rowatt_inter_file = None    
            if '# plm' in accept_list:  
                plm_file = path_of_X + '/plm/' + pdb_name + '.plm' #if not exists plm file, can generate from aln
                if not os.path.exists(plm_file):
                    plm_file = None   
            else:
                plm_file = None   
            if '# ccmpred' in accept_list:  
                ccmpred_file = path_of_X + '/plm/' + pdb_name + '.mat'
                if not os.path.exists(ccmpred_file):
                    ccmpred_file = None   
            else:
                ccmpred_file = None   
            if '# intradist_cb' in accept_list:  
                pred_dist_file_cb = path_of_X + '/pred_dist/' + pdb_name + '.txt'
                if not os.path.exists(pred_dist_file_cb):
                    pred_dist_file_cb = None   
            else:
                pred_dist_file_cb = None   
            if '# intradist_hv' in accept_list:  
                pred_dist_file_hv = path_of_X + '/pred_dist/' + pdb_name + '.htxt'
                if not os.path.exists(pred_dist_file_hv):
                    pred_dist_file_hv = None   
            else:
                pred_dist_file_hv = None   
            if '# interdist' in accept_list:  
                inter_dist_file = path_of_X + '/inter_dist/' + pdb_name + '.txt'
                if not os.path.exists(inter_dist_file):
                    inter_dist_file = None   
            else:
                inter_dist_file = None   
            
            if 'realdist_hdist' in predict_method:
                targetrealdist = path_of_Y[0] + pdb_name + '.txt'
                if not os.path.exists(targetrealdist):
                        print("target file not exists: ",targetrealdist, " pass!")
                        continue  
                targethdist = path_of_Y[1] + pdb_name + '.htxt'
                if not os.path.exists(targethdist):
                        print("target file not exists: ",targethdist, " pass!")
                        continue  
            else:
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.exists(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            # print(pred_dist_file_cb)
            featuredata = get2d_feature_by_list(pdb_name, accept_list, aln, rate = droprate, rowatt_file = rowatt_file, rowatt_inter_file=rowatt_inter_file, 
                plm_file = plm_file, ccmpred_file = ccmpred_file, pssm_file = pssm_file, a3m_file=a3m_file, pred_dist_file_cb = pred_dist_file_cb, pred_dist_file_hv = pred_dist_file_hv, inter_dist_file = inter_dist_file)
            # print(pdb_len, featuredata.shape) 
            if type(featuredata) == bool:
                print("Bad alignment, Please check!\n")
                continue
            if featuredata.shape[-1] != feature_2D_num:
                print("Target %s has wrong feature shape! Should be %s but now %s, Continue!" %(pdb_name, feature_2D_num, featuredata.shape[-1]))
                continue
            # if pdb_len != max_pdb_lens:
            #     len_diff = max_pdb_lens-pdb_len
            #     X = np.pad(bb, ((0,len_diff),(0,len_diff),(0,0)), 'constant', constant_values=0)
            # else:
            X = featuredata
            if predict_method == 'realdist_hdist':
                Y_realdist = np.loadtxt(targetrealdist)
                Y_hdist = np.loadtxt(targethdist)
                Y_realdist_intra = Y_realdist[:pdb_len,:pdb_len]
                Y_realdist_inter = Y_realdist[:pdb_len,pdb_len:]
                Y_hdist_inter = Y_hdist[:pdb_len,pdb_len:]
                Y1 = real_value2mul_class(Y_realdist_intra, if_onehot=True, option='G')
                Y2 = real_value2mul_class(Y_realdist_inter, if_onehot=True, option='G')
                Y3 = real_value2mul_class(Y_hdist_inter, if_onehot=True, option='G')
            elif predict_method == 'realdist_hdist_nointra':
                Y_realdist = np.loadtxt(targetrealdist)
                Y_hdist = np.loadtxt(targethdist)
                Y_realdist_inter = Y_realdist[:pdb_len,pdb_len:]
                Y_hdist_inter = Y_hdist[:pdb_len,pdb_len:]
                Y1 = real_value2mul_class(Y_realdist_inter, if_onehot=True, option='G')
                Y2 = real_value2mul_class(Y_hdist_inter, if_onehot=True, option='G')
            elif predict_method == 'realdist_nointra':
                Y_realdist = np.loadtxt(targetfile)
                Y_realdist_inter = Y_realdist[:pdb_len,pdb_len:]
                Y1 = real_value2mul_class(Y_realdist_inter, if_onehot=True, option='G')
            elif predict_method == 'realdist_hdist_whole':
                Y_realdist = np.loadtxt(targetrealdist)
                Y_hdist = np.loadtxt(targethdist)
                Y1 = real_value2mul_class(Y_realdist, if_onehot=True, option='G')
                Y2 = real_value2mul_class(Y_hdist, if_onehot=True, option='G')
            else:
                Y_realdist = np.loadtxt(targetfile)
                Y = real_value2mul_class(Y_realdist, if_onehot=True, option='G')

            # if featuredata.shape[0] != Y_realdist.shape[0]:
            #     print(f"Target {pdb_name} has wrong dist map shape! distmap {Y_realdist.shape} featuredata {featuredata.shape}")
            #     continue

            if  predict_method == 'realdist_hdist':
                batch_X.append(X)
                batch_Y1.append(Y1)
                batch_Y2.append(Y2)
                batch_Y3.append(Y3)
                del X, Y1, Y2, Y3
            elif  predict_method == 'realdist_hdist_nointra' or predict_method == 'realdist_hdist_whole':
                batch_X.append(X)
                batch_Y1.append(Y1)
                batch_Y2.append(Y2)
                del X, Y1, Y2
            elif  predict_method == 'realdist_nointra':
                batch_X.append(X)
                batch_Y1.append(Y1)
                del X, Y1
            else:
                batch_X.append(X)
                batch_Y.append(Y)
                del X, Y
        # print(predict_method)
        if predict_method == 'realdist_hdist':
            batch_X =  np.array(batch_X)
            batch_Y1 =  np.array(batch_Y1)
            batch_Y2 =  np.array(batch_Y2)
            batch_Y3 =  np.array(batch_Y3)
            if len(batch_X.shape) < 4 or len(batch_Y1.shape) < 4 or len(batch_Y2.shape) < 4 or len(batch_Y3.shape) < 4:
                print('Data shape error, pass!\n')
                continue
            yield batch_X, [batch_Y1, batch_Y2, batch_Y3]
        elif predict_method == 'realdist_hdist_nointra' or predict_method == 'realdist_hdist_whole':
            batch_X =  np.array(batch_X)
            batch_Y1 =  np.array(batch_Y1)
            batch_Y2 =  np.array(batch_Y2)
            if len(batch_X.shape) < 4 or len(batch_Y1.shape) < 4 or len(batch_Y2.shape) < 4:
                print('Data shape error, pass!\n')
                continue
            yield batch_X, [batch_Y1, batch_Y2]
        elif predict_method == 'realdist_nointra':
            batch_X =  np.array(batch_X)
            batch_Y1 =  np.array(batch_Y1)
            if len(batch_X.shape) < 4 or len(batch_Y1.shape) < 4:
                print('Data shape error, pass!\n')
                continue 
            yield batch_X, batch_Y1
        else:
            batch_X =  np.array(batch_X)
            batch_Y =  np.array(batch_Y)
            if len(batch_X.shape) < 4 or len(batch_Y.shape) < 4:
                print('Data shape error, pass!\n')
                continue
            yield batch_X, batch_Y

def HomoPred_train(feature_num, CV_dir, model_prefix,
    epoch_outside,epoch_inside,epoch_rerun,win_array,nb_filters, nb_layers, batch_size_train, path_of_lists, path_of_Y, path_of_X, Maximum_length, reject_fea_file='None',
    initializer = "he_normal", predict_method = "realdist_hdist", if_use_binsize = False, rate = 1.0): 

    accept_list = []
    if reject_fea_file != 'None':
        with open(reject_fea_file) as f:
            for line in f:
                if line.startswith('#'):
                    feature_name = line.strip()
                    feature_name = feature_name[0:]
                    accept_list.append(feature_name)
    print("Load feature: ", accept_list)                
    feature_2D_num=feature_num # the number of features for each residue
 
    print("Load feature number: ", feature_2D_num)
    ### Define the model 
    model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
    model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
    model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)

    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
    if model_prefix == 'HomoPred_Net':
        HomoPred = HomoPredRes_with_paras_2D(win_array,feature_2D_num, nb_filters,nb_layers, initializer, predict_method)
    else:
        pass
        print('Please input valid model prefix')
        sys.exit(1)

    if os.path.exists(model_out) == False:
        model_json = HomoPred.to_json()
        print("Saved model to disk")
        with open(model_out, "w") as json_file:
            json_file.write(model_json)
    else:
        with CustomObjectScope({'InstanceNormalization': InstanceNormalization, 'RowNormalization': RowNormalization, 'ColumNormalization': ColumNormalization, 'tf.split':tf.split, 'tf':tf}):
            json_string = open(model_out).read()
            HomoPred = model_from_json(json_string)

    rerun_flag=0
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    if os.path.exists(model_weight_out):
        print("######## Loading existing weights ",model_weight_out)
        HomoPred.load_weights(model_weight_out)
        rerun_flag = 1
    else:
        print("######## Setting initial weights")   
        chkdirs(val_acc_history_out)     
        with open(val_acc_history_out, "a") as myfile:
          myfile.write("Epoch\tprec_l5\tprec_l2\tprec_1l\tmcc_l5\tmcc_l2\tmcc_1l\trecall_l5\trecall_l2\trecall_1l\tf1_l5\tf1_l2\tf1_1l\n")
        

    #predict_method has three value : bin_class, mul_class, real_dist
    if predict_method == 'realdist_hdist':
        path_of_Y_train = [path_of_Y + '/real_dist/', path_of_Y + '/h_dist/']
        path_of_Y_evalu = [path_of_Y + '/real_dist/', path_of_Y + '/h_dist/']
        loss_function = {'intradist':'categorical_crossentropy', 'interdist':'categorical_crossentropy', 'interhdist':'categorical_crossentropy'}
        loss_weight = {'intradist':1.0, 'interdist':1.0, 'interhdist':1.0}
    elif predict_method == 'realdist_hdist_nointra' or predict_method == 'realdist_hdist_whole':
        path_of_Y_train = [path_of_Y + '/real_dist/', path_of_Y + '/h_dist/']
        path_of_Y_evalu = [path_of_Y + '/real_dist/', path_of_Y + '/h_dist/']
        loss_function = {'interdist':'categorical_crossentropy', 'interhdist':'categorical_crossentropy'}
        loss_weight = {'interdist':1.0, 'interhdist':1.0}     
    elif predict_method == 'intradist':
        path_of_Y_train = path_of_Y + '/real_dist/'
        path_of_Y_evalu = path_of_Y + '/real_dist/'
        loss_function = {'intradist':'categorical_crossentropy'}
    
    if 'realdist_hdist' in predict_method:
        HomoPred.compile(loss=loss_function, loss_weights = loss_weight, metrics=['acc'], optimizer=opt)
    else:
        HomoPred.compile(loss=loss_function, metrics=['acc'], optimizer=opt)

    model_weight_epochs = "%s/model_weights/"%(CV_dir)
    model_weights_top = "%s/model_weights_top/"%(CV_dir)
    model_val_acc= "%s/val_acc_inepoch/"%(CV_dir)
    chkdirs(model_weight_epochs)
    chkdirs(model_weights_top)
    chkdirs(model_val_acc)
 
    train_dict = build_dataset_dictionaries(path_of_lists+'/train.lst', Maximum_length, shuffle = False)   
    vali_dict = build_dataset_dictionaries(path_of_lists+'/vali.lst', Maximum_length, shuffle = False)   

    print('Total Number of Training dataset = ',str(len(train_dict)))
    print('Total Number of Validation dataset = ',str(len(vali_dict)))

    # callbacks=[reduce_lr]
    avg_prec_best = 0
    lr_decay = False
    train_loss_list = []
    intra_rdist_loss = []
    inter_rdist_loss = []
    inter_hdist_loss = []

    os.system('cp %s %s'%(reject_fea_file, CV_dir))
    train_file = path_of_lists+'/train.lst'
    for epoch in range(epoch_rerun,epoch_outside):
        if (epoch >=30 and lr_decay == False): 
            print("Setting lr_decay as true")
            lr_decay = True
        if (epoch >=70 and lr_decay == False): 
            print("Setting lr_decay as true")
            lr_decay = True
            opt = SGD(lr=0.001, momentum=0.9, decay=0.00, nesterov=True)#0.001  
            HomoPred.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)
        if (epoch >=80 and lr_decay == False): 
            print("Setting lr_decay as true")
            lr_decay = True
            opt = SGD(lr=0.00001, momentum=0.9, decay=0.00, nesterov=True)#0.001  
            HomoPred.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)

        print("\n############ Running epoch ", epoch)
        if epoch == 0 and rerun_flag == 0: 
            first_inepoch = 1
            history = HomoPred.fit_generator(generate_data_from_aln(path_of_lists+'/train.lst', path_of_X, path_of_Y_train, batch_size_train, 
                reject_fea_file, feature_2D_num = feature_2D_num, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length, 
                droprate = rate), steps_per_epoch = len(train_dict)//batch_size_train, epochs = first_inepoch, max_queue_size=20, workers=1, use_multiprocessing=False)    
            if predict_method == 'realdist_hdist_nointra':
                inter_rdist_loss.append(history.history['interdist_loss'][first_inepoch-1])  
                inter_hdist_loss.append(history.history['interhdist_loss'][first_inepoch-1])  
            elif predict_method == 'realdist_hdist':
                intra_rdist_loss.append(history.history['intradist_loss'][first_inepoch-1])  
                inter_rdist_loss.append(history.history['interdist_loss'][first_inepoch-1])  
                inter_hdist_loss.append(history.history['interhdist_loss'][first_inepoch-1])  
            else:
                train_loss_list.append(history.history['loss'][first_inepoch-1]) 
        else: 
            history = HomoPred.fit_generator(generate_data_from_aln(path_of_lists+'/train.lst', path_of_X, path_of_Y_train, batch_size_train, reject_fea_file, 
            feature_2D_num = feature_2D_num, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length, droprate = rate), 
            steps_per_epoch = len(train_dict)//batch_size_train, epochs = 1, max_queue_size=20, workers=1, use_multiprocessing=False)        
            if predict_method == 'realdist_hdist_nointra':
                inter_rdist_loss.append(history.history['interdist_loss'][0])  
                inter_hdist_loss.append(history.history['interhdist_loss'][0])   
            elif predict_method == 'realdist_hdist':
                intra_rdist_loss.append(history.history['intradist_loss'][0])  
                inter_rdist_loss.append(history.history['interdist_loss'][0])  
                inter_hdist_loss.append(history.history['interhdist_loss'][0])   
            else:
                train_loss_list.append(history.history['loss'][0])
        HomoPred.save_weights(model_weight_out)

        ### save models
        model_weight_out_inepoch = "%s/model-train-weight-%s-epoch%i.h5" % (model_weight_epochs,model_prefix,epoch)
        HomoPred.save_weights(model_weight_out_inepoch)
        ##### running validation

        print("Now evaluate for epoch ",epoch)
        val_acc_out_inepoch = "%s/validation_epoch%i.acc_history" % (model_val_acc, epoch)    

        step_num = 0 
        out_topL_prec = 0.0
        out_topL2_prec = 0.0
        out_topL5_prec = 0.0 
        out_top5_prec_com = 0.0
        out_top10_prec_com = 0.0
        out_topL10_prec_com = 0.0
        out_topL5_prec_com = 0.0 
        out_top5_prec_com_cb = 0.0
        out_top10_prec_com_cb = 0.0
        out_topL10_prec_com_cb= 0.0
        out_topL5_prec_com_cb= 0.0 
        print(("SeqName\tSeqLen\tprec_l\tprec_l2\tprec_l5\tprec_top5\tprec_top10\tprec_l10\n"))    

        for key in vali_dict: 
            value = vali_dict[key]
            if '_' in value:
                lenA = int(value.split('_')[0]) 
                lenB = int(value.split('_')[1]) 
                value = lenA #name lenA_lenB
                short_len = min(lenA, lenB)
            else:
                value = int(value)
                short_len = value
            if value < 30: 
                continue
            aln = path_of_X + '/alignment/' + key + '.aln'
            # if not os.path.isfile(aln):
            #     print("Alignment file not exists: ",aln, " pass!")
            #     continue      
            pssm_file = None
            a3m_file = None
            plm_file = None   
            ccmpred_file = None  
            pred_dist_file = None
            pred_dist_file_cb = None
            pred_dist_file_cb = None
            if '# pssm' in accept_list:  
                pssm_file = path_of_X + '/pssm/' + key + '_pssm.txt'
                if not os.path.exists(pssm_file):
                    print("Row attention file not exists: ",pssm_file, " pass!")
                    continue 
            else:
                pssm_file = None   
            if '# rowatt' in accept_list:  
                rowatt_file = path_of_X + '/rowatt/' + key + '.npy'
                if not os.path.exists(rowatt_file):
                    rowatt_file = None
                    a3m_file = path_of_X + '/a3m/' + key + '.a3m'
                    if not os.path.exists(a3m_file): 
                        print("a3m file not exists: ",a3m_file, " pass!")
                        continue 
            else:
                rowatt_file = None 
            if '# rowatt_diff' in accept_list:  
                rowatt_diff_file = path_of_X + '/rowatt_diff/' + key + '.npy'
                if not os.path.exists(rowatt_diff_file):
                    print("rowatt_diff_file not exists: ",rowatt_diff_file, " pass!")
                    continue 
            else:
                rowatt_diff_file = None    
            if '# rowatt_inter' in accept_list:  
                rowatt_inter_file = path_of_X + '/rowatt_inter/' + key + '.npy'
                if not os.path.exists(rowatt_inter_file):
                    print("rowatt_inter_file not exists: ",rowatt_inter_file, " pass!")
                    continue 
            else:
                rowatt_inter_file = None    
            if '# plm' in accept_list:  
                plm_file = path_of_X + '/plm/' + key + '.plm' #if not exists plm file, can generate from aln
                if not os.path.exists(plm_file):
                    plm_file = None   
            else:
                plm_file = None    
            if '# ccmpred' in accept_list:  
                ccmpred_file = path_of_X + '/plm/' + key + '.mat'
                if not os.path.exists(ccmpred_file):
                    ccmpred_file = None   
            else:
                ccmpred_file = None  
            if '# intradist_cb' in accept_list:  
                pred_dist_file_cb = path_of_X + '/pred_dist/' + key + '.txt'
                if not os.path.exists(pred_dist_file_cb):
                    pred_dist_file_cb = None   
            else:
                pred_dist_file = None   
            if '# intradist_hv' in accept_list:  
                pred_dist_file_hv = path_of_X + '/pred_dist/' + key + '.htxt'
                if not os.path.exists(pred_dist_file_hv):
                    pred_dist_file_hv = None   
            else:
                pred_dist_file_hv = None   
            if '# interdist' in accept_list:  
                inter_dist_file = path_of_X + '/inter_dist/' + key + '.txt'
                if not os.path.exists(inter_dist_file):
                    inter_dist_file = None   
            else:
                inter_dist_file = None   
 
            # print(key) 
            selected_list_2D = get2d_feature_by_list(key, accept_list, aln, rowatt_file = rowatt_file, rowatt_inter_file = rowatt_inter_file, plm_file = plm_file, 
                ccmpred_file = ccmpred_file, pssm_file = pssm_file, a3m_file=a3m_file, pred_dist_file_cb=pred_dist_file_cb)
            if type(selected_list_2D) == bool:
                continue
            selected_list_2D = selected_list_2D[np.newaxis,:,:,:] 

            if predict_method == 'intradist':
                targetrdist = path_of_Y_evalu + key + '.txt'
                if not os.path.isfile(targetrdist):
                        print("target file not exists: ",targetrdist, " pass!")
                        continue   
                Y_rdist = np.loadtxt(targetrdist)
                true_complex_r = np.copy(Y_rdist).squeeze()
                true_complex_r = get_upright(true_complex_r, tar_range = 'long')
                true_complex_r[true_complex_r<8] = 1  # 6
                true_complex_r[true_complex_r>=8] = 0
            else:
                targethdist = path_of_Y_evalu[1] + key + '.htxt'
                if not os.path.isfile(targethdist):
                        print("target file not exists: ",targethdist, " pass!")  
                        continue    
                targetrdist = path_of_Y_evalu[0] + key + '.txt'
                if not os.path.isfile(targetrdist):
                        print("target file not exists: ",targetrdist, " pass!")
                        continue    
                Y_hdist = np.loadtxt(targethdist)
                Y_rdist_intra = Y_hdist[:value,:value]
                Y_hdist_inter = Y_hdist[:value,value:]
                true_contact = np.copy(Y_rdist_intra).squeeze()
                true_contact[true_contact<8] = 1  # 8
                true_contact[true_contact>=8] = 0
                true_complex = np.copy(Y_hdist_inter).squeeze()
                true_complex[true_complex<8] = 1  # 6
                true_complex[true_complex>=8] = 0

                Y_rdist = np.loadtxt(targetrdist)
                Y_rdist_inter = Y_rdist[:value,value:]
                true_complex_r = np.copy(Y_rdist_inter).squeeze()
                true_complex_r[true_complex_r<8] = 1  # 6
                true_complex_r[true_complex_r>=8] = 0

            HomoPred_prediction = HomoPred.predict([selected_list_2D], batch_size= 1)

            topL_prec = 0.0
            topL2_prec = 0.0
            topL5_prec = 0.0
            top5_prec_com = 0.0
            top10_prec_com = 0.0
            topL10_prec_com = 0.0
            topL5_prec_com = 0.0
            top5_prec_com_cb = 0.0
            top10_prec_com_cb = 0.0
            topL10_prec_com_cb = 0.0            
            topL5_prec_com_cb = 0.0
            if 'realdist_hdist_nointra' in predict_method:
                Y_hat_rdist_inter = HomoPred_prediction[0][:,:,:,0:13].sum(axis=-1).squeeze()
                Y_hat_hdist_inter = HomoPred_prediction[1][:,:,:,0:13].sum(axis=-1).squeeze()

                true_complex_vec = true_complex.reshape(-1)
                pred_complex_vec = Y_hat_hdist_inter.reshape(-1)
                top5_prec_com   = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 5)) * 100
                top10_prec_com  = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 10)) * 100
                topL10_prec_com = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, value/10)) * 100
                topL5_prec_com = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, value/5)) * 100

                true_complex_vec = true_complex_r.reshape(-1)
                pred_complex_vec = Y_hat_rdist_inter.reshape(-1)
                top5_prec_com_cb   = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 5)) * 100
                top10_prec_com_cb  = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 10)) * 100
                topL10_prec_com_cb = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, value/10)) * 100
                topL5_prec_com_cb = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, value/5)) * 100
            elif predict_method == 'realdist_hdist_whole':
                Y_hat_rdist_inter = HomoPred_prediction[0][:,:,:,0:13].sum(axis=-1).squeeze()
                Y_hat_hdist_inter = HomoPred_prediction[1][:,:,:,0:13].sum(axis=-1).squeeze()

                Y_hat_rdist_inter = Y_hat_rdist_inter[:value,value:]
                Y_hat_hdist_inter = Y_hat_hdist_inter[:value,value:]
                true_complex_vec = true_complex.reshape(-1)
                pred_complex_vec = Y_hat_hdist_inter.reshape(-1)
                top5_prec_com   = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 5)) * 100
                top10_prec_com  = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 10)) * 100
                topL10_prec_com = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, value/10)) * 100
                topL5_prec_com = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, value/5)) * 100

                true_complex_vec = true_complex_r.reshape(-1)
                pred_complex_vec = Y_hat_rdist_inter.reshape(-1)
                top5_prec_com_cb   = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 5)) * 100
                top10_prec_com_cb  = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 10)) * 100
                topL10_prec_com_cb = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, value/10)) * 100
                topL5_prec_com_cb = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, value/5)) * 100
            elif predict_method == 'realdist_nointra':
                Y_hat_rdist_inter = HomoPred_prediction[:,:,:,0:13].sum(axis=-1).squeeze()

                true_complex_vec = true_complex_r.reshape(-1)
                pred_complex_vec = Y_hat_rdist_inter.reshape(-1)
                top5_prec_com   = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 5)) * 100
                top10_prec_com  = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 10)) * 100
                topL10_prec_com = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, value/10)) * 100
            else:
                Y_hat_rdist = HomoPred_prediction[:,:,:,0:13].sum(axis=-1).squeeze()
                Y_hat_rdist = get_upright(Y_hat_rdist, tar_range = 'long')
                true_complex_vec = true_complex_r.reshape(-1)
                pred_complex_vec = Y_hat_rdist.reshape(-1)
                top5_prec_com_cb   = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 5)) * 100
                top10_prec_com_cb  = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 10)) * 100
                topL10_prec_com_cb = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, value/10)) * 100
                topL5_prec_com_cb  = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, value/5)) * 100

            val_acc_history_content = f'{key} {value} {top5_prec_com:.4f} {top10_prec_com:.4f} {topL10_prec_com:.4f} {topL5_prec_com:.4f} {top5_prec_com_cb:.4f} {top10_prec_com_cb:.4f} {topL10_prec_com_cb:.4f} {topL5_prec_com_cb:.4f}'
            print(val_acc_history_content)
            with open(val_acc_out_inepoch, "a") as myfile:
                myfile.write(val_acc_history_content)
                myfile.write('\n')

            out_top5_prec_com += top5_prec_com
            out_top10_prec_com += top10_prec_com
            out_topL10_prec_com += topL10_prec_com
            out_topL5_prec_com += topL5_prec_com
            out_top5_prec_com_cb += top5_prec_com_cb
            out_top10_prec_com_cb += top10_prec_com_cb
            out_topL10_prec_com_cb += topL10_prec_com_cb
            out_topL5_prec_com_cb += topL5_prec_com_cb
            step_num += 1
        print ('step_num=', step_num)

        out_top5_prec_com /= step_num
        out_top10_prec_com /= step_num
        out_topL10_prec_com /= step_num
        out_topL5_prec_com /= step_num
        out_top5_prec_com_cb /= step_num
        out_top10_prec_com_cb /= step_num
        out_topL10_prec_com_cb /= step_num
        out_topL5_prec_com_cb /= step_num
        avg_prec = (out_topL5_prec_com + out_topL5_prec_com_cb)/2
        val_acc_history_content = f'{epoch} {out_top5_prec_com:.4f} {out_top10_prec_com:.4f} {out_topL10_prec_com:.4f} {out_topL5_prec_com:.4f} {out_top5_prec_com_cb:.4f} {out_top10_prec_com_cb:.4f} {out_topL10_prec_com_cb:.4f} {out_topL5_prec_com_cb:.4f} {avg_prec:.4f}'

        with open(val_acc_history_out, "a") as myfile:
            myfile.write(val_acc_history_content)  
            myfile.write('\n')

        print('The validation accuracy is ',val_acc_history_content)
        if avg_prec >= avg_prec_best:
            avg_prec_best = avg_prec 
            score_imed = "Accuracy Precision of Val: %s:%.4f\t\n" % (epoch, avg_prec_best)
            print("Saved best weight to disk, ", score_imed)
            HomoPred.save_weights(model_weight_out_best)

        if (lr_decay and epoch > 30):
            current_lr = K.get_value(HomoPred.optimizer.lr)
            print("Current learning rate is {} ...".format(current_lr))
            if (epoch % 20 == 0):
                K.set_value(HomoPred.optimizer.lr, current_lr * 0.1)
                print("Decreasing learning rate to {} ...".format(current_lr * 0.1))
        if (lr_decay and epoch > 70):
            current_lr = K.get_value(HomoPred.optimizer.lr)
            print("Current learning rate is {} ...".format(current_lr))
            if (epoch % 10 == 0):
                K.set_value(HomoPred.optimizer.lr, current_lr * 0.1)
                print("Decreasing learning rate to {} ...".format(current_lr * 0.1))

        if predict_method == 'realdist_hdist_nointra':
            print("interchain realdist loss history:", inter_rdist_loss)
            print("interchain heavydist loss history:", inter_hdist_loss)
        elif 'realdist_hdist' in predict_method:
            print("intrachain realdist loss history:", intra_rdist_loss)
            print("interchain realdist loss history:", inter_rdist_loss)
            print("interchain heavydist loss history:", inter_hdist_loss)
        else:      
            print("Train loss history:", train_loss_list)

    print("Training finished, best validation acc = ",avg_prec_best)
    return avg_prec_best