# -*- coding: utf-8 -*-
"""
Created on Wed June 30 21:47:26 2021

@author: Zhiye
"""
import os, sys, glob, re, time, platform, argparse
sys.path.insert(0, sys.path[0])
import numpy as np
from sklearn import metrics
from util import *
from pdb_process import process_pdbfile, get_sequence_from_pdb

def EvaluateComplex(true_complex_vec, pred_complex_vec, tar_length):
    top5_prec_com   = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 5)) * 100
    top10_prec_com  = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, 10)) * 100
    topL10_prec_com = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, tar_length/10)) * 100
    topL5_prec_com = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, tar_length/5)) * 100
    topL2_prec_com = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, tar_length/2)) * 100
    topL_prec_com = metrics.precision_score(true_complex_vec, ceil_topxL_to_one(pred_complex_vec, tar_length)) * 100

    return format(top5_prec_com, '.4f'), format(top10_prec_com, '.4f'), format(topL10_prec_com, '.4f'), format(topL5_prec_com, '.4f'), \
        format(topL2_prec_com, '.4f'), format(topL_prec_com, '.4f')
    # return top5_prec_com, top10_prec_com, topL10_prec_com, topL5_prec_com, topL2_prec_com, topL_prec_com

def get_top_avg_prob(contact_map, top_option='topL5'):
    length = int((contact_map.shape[0] + contact_map.shape[1])/2)
    if top_option == 'topL5':
        contact_num=length/5
    elif top_option == 'topL2':
        contact_num=length/2
    elif top_option == 'topL':
        contact_num=length
    contact_num = int(contact_num)
    contact_vec = contact_map.reshape(-1)
    top_con = contact_vec[np.argpartition(contact_vec, -contact_num)[-contact_num:]]
    return np.mean(top_con)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heavy atom distance map evaluation for CDPred')
    parser.add_argument('-p', '--pred_map', help="predtion distance map", type=str, required=True)
    parser.add_argument('-t', '--true_map', help="ground truth distance map", type=str, required=True)
    parser.add_argument('-f1', '--fasta_file1', help="chain 1 fasta file of dimer", type=str, required=True)
    parser.add_argument('-f2', '--fasta_file2', help="chain 2 fasta file of dimer", type=str, required=True)
    
    args = parser.parse_args()
    pred_map = args.pred_map
    true_map = args.true_map
    fasta_file1 = args.fasta_file1
    fasta_file2 = args.fasta_file2

    pred_file = pred_map
    true_file = true_map           
    if not os.path.exists(pred_file) or not os.path.exists(true_file): 
        print(f'{pred_file} or {true_file} not exists')
        sys.exit(1)
    if not pred_file.endswith('.htxt') or not true_file.endswith('.htxt'):
        print(f'{pred_file} or {true_file} with wrong suffix, please check it!')
        sys.exit(1)       
    if not os.path.exists(fasta_file1) or not os.path.exists(fasta_file2): 
        print(f'{fasta_file1} or {fasta_file2} not exists')
        sys.exit(1)

    tar_name = os.path.basename(true_file).split('.')[0]
    lenA = len(open(fasta_file1, 'r').readlines()[1].strip('\n'))
    lenB = len(open(fasta_file2, 'r').readlines()[1].strip('\n'))

    pred_complex = np.loadtxt(pred_file)
    true_dist = np.loadtxt(true_file)
    tar_length = lenA
    if pred_complex.shape[0] > true_dist.shape[0]:
        pred_complex = pred_complex[:tar_length,tar_length:]
        hdist_inter = true_dist
    elif true_dist.shape[0] > pred_complex.shape[0]:
        hdist_inter = true_dist[:tar_length,tar_length:]
    else:
        hdist_inter = true_dist[:tar_length,tar_length:]
    true_complex = np.copy(hdist_inter).squeeze()
    if not np.max(hdist_inter) < 1:
        true_complex[true_complex<8] = 1  # 6
        true_complex[true_complex>=8] = 0
    true_complex_vec = true_complex.reshape(-1)
    pred_complex_vec = pred_complex.reshape(-1)
    short_len = min(lenA, lenB)
    top5_prec_com, top10_prec_com, topL10_prec_com, topL5_prec_com, topL2_prec_com, topL_prec_com= EvaluateComplex(true_complex_vec, pred_complex_vec, short_len)
    pred_topl2_prob = get_top_avg_prob(pred_complex)
    info_title = f'\n{"NAME".ljust(15)} {"LEN_A".ljust(5)} {"LEN_B".ljust(5)} {"TOP5".ljust(10)} {"TOP10".ljust(10)} {"TOPL/10".ljust(10)} {"TOPL/5".ljust(10)} {"TOPL/2".ljust(10)} {"TOPL".ljust(10)}'
    print(info_title)
    acc_history = f'{tar_name.ljust(15)} {str(lenA).ljust(5)} {str(lenB).ljust(5)} {top5_prec_com.ljust(10)} {top10_prec_com.ljust(10)} {topL10_prec_com.ljust(10)} {topL5_prec_com.ljust(10)} {topL2_prec_com.ljust(10)} {topL_prec_com.ljust(10)}\n'
    print(acc_history)