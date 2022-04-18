# -*- coding: utf-8 -*-
"""
Created on Thru March 10 21:47:26 2022

@author: Zhiye
"""
import os, sys
import time
import numpy as np
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine dimer rr file into multimer rr file')
    parser.add_argument('-n', '--name', help="protein name", type=str, required=True)
    parser.add_argument('-r', '--rr_dir', help="the predmap folder of CDPred which contain all the rr file", type=str, required=True)
    parser.add_argument('-i', '--inter_chain_pairs', help="all the inter-chain pairs of multimers, i.e.T1034A_T1034B|T1034A_T1034C|T1034B_T1034C", type=str, required=True)
    parser.add_argument('-d', '--distance_threshold', help="less than distance threshold pairs will be select", type=int, default=12, required=False)
    parser.add_argument('-o', '--out_path', help="output folder", type=str, required=True)

    args = parser.parse_args()
    name = args.name
    rr_dir = os.path.abspath(args.rr_dir)
    inter_chain_pairs = args.inter_chain_pairs
    out_path = os.path.abspath(args.out_path)

    inter_chain_pair_list = inter_chain_pairs.split('|')
    chain_list = []
    for pair in inter_chain_pair_list:
        chainA = pair.split('_')[0]
        chainB = pair.split('_')[1]
        if not chainA in chain_list:
            chain_list.append(chainA)
        if not chainB in chain_list:
            chain_list.append(chainB)
    chain_num = len(chain_list)

    count = 0
    new_rr_file = f'{rr_dir}/{name}_dist.rr'
    if os.path.exists(new_rr_file):
        os.remove(new_rr_file)
    for i in range(chain_num):
        for j in range(i+1, chain_num):
            inter_chain_pair = inter_chain_pair_list[count]
            count += 1
            chainA = inter_chain_pair.split('_')[0]
            chainB = inter_chain_pair.split('_')[1]
            if chainA[:-1] == chainB[:-1]:
                rrfile = f'{rr_dir}/{chainA[:-1]}_dist.rr'
            else:
                rrfile = f'{rr_dir}/{chainA[:-1]}_{chainB[:-1]}_dist.rr'
            rrlines = open(rrfile, 'r').readlines()
            for line in rrlines:
                distance = float(line.rstrip().split(' ')[-1])
                if distance <=12:
                    new_line = f'{i+1} {j+1} {line}'
                    open(new_rr_file, 'a').write(new_line)




