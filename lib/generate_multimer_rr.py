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
    parser.add_argument('-r', '--rrfile_list', help="rrfile or rrfile list", nargs='+', type=str, required=True)
    parser.add_argument('-cn', '--chain_number', help="number of the chain in multimer", type=int, required=True)
    parser.add_argument('-d', '--distance_threshold', help="less than distance threshold pairs will be select", type=int, default=12, required=False)
    parser.add_argument('-o', '--out_path', help="output folder", type=str, required=True)
    $dist_thred

    args = parser.parse_args()
    name = args.name
    rrfile_list = args.rrfile_list
    chain_number = args.chain_number
    out_path = os.path.abspath(args.out_path)


    if len(rrfile_list) <=1:
        #homomer
        rrfile = os.path.abspath(rrfile_list[0])
        rrlines = open(rrfile, 'r').readlines()
        for i in range(chain_number):
            for j in range(i+1, chain_number):
                for line in rr_lines:
                    distance = float(line.rstrip().split(' ')[-1])
                    if distance <=12:
                        new_line = f'{i+1} {j+1} {line}'
                        open(new_rr_file, 'a').write(new_line)
    else:
        count = 0
        for i in range(chain_number):
            for j in range(i+1, chain_number):
                rrfile = os.path.abspath(rrfile_list[count])
                rrlines = open(rrfile, 'r').readlines()
                for line in rr_lines:
                    distance = float(line.rstrip().split(' ')[-1])
                    if distance <=12:
                        new_line = f'{i+1} {j+1} {line}'
                        open(new_rr_file, 'a').write(new_line)
        #heteromer



