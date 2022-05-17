import os
import sys
import argparse
import warnings
from pdb_process import get_sequence_from_pdb
from util import *

#input:  pdb file list and complex stocihiometry
#output: fasta sequence and dimer pairs 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pdb process')
    parser.add_argument('-p', '--pdb_file_list', help="pdb file list", nargs='+', type=str, required=True)
    parser.add_argument('-s', '--stocihiometry', help="Complex stocihiometry", type=str, required=True)
    parser.add_argument('-o', '--out_folder', help="output folder", type=str, required=True)


    args = parser.parse_args()  
    pdb_file_list = args.pdb_file_list
    out_folder = os.path.abspath(args.out_folder)
    stocihiometry = args.stocihiometry

    warnings.filterwarnings('ignore')
    chkdirs(out_folder)

    uni_chain = stocihiometry.split('/')
    chain_number = 0
    homomeric_list = []
    heteromeric_list = []
    new_pdb_list = []
    new_pdb_file_list = []
    for chain in uni_chain:
        chain_id = chain.split(':')[0]
        ori_pdb_file = f'{out_folder}/{chain_id}.pdb'
        if not os.path.exists(ori_pdb_file):
            print("Plase make sure the stocihiometry share the same pdb id with pdb file list")
            sys.exit(1)
        if ':' in chain:
            homo_num = int(chain.split(':')[-1])
        else:
            homo_num = 1
        if homo_num >=1:
            homomeric_list.append(chain_id)
        heteromeric_list.append(chain_id)
        for i in range(homo_num):
            new_chain_id = f'{chain_id}{i}'
            new_pdb_list.append(new_chain_id)
            os.system(f'cp {ori_pdb_file} {out_folder}/{new_chain_id}.pdb')
            new_pdb_file_list.append(f'{out_folder}/{new_chain_id}.pdb')
        chain_number += 1*homo_num
    
    for pdb_file in new_pdb_file_list:
        pdb_file = os.path.abspath(pdb_file)
        os.system(f'cp {pdb_file} {out_folder}')
        name = os.path.basename(pdb_file).split('.')[0]
        out_file = f'{out_folder}/{name}.fasta'
        sequence_list = get_sequence_from_pdb(pdb_file)
        if len(sequence_list) >= 2:
            print('Warning, Please make sure the input pdb file only have one single chain!')
        sequence = sequence_list[0]
        open(out_file, 'w').write(f'>{name}\n{sequence}\n')
        
    # all hetero pairs
    heteromeric_pairs = []
    if len(heteromeric_list) >= 2:
        for i in range(len(heteromeric_list)):
            for j in range(i+1, len(heteromeric_list)):
                heteromeric_pairs.append(f'{heteromeric_list[i]}_{heteromeric_list[j]}')
    # all inter pairs
    all_inter_paris = []
    for i in range(len(new_pdb_list)):
        for j in range(i+1, len(new_pdb_list)):
            all_inter_paris.append(f'{new_pdb_list[i]}_{new_pdb_list[j]}')

    if len(homomeric_list) == 0:
        homomeric_list = '#'
    if len(heteromeric_pairs) == 0:
        heteromeric_pairs = '#'
    print(f"{'|'.join(homomeric_list)} {'|'.join(heteromeric_pairs)} {'|'.join(all_inter_paris)}  {'|'.join(new_pdb_file_list)}")
