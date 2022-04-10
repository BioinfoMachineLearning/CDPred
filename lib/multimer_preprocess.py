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
    for pdb_file in pdb_file_list:
        pdb_file = os.path.abspath(pdb_file)
        name = os.path.basename(pdb_file).split('.')[0]
        out_file = f'{out_folder}/{name}.fasta'
        sequence_list = get_sequence_from_pdb(pdb_file)
        if len(sequence_list) >= 2:
            print('Warning, Please make sure the input pdb file only have one single chain!')
        sequence = sequence_list[0]
        open(out_file, 'w').write(f'>{name}\n{sequence}\n')

    uni_chain = stocihiometry.split('/')
    chain_number = 0
    for chain in uni_chain:
        if ':' in chain:
            homo_num = int(chain.split(':')[-1])
        else:
            homo_num = 1
        chain_number += 1*homo_num
    