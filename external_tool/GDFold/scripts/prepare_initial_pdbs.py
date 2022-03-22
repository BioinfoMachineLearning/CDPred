
import os
import os.path
import time
import glob
import sys

targets_path = "C:\\Users\\nsolt\\Desktop\\pdb_zhiye\\dist_res_experimenst\\targets_casp1314_homo.txt"
pdbs_path = "C:\\Users\\nsolt\\Desktop\\pdb_zhiye\\dist_res_experimenst\\pdb_homo\\chains\\"
current_dir = os.getcwd()

pdbNames = []
with open(targets_path, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            clos = line.strip().split()
            pdbNames.append(clos[0])
            

for target in pdbNames:
    first_pdb = pdbs_path + "\\" + target.split('_')[0] + ".pdb"
    second_pdb = pdbs_path + "\\" + target.split('_')[1] + ".pdb"
    #first_pdb = pdbs_path + "\\" + target + "_A.pdb"
    #second_pdb = pdbs_path + "\\" + target + "_B.pdb"
    
    if not os.path.isfile(first_pdb) or not os.path.isfile(second_pdb):
        print(first_pdb, second_pdb)
    else:
        cmd = "python " + current_dir + "\\rotate_translate_pdbs.py " + first_pdb + " " + second_pdb + " " + target
        os.system(cmd)
    
    
