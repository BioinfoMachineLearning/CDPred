
import os
import os.path
import glob
import sys
import re

#dir_path = sys.argv[1]
dir_path = 'C:\\Users\\nsolt\\Desktop\\pdb_zhiye\\dist_res_experimenst\\pdb_hetero\\true_pdbs\\'

f = open('C:\\Users\\nsolt\\Desktop\\pdb_zhiye\\dist_res_experimenst\\targets_casp1314_hetero.txt', 'a')

print(os.getcwd())

fileNames = glob.glob(dir_path+"/*pdb")

print(len(fileNames))

for file_path in fileNames:

    file_name = os.path.basename(file_path)
    target_id = file_name.split('.')[0]

    f.write(target_id)
    f.write('\n')
    