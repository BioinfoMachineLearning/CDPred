import os
import sys
import random
from random import randrange as rand_num
sys.path.append(r"C:\Program Files\PyMOL\AxPyMOL\AxPyMOL\Lib\site-packages")
import __main__
__main__.pymol_argv = ['pymol','-qc'] # Pymol: quiet and no GUI
import pymol
pymol.finish_launching()

pdb_file1 =sys.argv[1]
pdb_file2 = sys.argv[2]
target_name = sys.argv[3]

#pdb_name =pdb_file.split('.')[0]
#pdb_name1 =pdb_file1.split('.')[0]

pdb_name1 = os.path.basename(pdb_file1)
pdb_name2 = os.path.basename(pdb_file2)



pymol.cmd.load(pdb_file1, pdb_name1)
pymol.cmd.load(pdb_file2, pdb_name2)
pymol.cmd.disable("all")
pymol.cmd.enable(pdb_name1)
pymol.cmd.enable(pdb_name2)
print (pymol.cmd.get_names())


pymol.cmd.translate([rand_num(1,50), rand_num(1,50), rand_num(1,50)], pdb_name1)

for i in range(40):
    pymol.cmd.rotate("x", rand_num(1,360), pdb_name1)
    pymol.cmd.rotate("y", rand_num(1,360), pdb_name1)
    pymol.cmd.rotate("z", rand_num(1,360), pdb_name1)


pymol.cmd.translate([rand_num(1,50), rand_num(1,50), rand_num(1,50)], pdb_name2)

for i in range(40):
    pymol.cmd.rotate("x", rand_num(1,360), pdb_name2)
    pymol.cmd.rotate("y", rand_num(1,360), pdb_name2)
    pymol.cmd.rotate("z", rand_num(1,360), pdb_name2)


print(target_name)
save_path = "C:\\Users\\nsolt\\Desktop\\pdb_zhiye\\dist_res_experimenst\\intial_start_homo\\" + "\\" + target_name + ".start.pdb"
pymol.cmd.save(save_path)

pymol.cmd.quit()