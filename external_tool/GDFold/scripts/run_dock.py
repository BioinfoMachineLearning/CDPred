

import sys
import os



type_of_protein = sys.argv[1]
type_of_res = sys.argv[2]


if type_of_protein == 'dimer' and type_of_res == 'con':
  target_name = sys.argv[3]
  first_pdb = sys.argv[4]
  second_pdb = sys.argv[5]
  res_file = sys.argv[6]
  OUT = sys.argv[7]
  weight_file = sys.argv[8]
  top_num = sys.argv[9]
  
  
  cmd = f'python /data/multicom4s_tool/GD_tools/scripts/dock_from_random_init_original.py {target_name} {first_pdb} {second_pdb} {res_file} {OUT} {weight_file} {top_num}'
  os.system(cmd)
  
elif type_of_protein == 'dimer' and type_of_res == 'dist':
  target_name = sys.argv[3]
  first_pdb = sys.argv[4]
  second_pdb = sys.argv[5]
  res_file = sys.argv[6]
  OUT = sys.argv[7]
  weight_file = sys.argv[8]
  thre = sys.argv[9]
  
  cmd = f'python /data/multicom4s_tool/GD_tools/scripts/dock_from_random_init_original_dist.py {target_name} {first_pdb} {second_pdb} {res_file} {OUT} {weight_file} {thre}'
  os.system(cmd)
  
  

elif type_of_protein == 'multi' and type_of_res == 'con':

  target_name = sys.argv[3]
  pdb_path = sys.argv[4]
  res_file = sys.argv[5]
  OUT = sys.argv[6]
  weight_file = sys.argv[7]
  chains = sys.argv[8]
    
  cmd = f'python /data/multicom4s_tool/GD_tools/scripts/docking_gd_parallel_multi.py {pdb_path} {res_file} {OUT} {weight_file} {target_name} {chains}'
  os.system(cmd)
  

elif type_of_protein == 'multi' and type_of_res == 'dist':

  target_name = sys.argv[3]
  pdb_path = sys.argv[4]
  res_file = sys.argv[5]
  OUT = sys.argv[6]
  weight_file = sys.argv[7]
  chains = sys.argv[8]
  thre = float(sys.argv[9])
    
  cmd = f'python /data/multicom4s_tool/GD_tools/scripts/docking_gd_parallel_multi_dist.py {pdb_path} {res_file} {OUT} {weight_file} {target_name} {chains} {thre}'
  os.system(cmd)
  
  
