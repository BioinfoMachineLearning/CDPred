import os
import sys


from append_pdbs import append_pdbs

sys.path.append(r"C:\Program Files\PyMOL\AxPyMOL\AxPyMOL\Lib\site-packages")


targets_path = "C:\\Users\\nsolt\\Desktop\\pdb_zhiye\\dist_res_experimenst\\targets_casp1314_hetero.txt"
chain_files = "C:\\Users\\nsolt\\Desktop\\pdb_zhiye\\dist_res_experimenst\\pdb_hetero\\chains\\"
true_pdbs_path = "C:\\Users\\nsolt\\Desktop\\pdb_zhiye\\dist_res_experimenst\\pdb_hetero\\true_pdbs\\"



def get_targets(file_path):
    targets = []
    with open(file_path, 'r') as f:
        content = f.readlines()

        for line in content:
            targets.append(line.strip())

    return targets


def make_pdbs():
    targets = get_targets(targets_path)

    for i in range(len(targets)):
        first_chain = chain_files + "\\" + targets[i].split('_')[0] + ".pdb"
        second_chain = chain_files + "\\" + targets[i].split('_')[1] + ".pdb"

        if not os.path.isfile(first_chain) or not os.path.isfile(second_chain):
            print(first_chain, second_chain)
        else:
            append_pdbs(first_chain, second_chain, true_pdbs_path)


make_pdbs()

