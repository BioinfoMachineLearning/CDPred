import os,sys
import numpy as np
from Bio.PDB.PDBParser import PDBParser
import json

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

def build_dataset_dictionaries(data_file, Maximum_length=500, shuffle=False):
    shuffle = shuffle
    line_list = open(data_file).readlines()
    if shuffle==True:
        random.shuffle(line_list)
    data_dict = {}
    for line in line_list:
        line = line.strip('\n').split(' ')
        if len(line) < 3:
            tar_name = line[0]
            tar_length = line[1]
            if int(tar_length) > Maximum_length or int(tar_length) < 30:
                continue
            data_dict[tar_name] = tar_length
        else:
            tar_name = line[0]
            lenA = int(line[1])
            lenB = int(line[2])
            tar_length = lenA + lenB
            if (tar_length) > Maximum_length or int(tar_length) < 30:
                continue
            data_dict[tar_name] = f'{lenA}_{lenB}'
    return data_dict

# short[6-11], medium[12-23], long[24,~]  less 16A
def get_upright(input_map, tar_range = 'long'):
    L = int(input_map.shape[0])
    vector = []
    if tar_range == 'long':
        for i in range(0, L):
            for j in range(i+24, L):
                vector.append(input_map[i,j])
    elif tar_range == 'medium':
        for i in range(0, L):
            for j in range(i+12, i+24):
                if j >= L:
                    continue
                vector.append(input_map[i,j])
    elif tar_range == 'short':
        for i in range(0, L):
            for j in range(i+6, i+12):
                if j >= L:
                    continue
                vector.append(input_map[i,j])
    elif type(tar_range) == int:
        for i in range(0, L):
            for j in range(i+tar_range, L):
                if j >= L:
                    continue
                vector.append(input_map[i,j])
    return np.array(vector)

def ceil_topxL_to_one(Y_hat, x):
    Y_ceiled = np.copy(Y_hat)
    xL = int(x)
    Y_ceiled[:] = np.zeros(len(Y_hat[:]))
    Y_ceiled[np.argpartition(Y_hat[:], -xL)[-xL:]] = 1
    return Y_ceiled.astype(int)

def get_cb_dist_from_pdbfile(pdb_file, length):
    seq_name = os.path.basename(pdb_file).split('.')[0]
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(seq_name, pdb_file)
    model = structure[0]
    chain_id = list(model.child_dict.keys())
    chain = model[chain_id[0]]

    dist_map = np.zeros((length, length))

    L = len(chain)
    print(L, length)
    if L != length:
        print('got length error of %s, should be %s, got %s'%(pdb_file, length, L))
        return dist_map

    chain_dict = list(chain.child_dict.keys())
    for i in range(0, length):
        for j in range(i, length):
            if i == j:
                continue
            residue_i = chain[chain_dict[i]]
            residue_j = chain[chain_dict[j]]
            try:
                cb_i = residue_i["CB"]
            except KeyError as e:
                cb_i = residue_i["CA"]
            try:
                cb_j = residue_j["CB"]
            except KeyError as e:
                cb_j = residue_j["CA"]
            cbcb_dist = cb_i - cb_j
            dist_map[i, j] = cbcb_dist
    dist_map += dist_map.T
    return dist_map

def get_heavyatom_dist_from_pdbfile(pdb_file, length):
    seq_name = os.path.basename(pdb_file).split('.')[0]
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(seq_name, pdb_file)

    model = structure[0]

    chain_id = list(model.child_dict.keys())
    chain = model[chain_id[0]]

    h_dist_map = np.zeros((length, length))

    L = len(chain)
    print(L, length)
    if L != length:
        print('got length error of %s, should be %s, got %s'%(pdb_file, length, L))
        return h_dist_map

    chain_dict = list(chain.child_dict.keys())
    for i in range(0, length):
        for j in range(i, length):
            if i == j:
                continue
            res_i = chain[chain_dict[i]]
            res_j = chain[chain_dict[j]]
            dist_list =[]
            for atom_i in res_i:
                for atom_j in res_j:
                    if ('C' in atom_i.name or 'N' in atom_i.name or 'O' in atom_i.name or 'S' in atom_i.name) and \
                        ('C' in atom_j.name or 'N' in atom_j.name or 'O' in atom_j.name or 'S' in atom_j.name):
                        dist_list.append(atom_i-atom_j)
                    else:
                        continue
            min_dist = np.min(dist_list) 
            h_dist_map[i, j] = min_dist
    h_dist_map += h_dist_map.T
    return h_dist_map

def get_top5_model_from_pool(name, tar_dir):
    repeat_dirs = os.listdir(tar_dir)
    plddts_list = []
    pdb_name_list = []
    for dir_name in repeat_dirs:
        full_dirname = f'{tar_dir}/{dir_name}/{name}/'
        if not os.path.exists(full_dirname):
            continue
        json_file = f'{full_dirname}/ranking_debug.json'
        if not os.path.exists(json_file):
            continue
        f = open(json_file, 'r')
        json_dict = json.load(f)
        order = json_dict['order']
        if 'iptm+ptm' in json_dict.keys():
            plddts = json_dict['iptm+ptm']
        else:
            plddts = json_dict['plddts']
        ranked = 0
        for i in order:
            temp_plddt = plddts[i]
            temp_pdb_file = f'{full_dirname}/ranked_{ranked}.pdb'
            ranked += 1
            plddts_list.append(temp_plddt)
            pdb_name_list.append(temp_pdb_file)
            print(temp_plddt, temp_pdb_file)
    top5_lddts_index = np.argsort(plddts_list)[-5:]
    final_model_dir = f'{tar_dir}/TOP5/'
    plddt_file = f'{final_model_dir}/plddts.txt'
    if os.path.exists(plddt_file):
        os.remove(plddt_file)
    if not os.path.exists(final_model_dir):
        os.mkdir(final_model_dir)
    count = len(top5_lddts_index)
    for index in top5_lddts_index:
        top_pdb_file = f'{final_model_dir}/{name}_{count}.pdb'
        count -= 1
        print(f'cp {pdb_name_list[index]} {top_pdb_file}')
        open(plddt_file, 'a').write(f'{top_pdb_file} {plddts_list[index]}\n')
        os.system(f'cp {pdb_name_list[index]} {top_pdb_file}')
    return final_model_dir