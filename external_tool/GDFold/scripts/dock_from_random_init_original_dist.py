

import os
import sys
import glob
from math import sin, cos
import math
import random
from random import randrange as rand_num
import numpy as np

import sys
from rosetta import *
from pyrosetta import *
from rosetta.protocols.rigid import *
from rosetta.core.scoring import *
from pyrosetta import PyMOLMover
from rosetta.protocols.rigid import *
import pyrosetta.rosetta.protocols.rigid as rigid_moves
import pyrosetta.rosetta.protocols.rigid as rigid_moves



def add_chain(pdb_file, letter):
    new_pdb = []
    with open(pdb_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith('ATOM'):
                newline = line[0:21] + letter + line[22:]
                new_pdb.append(newline)
            # else:
            #     new_pdb.append(line)
    # with open(pdb_file, 'w') as f:
    #     for new_line in new_pdb:
    #         f.write(new_line)
    return new_pdb
            
def append_pdbs(pdb_file1, pdb_file2):
    new_pdb = []
    with open(pdb_file1, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith('ATOM'):
                new_pdb.append(line)
    separating_line = 'TER' + '\n'
    new_pdb.append(separating_line)

    with open(pdb_file2, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith('ATOM'):
                new_pdb.append(line)

    end_line = 'END'
    new_pdb.append(end_line)
    return new_pdb

def get_rotation_matrix(axis_name, degree_magnitude):
    degree_magnitude = math.radians(degree_magnitude)
    if axis_name == 'x':
        rotation_matrix = np.array([[1, 0, 0],[0, cos(degree_magnitude), -sin(degree_magnitude)],[0, sin(degree_magnitude), cos(degree_magnitude)]])
    elif axis_name == 'y':
        rotation_matrix = np.array([[cos(degree_magnitude), 0, sin(degree_magnitude)],[0, 1, 0],[-sin(degree_magnitude), 0, cos(degree_magnitude)]])
    elif axis_name == 'z':
        rotation_matrix = np.array([[cos(degree_magnitude), -sin(degree_magnitude), 0],[sin(degree_magnitude), cos(degree_magnitude), 0],[0, 0, 1]])

    return rotation_matrix

def rotatePose(pose, R):
    start_A = pose.conformation().chain_begin(1)
    end_A = pose.conformation().chain_end(1)
    for r in range(start_A, end_A+1):
        for a in range(1, len(pose.residue(r).atoms())+1):
            v = np.array([pose.residue(r).atom(a).xyz()[0], pose.residue(r).atom(a).xyz()[1], pose.residue(r).atom(a).xyz()[2]])
            newv = R.dot(v)
            pose.residue(r).atom(a).xyz(numeric.xyzVector_double_t(newv[0], newv[1], newv[2]))
            
    return pose
    
def translatePose(pose, t):
    start_A = pose.conformation().chain_begin(1)
    end_A = pose.conformation().chain_end(1)
    for r in range(start_A, end_A+1):
        for a in range(1, len(pose.residue(r).atoms())+1):
            newx = pose.residue(r).atom(a).xyz()[0] + t[0]
            newy = pose.residue(r).atom(a).xyz()[1] + t[1]
            newz = pose.residue(r).atom(a).xyz()[2] + t[2]
            pose.residue(r).atom(a).xyz(numeric.xyzVector_double_t(newx, newy, newz))
            
    return pose
        
def setResFilebyTopNum(in_res_file, out_res_file, top_num):
    line_list = open(in_res_file, 'r').readlines()
    count = 0
    with open(out_res_file, 'a') as myfile:
        myfile.write(line_list[0])
        for line in line_list[1:]:
            count += 1
            if count >= top_num:
                new_line = ' '.join(line.strip('\n').split(' ')[:4]+list('0'))+'\n'
            else:
                new_line = ' '.join(line.strip('\n').split(' ')[:4]+list('1'))+'\n'
            myfile.write(newline)


if __name__=="__main__":
    if len(sys.argv) != 8:
        print('Wrong input parameters\n\n')
        print(len(sys.argv))
        exit()
    target_name = sys.argv[1]
    first_pdb = sys.argv[2]
    second_pdb = sys.argv[3]
    res_file = sys.argv[4]
    OUT = sys.argv[5]
    weight_file = sys.argv[6]
    thre = sys.argv[7]

    script_path = os.path.split(os.path.realpath(__file__))[0]
    
    select_top = False
    if not os.path.exists(OUT):
        os.mkdir(OUT)
    os.chdir(OUT)
    init()
    
    
    chain1 = target_name.split('_')[0][-1]
    chain2 = target_name.split('_')[1][-1]
    
    target_id = target_name.split('_')[0][:-1]
    
    chainA_file = f'{OUT}/{target_id}{chain1}.pdb'
    chainB_file = f'{OUT}/{target_id}{chain2}.pdb'
    init_pdb_file = f'{OUT}/{target_id}.pdb'

    chainA = add_chain(first_pdb, chain1)
    chainB = add_chain(second_pdb, chain2)
    open(chainA_file, 'w').write(''.join(chainA))
    open(chainB_file, 'w').write(''.join(chainB))
    initial_start = append_pdbs(chainA_file, chainB_file)
    open(init_pdb_file, 'w').write(''.join(initial_start))

    pose = pyrosetta.pose_from_pdb(init_pdb_file)

    pose = translatePose(pose, [rand_num(1, 60), 0, 0]).clone()

    for i in range(40):
        R_X = get_rotation_matrix('x', rand_num(1, 360))
        R_Y = get_rotation_matrix('y', rand_num(1, 360))
        R_Z = get_rotation_matrix('z', rand_num(1, 360))
        
        pose = rotatePose(pose, R_X).clone()
        pose = rotatePose(pose, R_Y).clone()
        pose = rotatePose(pose, R_Z).clone()
        
        
    pose = translatePose(pose, [0, rand_num(1, 60), 0]).clone()
    pose = translatePose(pose, [0, 0, rand_num(1, 60)]).clone()

    init_pdb_file = f'{OUT}/{target_id}.pdb'
    pose.dump_pdb(init_pdb_file)
    
    
    cmd = f'python {script_path}/docking_new_dist.py {init_pdb_file} {res_file} {OUT} {weight_file} 100 {thre} {target_name}'
    os.system(cmd)
