

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

import string



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



def append_multi_pdbs(pdb_files):
    new_pdb = []
    
    
    for pdb_file in pdb_files:
     
        with open(pdb_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                if line.startswith('ATOM'):
                    new_pdb.append(line)
        separating_line = 'TER' + '\n'
        new_pdb.append(separating_line)

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
            myfile.write(new_line)



