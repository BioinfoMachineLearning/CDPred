import os
import sys
import argparse
from Bio import SeqIO
import warnings
from util import *

#'TER', 'END'
def delete_elements(fhandle, element='TER'):
    for line in fhandle:
        if line.startswith(element):
                continue
        yield line
    
def pad_line(line):
    """Helper function to pad line to 80 characters in case it is shorter"""
    size_of_line = len(line)
    if size_of_line < 80:
        padding = 80 - size_of_line + 1
        line = line.strip('\n') + ' ' * padding + '\n'
    return line[:81]  # 80 + newline character

#starting_resid = -1
def renumber_residues(fhandle, starting_resid):
    """Resets the residue number column to start from a specific number.
    """
    _pad_line = pad_line
    prev_resid = None  # tracks chain and resid
    resid = starting_resid - 1  # account for first residue
    records = ('ATOM', 'HETATM', 'TER', 'ANISOU')
    for line in fhandle:
        line = _pad_line(line)
        if line.startswith(records):
            line_resuid = line[17:27]
            if line_resuid != prev_resid:
                prev_resid = line_resuid
                resid += 1
                if resid > 9999:
                    emsg = 'Cannot set residue number above 9999.\n'
                    sys.stderr.write(emsg)
                    sys.exit(1)

            yield line[:22] + str(resid).rjust(4) + line[26:]

        else:
            yield line

#starting_value = -1
def renumber_atom_serials(fhandle, starting_value):
    """Resets the atom serial number column to start from a specific number.
    """

    # CONECT 1179  746 1184 1195 1203
    fmt_CONECT = "CONECT{:>5s}{:>5s}{:>5s}{:>5s}{:>5s}" + " " * 49 + "\n"
    char_ranges = (slice(6, 11), slice(11, 16),
                   slice(16, 21), slice(21, 26), slice(26, 31))

    serial_equiv = {'': ''}  # store for conect statements

    serial = starting_value
    records = ('ATOM', 'HETATM')
    for line in fhandle:
        if line.startswith(records):
            serial_equiv[line[6:11].strip()] = serial
            yield line[:6] + str(serial).rjust(5) + line[11:]
            serial += 1
            if serial > 99999:
                emsg = 'Cannot set atom serial number above 99999.\n'
                sys.stderr.write(emsg)
                sys.exit(1)

        elif line.startswith('ANISOU'):
            # Keep atom id as previous atom
            yield line[:6] + str(serial - 1).rjust(5) + line[11:]

        elif line.startswith('CONECT'):
            # 6:11, 11:16, 16:21, 21:26, 26:31
            serials = [line[cr].strip() for cr in char_ranges]

            # If not found, return default
            new_serials = [str(serial_equiv.get(s, s)) for s in serials]
            conect_line = fmt_CONECT.format(*new_serials)

            yield conect_line
            continue

        elif line.startswith('MODEL'):
            serial = starting_value
            yield line

        elif line.startswith('TER'):
            yield line[:6] + str(serial).rjust(5) + line[11:]
            serial += 1

        else:
            yield line

def alter_chain(fhandle, chain_id):
    """Sets the chain identifier column in all ATOM/HETATM records to a value.
    """
    _pad_line = pad_line
    records = ('ATOM', 'HETATM', 'TER', 'ANISOU')
    for line in fhandle:
        if line.startswith(records):
            line = _pad_line(line)
            yield line[:21] + chain_id + line[22:]
        else:
            yield line

def write_pdb_file(new_pdb, pdb_file):
    if os.path.exists(pdb_file):
        os.remove(pdb_file)
    try:
        _buffer = []
        _buffer_size = 5000  # write N lines at a time
        for lineno, line in enumerate(new_pdb):
            if not (lineno % _buffer_size):
                open(pdb_file, 'a').write(''.join(_buffer))
                _buffer = []
            _buffer.append(line)

        open(pdb_file, 'a').write(''.join(_buffer))
    except IOError:
        # This is here to catch Broken Pipes
        # for example to use 'head' or 'tail' without
        # the error message showing up
        pass

def get_sequence_from_pdb(pdb_file):
    fh = open(pdb_file, 'r')
    sequence_list = []
    for record in SeqIO.parse(fh, 'pdb-atom'):
        sequence_list.append(str(record.seq))
    return sequence_list

#process complex_pdb_file 
def process_pdbfile(in_file, out_file):
    fhandle = open(in_file, 'r')
    fhandle = delete_elements(fhandle, 'TER')
    fhandle = alter_chain(fhandle, 'A')
    fhandle = renumber_residues(fhandle, 1)
    fhandle = renumber_atom_serials(fhandle, 1)
    write_pdb_file(fhandle, out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pdb process')
    parser.add_argument('-p', '--pdb_file', help="pdb file", type=str, required=True)
    parser.add_argument('-o', '--out_folder', help="output folder", type=str, required=True)
    parser.add_argument('-op', '--option', help="Choose the function want to run", type=str, required=True)


    args = parser.parse_args()  
    pdb_file = os.path.abspath(args.pdb_file)
    out_folder = os.path.abspath(args.out_folder)
    option = args.option
    fhandle = open(pdb_file, 'r')

    warnings.filterwarnings('ignore')
    chkdirs(out_folder)

    name = os.path.basename(pdb_file).split('.')[0]
    if option == 'process_pdbfile':
        out_file = f'{out_folder}/{name}.pdb'
        fhandle = delete_elements(fhandle, 'TER')
        fhandle = alter_chain(fhandle, 'A')
        fhandle = renumber_residues(fhandle, 1)
        fhandle = renumber_atom_serials(fhandle, 1)
        write_pdb_file(fhandle, out_file)
    elif option == 'get_sequence_from_pdb':
        out_file = f'{out_folder}/{name}.fasta'
        sequence_list = get_sequence_from_pdb(pdb_file)
        if len(sequence_list) >= 2:
            print('Warning, Please make sure the input pdb file only have one single chain!')
        sequence = sequence_list[0]
        open(out_file, 'w').write(f'>{name}\n{sequence}\n')
    print(f'Save results in {os.path.abspath(out_file)}')