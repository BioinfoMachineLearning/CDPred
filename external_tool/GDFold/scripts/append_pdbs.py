
import os


def append_pdbs(pdb_file1, pdb_file2, path):
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

    filename1 = os.path.basename(pdb_file1).split('.')[0]
    filename2 = os.path.basename(pdb_file2).split('.')[0]
    
    target_id = f'{filename1}_{filename2}'
    new_pdb_file = path + "\\" + target_id + '.pdb'

    with open(new_pdb_file, 'w') as f:
        for new_line in new_pdb:
            f.write(new_line)


