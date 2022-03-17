import os, sys, math, random
import esm, torch, string, time
import numpy as np
from subprocess import Popen, PIPE, STDOUT, run
from Bio import SeqIO
from data import Alphabet
import itertools
from typing import List, Tuple
import argparse
from constants import *
## DBTOOL_FLAG
# db_tool_dir='/storage/htc/bdm/zhiye/DNCON4_db_tools/'
tempdir = '/var/tmp/'

PLM  =  os.path.dirname(os.path.split(os.path.realpath(__file__))[0]) + '/bin/ccmpred' #PLM x.aln x.mat x.plm
PSSM  =  os.path.dirname(os.path.split(os.path.realpath(__file__))[0]) + '/bin/genpssm.pl' #PLM x.aln x.mat x.plm
ESM1b = os.path.dirname(os.path.split(os.path.realpath(__file__))[0]) + '/bin/esm_msa1b_t12_100M_UR50S.pt' 
# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def get_feature_lists(reject_fea_file):
    accept_list = []
    if reject_fea_file != 'None':
        with open(reject_fea_file) as f:
            for line in f:
                if line.startswith('#'):
                    feature_name = line.strip()
                    feature_name = feature_name[0:]
                    accept_list.append(feature_name)
    return accept_list

#return feature size (L, L, 441)
def computeplm(name, msafile, save_ccmpred_path=None):
    os.chdir(tempdir)
    cmd = PLM + ' ' + msafile + ' ' + tempdir + name + '.mat ' + tempdir + name + '.plm'
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output,error = p.communicate()
    if not os.path.exists(tempdir + name + '.plm'):
        print('feature generate error %s'%name)
        print(cmd)
        return False
    if save_ccmpred_path is not None and os.path.exists(save_ccmpred_path):
        os.system('cp %s %s'%(tempdir + name + '.mat', save_ccmpred_path+'/'))
    else:
        plm_rawdata = np.fromfile(tempdir + name + '.plm', dtype=np.float32)
        L = int(math.sqrt(plm_rawdata.shape[0]/21/21))
        try:
            inputs_plm = plm_rawdata.reshape(441,L,L)
        except ValueError:
            return False
        plm = inputs_plm.transpose(1,2,0)
    if os.path.exists(tempdir + name + '.mat'):
        os.remove(tempdir + name + '.mat')
    if os.path.exists(tempdir + name + '.plm'):
        os.remove(tempdir + name + '.plm')
    return True

def compute_ccmpred(name, msafile, save_ccmpred_path):
    cmd = f'{PLM} {msafile} {save_ccmpred_path}/{name}.mat'
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output,error = p.communicate()
    if not os.path.exists(f'{save_ccmpred_path}/{name}.mat'):
        print('feature generate error %s'%name)
        print(cmd)
        return False
    return True

#return feature size (L,L,40)
def computepssm_fromfile(pssmfile):
    pssm_rawdata = np.loadtxt(pssmfile)
    pssm_rawdata = pssm_rawdata.transpose(1, 0)
    L = pssm_rawdata.shape[0]
    ch = pssm_rawdata.shape[1]
    pssm = pssm_rawdata.reshape([L,1,ch])
    pssm = pssm.repeat(L, axis=1)
    pssmT = np.copy(pssm)
    pssmT = pssmT.transpose(1,0,2)
    pssm = np.concatenate([pssm, pssmT], axis=-1)
    return pssm

#return feature size (L, L, 44)
def computepssm(name, fasta, outdir, unirefdb):
    os.chdir(outdir)
    cmd = f'perl {PSSM} {fasta} {outdir} {unirefdb}'
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output,error = p.communicate()
    if not os.path.exists(tempdir + name + '_pssm.txt'):
        print('feature generate error %s'%name)
        print(cmd)
        print(output)
        print(error)
        return False
    else:
        return True

def dropaln(name, msafile, rate=0):
    tempfile = tempdir + name + '.aln'
    lines = open(msafile, 'r').readlines()
    slices = random.sample(lines, math.ceil(len(lines) * (1 - rate))) 
    open(tempfile, 'w').write(''.join(slices))
    # print(tempfile, len(lines), len(slices))
    return tempfile

# from data_rowatt import Alphabet
def pre_load_esm(if_train=True):
    global device
    global msa_transformer
    global msa_batch_converter
    if if_train:
        device = 'cpu'
    else:
        device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    msa_transformer, msa_alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
    # msa_transformer, msa_alphabet = esm.pretrained.load_model_and_alphabet(ESM1b)
    msa_transformer = msa_transformer.eval().to(device)
    msa_alphabet = Alphabet.from_architecture('msa_transformer')
    msa_batch_converter = msa_alphabet.get_batch_converter()
    
#return [L, L, 144]
def computerowatt(name, a3mfile, depth=64): 
    if not 'device' in dir() or not 'msa_transformer' in dir() or not 'msa_batch_converter' in dir():
        device = 'cpu'
        msa_transformer, msa_alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
        # msa_transformer, msa_alphabet = esm.pretrained.load_model_and_alphabet(ESM1b)
        msa_transformer = msa_transformer.eval().to(device)
        msa_alphabet = Alphabet.from_architecture('msa_transformer')
        msa_batch_converter = msa_alphabet.get_batch_converter()
        
    tar_name = name
    msa_data = read_msa(a3mfile, depth)
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    msa_batch_tokens = msa_batch_tokens.to(device)
    L = msa_batch_tokens.size()[-1]
    ch = 144
    results = msa_transformer(msa_batch_tokens, return_contacts=True)
    rowatt_map = results['row_attentions']
    rowatt_map = rowatt_map.cpu().data.numpy()
    rowatt_map = rowatt_map.reshape([ch, L, L])
    rowatt_map = rowatt_map.transpose(1, 2, 0)
    return rowatt_map

def get_crop_index(fasta_len, crop_len=512, crop_num=4):
    step_len = int(np.ceil((fasta_len - crop_len)/(crop_num-1)))
    crop_set = np.ones((crop_num+1, fasta_len)) * -1
    for i in range(crop_num):
        start = i*step_len
        if i*step_len + crop_len <=fasta_len:
            end = i*step_len + crop_len
        else:
            end=fasta_len
        sub_set = list(range(start, end ,1))
        crop_set[i, sub_set]=sub_set
    last_dim = list(set(crop_set[0,:]).difference(set(crop_set[-2,:]))) + list(set(crop_set[-2,:]).difference(set(crop_set[0,:])))
    last_dim = list(map(int, last_dim))
    crop_set[-1, last_dim]=last_dim
    return crop_set.astype(int)

#return [L, L, 144]
def computerowatt_over1024(name, a3mfile, outdir, depth=64): 
    rowatt_file = outdir + '/' + name + '.npy'
    ori_a3m_lines = open(a3mfile, 'r').readlines()
    ori_seq = ori_a3m_lines[1].strip('\n')
    if len(ori_seq) > 1024: # msa transformer max length
        print('Large than 1024, start crop!')
        crop_index = get_crop_index(len(ori_seq), crop_len=1000, crop_num=4)
        crop_a3m_dir = f'{outdir}/tmp/'
        if not os.path.exists(crop_a3m_dir): os.mkdir(crop_a3m_dir)
        fea = np.zeros((len(ori_seq), len(ori_seq), 144)).astype(np.float32)
        for i in range(len(crop_index)):
            sub_crop = crop_index[i]
            crop_keep = sub_crop[sub_crop>=0]
            crop_a3m_file = f'{crop_a3m_dir}/{name}_{i}.a3m'
            crop_rowatt_file = f'{crop_a3m_dir}/{name}_{i}.npy'
            with open(crop_a3m_file, 'w') as myfile:
                for line in ori_a3m_lines:
                    if line.startswith('>'):
                        myfile.write(line)
                    else:
                        line = line.strip('\n')
                        line_arr = np.array(list(line))
                        new_line = line_arr[crop_keep]
                        new_line = ''.join(new_line) + '\n'
                        myfile.write(new_line)
            try:
                fea_crop = computerowatt(name, crop_a3m_file, depth=depth)
                np.save(crop_rowatt_file, fea_crop)
                fea_crop = fea_crop.reshape(-1, 144)
                count = 0
                for i in crop_keep:
                    for j in crop_keep:
                        if fea[i, j, :].all() == 0:
                            fea[i, j, :] = fea_crop[count, :]
                        else:
                            fea[i, j, :] = (fea[i, j, :] + fea_crop[count, :])/2
                        count+=1
            except:
                continue
    else:
        fea = computerowatt(name, a3mfile, depth=depth)
    np.save(rowatt_file, fea)
    print(fea.shape)
    print('save file to %s'%rowatt_file)

#[ch, L, L] --> [L, L, ch]
def load_rowatt_from_file(rowatt_file):
    rowatt = np.load(rowatt_file)
    if rowatt.shape[0] == 144:
        rowatt = rowatt.transpose(1, 2, 0)
    return rowatt

def get2d_feature_by_list(name, featurelist, msafile, rate=0, **fea_file):
    if rate > 0 and rate < 1:
        msafile = dropaln(name, msafile, rate=rate)
    plm_file = None
    pssm_file = None
    rowatt_file = None
    rowatt_inter_file = None
    a3m_file = None
    ccmpred_file = None
    pred_dist_file_cb = None
    pred_dist_file_hv = None
    if 'plm_file' in fea_file.keys():
        plm_file = fea_file['plm_file']
    if 'ccmpred_file' in fea_file.keys():
        ccmpred_file = fea_file['ccmpred_file']
    if 'pssm_file' in fea_file.keys():
        pssm_file = fea_file['pssm_file']
    if 'rowatt_file' in fea_file.keys():
        rowatt_file = fea_file['rowatt_file']
    if 'rowatt_diff_file' in fea_file.keys():
        rowatt_diff_file = fea_file['rowatt_diff_file']
    if 'rowatt_inter_file' in fea_file.keys():
        rowatt_inter_file = fea_file['rowatt_inter_file']
    if 'a3m_file' in fea_file.keys():
        a3m_file = fea_file['a3m_file']
    if 'pred_dist_file_cb' in fea_file.keys():
        pred_dist_file_cb = fea_file['pred_dist_file_cb']
    feature = False
    featuer_number = len(featurelist)
    if '# plm' in featurelist:
        if plm_file is not None:
            feature = np.load(plm_file)
        else:
            feature = computeplm(name, msafile)
    if '# ccmpred' in featurelist:
        if ccmpred_file is not None:
            feature = np.loadtxt(ccmpred_file)
            feature = feature[:,:, np.newaxis]
        else:
            return False
    if '# pssm' in featurelist:
        if featuer_number !=0 and isinstance(feature, bool) == False:
            if pssm_file is not None:
                feature = np.concatenate([feature, computepssm_fromfile(pssm_file)], axis = -1)
            else:
                feature = np.concatenate([feature, computepssm(name, msafile)], axis = -1)
        elif featuer_number == 0 and isinstance(feature, bool) == False:
            if pssm_file is not None:
                feature = computepssm_fromfile(pssm_file)
            else:
                feature = computepssm(name, msafile)
        else:
            if pssm_file is not None:
                feature = computepssm_fromfile(pssm_file)
            else:
                feature = computepssm(name, msafile)
    if '# rowatt' in featurelist:
        if featuer_number !=0 and isinstance(feature, bool) == False:         
            if rowatt_file is not None:
                feature = np.concatenate([feature, load_rowatt_from_file(rowatt_file)], axis = -1)
            elif a3m_file is not None:
                feature = np.concatenate([feature, computerowatt(name, a3m_file)], axis = -1)
        else:
            if rowatt_file is not None:
                feature = load_rowatt_from_file(rowatt_file)
            elif a3m_file is not None:
                feature = computerowatt(name, a3m_file)
    if '# intradist_cb' in featurelist:
        if featuer_number !=0 and isinstance(feature, bool) == False:        
            if pred_dist_file_cb is not None:
                intra_dist = np.loadtxt(pred_dist_file_cb)
                intra_dist = intra_dist[:,:, np.newaxis]
                if intra_dist.shape[0] != feature.shape[0]:
                    print(intra_dist.shape, feature.shape)
                    return False
                feature = np.concatenate([feature, intra_dist], axis = -1)
            else:
                feature = False
        else:       
            if pred_dist_file_cb is not None:
                intra_dist = np.loadtxt(pred_dist_file_cb)
                intra_dist = intra_dist[:,:, np.newaxis]
                feature = intra_dist
            else:
                feature = False
    return feature

def cal_feature_num(reject_fea_file):
    accept_list = []
    with open(reject_fea_file) as f:
        for line in f:
            if line.startswith('#'):
                feature_name = line.strip()
                feature_name = feature_name[0:]
                accept_list.append(feature_name)
    feature_num = 0
    if '# plm' in accept_list:
        feature_num += 441
    if '# rowatt' in accept_list:
        feature_num += 144
    if '# rowatt_diff' in accept_list:
        feature_num += 144
    if '# rowatt_inter' in accept_list:
        feature_num += 144
    if '# pssm' in accept_list:
        feature_num += 40
    if '# ccmpred' in accept_list:
        feature_num += 1
    if '# intradist_cb' in accept_list:
        feature_num += 1
    if '# intradist_hv' in accept_list:
        feature_num += 1
    if '# interdist' in accept_list:
        feature_num += 1
    return feature_num


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.description="Generate feature by lists"
    parser.add_argument("-n", "--name", help="name of outfile", type=str, required=True)
    parser.add_argument("-a", "--a3m_file", help="MSA file end as '.a3m'", type=str, required=True)
    parser.add_argument("-d", "--depth", help="depth of rowatt",type=int, default=256, required=False)
    parser.add_argument("-o", "--outdir", help="output folder", type=str, required=True)
    # parser.add_argument("-db", "--unirefdb", help="unirefdb", type=str, required=True)

    args = parser.parse_args()
    name = args.name
    a3m_file = args.a3m_file
    depth = args.depth
    outdir = args.outdir
    # unirefdb = args.unirefdb
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(a3m_file):
        print(a3m_file, ' not exists!')
        sys.exit(1)
    aln_file = f'{outdir}/{name}.aln'
    os.system("grep -v '^>' %s | sed 's/[a-z]//g' >  %s" % (a3m_file, aln_file))
    fasta_file = f'{outdir}/{name}.fasta'
    fasta = open(aln_file, 'r').readlines()[1].strip('\n')
    open(fasta_file, 'w').write(f'>{name}\n{fasta}\n')
    time_start = time.time()
    pre_load_esm(if_train=False)
    computeplm(name, aln_file, save_ccmpred_path=outdir)
    computerowatt_over1024(name, a3m_file, outdir=outdir, depth=depth)
    computepssm(name, fasta_file, outdir, unirefdb)
    print('{}s'.format(time.time() - time_start))