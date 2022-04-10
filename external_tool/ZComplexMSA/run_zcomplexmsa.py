import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
from lib.tool import hhblits
from lib.tool import jackhmmer
from lib.common.util import is_dir, is_file, read_option_file, makedir_if_not_exists
from lib.complex_alignment_generation.species_interact import Species_interact
from lib.monomer_alignment_generation.alignment import *
from lib.complex_alignment_generation.pipeline import write_concatenated_alignment
import pathlib
GLOABL_PATH = os.path.split(os.path.realpath(__file__))[0]

def run_hhblits(inparams):

    fasta, outdir, hhblits_binary, database = inparams

    hhblits_runner = hhblits.HHBlits(binary_path=hhblits_binary, databases=[database])

    outfile = outdir + '/' + pathlib.Path(fasta).stem + '.a3m'

    return hhblits_runner.query(fasta, outfile)


def run_jackhmmer(inparams):

    fasta, outdir, jackhmmer_binary, database = inparams

    jackhmmer_runner = jackhmmer.Jackhmmer(binary_path=jackhmmer_binary, database_path=database)

    outfile = outdir + '/' + pathlib.Path(fasta).stem + '.sto'

    return jackhmmer_runner.query(fasta, outfile)


if __name__ == '__main__':
    print(GLOABL_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument('--option_file', type=is_file, required=True)
    parser.add_argument('--fasta1', help="fasta file", type=is_file, required=True)
    parser.add_argument('--fasta2', help="fasta file, only need if is heterodimer", type=is_file, required=False, default=None)
    parser.add_argument('--hhblits', type=is_file, required=False, default=f'{GLOABL_PATH}/env/bin/hhblits')
    parser.add_argument('--jackhmmer', type=is_file, required=False, default=f'{GLOABL_PATH}/env/bin/jackhmmer')
    parser.add_argument('--outdir', type=is_dir, required=True)
    parser.add_argument('--option', help="option to select the type: i.e. homodimer, heterodimer", type=str, required=True)

    args = parser.parse_args()

    params = read_option_file(os.path.abspath(args.option_file))

    if args.option == 'homodimer':
        # test hhblits
        outdir = f'{os.path.abspath(args.outdir)}/'
        makedir_if_not_exists(outdir)
        # for homodimer
        inparams = [os.path.abspath(args.fasta1), outdir, args.hhblits, params['bfd_database']]
        run_hhblits(inparams)

    # for heterodimer
    elif args.option == 'heterodimer':
        if args.fasta2 == None:
            print('Heterodimer option need two fasta file input. Please check it')
            sys.exit(1)
        outdir = f'{os.path.abspath(args.outdir)}/'
        makedir_if_not_exists(outdir)

        process_list = []
        process_list.append([os.path.abspath(args.fasta1), outdir, args.jackhmmer, params['uniref90_fasta']])
        process_list.append([os.path.abspath(args.fasta2), outdir, args.jackhmmer, params['uniref90_fasta']])
        pool = Pool(processes=2)
        results = pool.map(run_jackhmmer, process_list)
        pool.close()
        pool.join()

        with open(results[0]['sto']) as f:
            aln_1 = Alignment.from_file(f, format="stockholm")

        with open(results[1]['sto']) as f:
            aln_2 = Alignment.from_file(f, format="stockholm")

        pair_ids = Species_interact.get_interactions(aln_1, aln_2)

        target_header, sequences_full, sequences_monomer_1, sequences_monomer_2, pair_ids =  \
            write_concatenated_alignment(pair_ids, aln_1, aln_2)

        # save the alignment files
        aln_1_name = aln_1.main_id.split('.')[0] 
        aln_2_name = aln_2.main_id.split('.')[0]
        print(aln_1_name)
        print(aln_2_name)
        mon_alignment_file_1 = f"{outdir}/{aln_1_name}_monomer_1.a3m"
        with open(mon_alignment_file_1, "w") as of:
            write_a3m(sequences_monomer_1, of)

        mon_alignment_file_2 = f"{outdir}/{aln_2_name}_monomer_2.a3m"
        with open(mon_alignment_file_2, "w") as of:
            write_a3m(sequences_monomer_2, of)

        pair_ids.to_csv(f"{outdir}/{aln_1_name}_{aln_2_name}_interact.csv", index=False)
        print(pair_ids)

        complex_ailgnment_file = f"{outdir}/{aln_1_name}_{aln_2_name}.a3m"
        with open(complex_ailgnment_file, "w") as of:
            write_a3m(sequences_full, of)

