# Complex structure MSA generation pipelines

**(1) Create virtual environment**

```
conda create -p ./env/ -c conda-forge -c bioconda hhsuite python==3.8
conda activate ./env/
conda install -y -c bioconda hmmer==3.3.2 hhsuite==3.3.0
pip install -r requirment.txt
```

**(2) Download the necessary database**

For homodimer:
Download the Big Fantastic Database(BFD)([here](https://bfd.mmseqs.com/))

For heterodimer:
```
wget http://sysbio.rnet.missouri.edu/CDPred_db/ComplexDB.tar.gz
tar -zxvf ComplexDB.tar.gz
```
Modify the database option file at ./bin/db_option
```
bfd_database = /Your_Download_Path/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt
uniref90_fasta = /Your_Download_Path/uniref90/uniref90.fasta
uniprot2pdb_dir = /Your_Download_Path/uniprot2pdb
uniprot2pdb_mapping_file = /Your_Download_Path/uniprot2pdb/uniprot2pdb.map
dimers_list = /Your_Download_Path/uniprot2pd/dimers_cm.list
```

**(3) run MSA generation for heterodimer**

```
python run_zcomplexmsa.py --option_file [database_option_file] --fasta1 [fasta_file1] --fasta2 [fasta_file2] --outdir [output_folder] --option [dimer_option]

```
Example:
```
python run_zcomplexmsa.py --option_file ./bin/db_option --fasta1 ./test/hetero/1AWCA.fasta --fasta2 ./test/hetero/1AWCB.fasta --outdir ./test/hetero --option heterodimer
```

**(4) run MSA generation for homodimer**

```
python run_zcomplexmsa.py --option_file [database_option_file] --fasta1 [fasta_file1] --outdir [output_folder] --option [dimer_option]
```
Example: 

```
python run_zcomplexmsa.py --option_file ./bin/db_option --fasta1 ./test/homo/2FDOA.fasta  --outdir ./test/homo --option homodimer
```
