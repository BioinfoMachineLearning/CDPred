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

For heterodimer:


**(1) run MSA generation for heterodimer**

```
conda activate ./env/

python /home/multicom4s_tool/ZComplexMSA/scripts/alignment_generation.py chainA.fasta chainB.fasta output_folder

example:
python /home/multicom4s_tool/ZComplexMSA/scripts/alignment_generation.py /home/multicom4s_tool/ZComplexMSA/test/hetero/1AWCA.fasta  /home/multicom4s_tool/ZComplexMSA/test/hetero/1AWCB.fasta /home/multicom4s_tool/ZComplexMSA/test/1AWCA_1AWCB/
```

**(2) run MSA generation for homodimer**

```
sh /home/multicom4s_tool/ZComplexMSA/scripts/hhblits.sh sequence_name fasta_file output_folder database 50

example:
sh /home/multicom4s_tool/ZComplexMSA/scripts/hhblits.sh 2FDOA /home/multicom4s_tool/ZComplexMSA/test/homo/2FDOA.fasta /home/multicom4s_tool/ZComplexMSA/test/2FDOA/ /home/multicom4s_tool/ZComplexMSA/database/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt 50
```

