![header](image/CDPred1.gif)

# CDPred
## Function of CDPred
CDPred is a deep transformer tool for predicting interchain residue-residue distances of protein dimers. For a homodimer consiting of two identical chains, it takes the tertiary structure and multiple sequence alignment (MSA) of a chain as input to predict the residue-residue distances between the two chains. For a heterodimer consisting of two different chains, it takes the tertiary structures and MSA of the two chains as input to predict residue-residue distances between the two chains. If no MSA is available for a dimer, users can use a custom script in this package to generate a MSA as input. The input is converted into numerical features that are used by 2D attention-based transformer networks to predict residue-residue distances between two chains in a dimer. 

# Contents
- [System Requirements](#system-requirements)
- [Installation guide](#installation-guide)
- [Running CDPred](#running-cdpred)
- [Demo](#demo)
- [Output files](#output-files)
- [Evaluation](#evaluation)
- [License](#license)

## System Requirements
### OS Requirements
This package is developed on Linux. The package has been tested on the following two Linux systems:  
Linux: Ubuntu 16.04  
Linux: CentOS Linux release 7.9.2009  
### Python Dependencies
The system is developed and tested under Python 3.6.x. The main dependent packages and their versions are as follows. For more detail, please check the requirment.txt file.
```
fair-esm==0.3.1
Keras==2.1.6
matplotlib==3.3.4
numpy==1.16.2
tensorflow==1.9.0
```

## Installation guide
**(1) Download CDPred package (a short path for the package is recommended)**

```
git clone https://github.com/BioinfoMachineLearning/CDPred.git

cd CDPred
```

**(2) Install and activate Python 3.6.x environment on Linux (required)**

The installation of Python 3.6.x may be different for different Linux systems. 

```
mkdir env
python3.6 -m venv env/CDPred_virenv
source env/CDPred_virenv/bin/activate
pip install --upgrade pip
pip install -r requirments.txt
```
**(3) Download Uniref90**
Download the Uniref90_01_2020 database for PSSM generation
```
wget http://sysbio.rnet.missouri.edu/CDPred_db/uniref90_01_2020.tar.gz
tar -zxvf uniref90_01_2020.tar.gz
```
Modify the Uniref90 path in script ./lib/constants.py as /Download_Path/uniref90_01_2020/uniref90
The installation and configuration of the virtual environment lasts about 10 minutes (minor difference on different devices).  
And the the Uniref90 database download will take about 40 minutes to 70 minutes, dependent on your network speed.

## Running CDPred
### Parameter Description of the CDPred prediction script.

Command: `CDPred_Installation_Path/lib/Model_predict.py -n [name] -p [pdb_file_list] -a [a3m_file] -m [model_option] -o [out_path]`
Parameters:
	`-n` – The name of the protein complex, can be protein ID or custom name.
	`-p` – The predicted monomer tertiary structure file or files with ".pdb" suffix. For homodimer inter-chain distance prediction, one 
		predicted monomer structure file is enough. For heterodimer inter-chain distance prediction, both chains' predicted 
		monomer structure files are required and needed to seperate by one space (Check the detail in [Demo](#demo) section).
	`-a` – Multiple sequence alignment (MSA) file in ".a3m" format. You can use your own or any third-party tool to generate MSA file, or you can follow the 
		instruction in [ZComplexMSA](https://github.com/BioinfoMachineLearning/CDPred/tree/main/external_tool/ZComplexMSA) to 
		install our custom MSA generation tool (Require large disk space and long time for dataset downloading).
	`-m` – Model option for different type prediction. Use "homodimer" for homodimer inter-chain distance prediction. 
		Use "heterodimer" for heterodimer inter-chain distance prediction
	`-o` – The custom output folder. It will be automaticly created if not exist.

## Demo
### Examples to make predictions on prepared input data

Demo1: Run CDPred on a homodimer target.

```
python lib/Model_predict.py -n T1084A_T1084B -p ./example/T1084A_T1084B.pdb -a ./example/T1084A_T1084B.a3m -m homodimer -o ./output/T1084A_T1084B/
```
The location of the pre-generated output files is ./example/expection_output/T1084A_T1084B/, and the location of the output file generated by your run is ./output/T1084A_T1084B/. The whole prediction process will last about 5 minutis

Demo2: Run CDPred on a heterodimer target.

```
python lib/Model_predict.py -n H1017A_H1017B -p ./example/H1017A.pdb ./example/H1017B.pdb -a ./example/H1017A_H1017B.a3m -m heterodimer -o ./output/H1017A_H1017B/
```
The location of the pre-generated output files is ./example/expection_output/H1017A_H1017B/, and the location of the output file generated by your run is ./output/H1017A_H1017B/. The whole prediction process will last about 5 minutis


## Output files
The outputs will be saved in directory provided via the`-o` flag of `Model_predict.py` . The outputs include multiple sequence alignment files, feature files, and prediction inter-chain distance/contact maps.
The `--output_dir` directory will have the following structure, "name" is provided via the `-n` flag of `Model_predict.py`:

```
<custom_output_name>/
	feature/
		"name"_pssm.txt
		"name".npy
		"name".mat
		"name".fasta
		"name".dist
		"name".aln
		"name".a3m
    predmap/
        "name"_dist.rr
        "name"_con.rr
        "name".htxt
        "name".dist
```

The contents of each output file are as follows:
*   `name_pssm.txt` – Position-specific scoring matrix (PSSM) feature.
*   `name.npy` – Row attention map generated by [ESM](https://github.com/facebookresearch/esm) and used as one main co-evolutionary feture.
*   `name.mat` – Co-evolutinary score matrix generate by [CCMpred](https://github.com/soedinglab/CCMpred).
*   `name.fasta` – Fasta sequence file of the input homodimer/heterodimer.
*   `name.dist` – A combination distance map in shape LxL (L:the length of dimer) of tow monomer's carbon alpha distance map that extract from input prediction monomer structure.
*   `name.aln` – Multiple sequence alignment in 'aln' .
*   `name.a3m` – Multiple sequence alignment.
*   `name_dist.rr` – Residue-Residue distance prediction in format `i, j, dist`.
    *   `i` and `j` indicate specifying pairs of residues.
    *   `dist` indicates the prediction Euclidean heavy-atom distance between `i` and `j`.
*   `name_con.rr` – Residue-Residue contact prediction in format `i, j, 0, 8, prob`.
    *   `i` and `j` indicate specifying pairs of residues.
    *   `0` and `8` indicate the distance limits defining a contact. Here a pair of residues is defined to be in contact when the minimum distance of heavy atoms is less then 8 Angstroms. 
    *	`prob` indicate the prediction contact probability under above distance limits.
*   `name.htxt` – Prediction inter-chain contact map.
*   `name.dist` – Prediction inter-chain distance map.

## Evaluation on a Small Dataset
Script File: `.lib/distmap_evaluate.py -p [pred_map] -t [true_map] -f1 [fasta_file1] -f2 [fasta_file2]`
Parameters:
	`-p` – The prediction contact map with '.htxt' suffix.
	`-t` – The nativate distance/contact map with '.htxt' suffix.
	`-f1` – The fasta sequence file of chain 1 of dimer.
	`-f2` – The fasta sequence file of chain 2 of dimer.

Demo1: Evaluate the homodimer target.

```
python ./lib/distmap_evaluate.py -p ./example/expection_output/T1084A_T1084B/predmap/T1084A_T1084B.htxt -t ./example/ground_truth/T1084A_T1084B.htxt -f1 ./example/ground_truth/T1084A.fasta -f2 ./example/ground_truth/T1084B.fasta
```
Expection output of Demo1:
```
NAME            LEN_A LEN_B TOP5       TOP10      TOPL/10    TOPL/5     TOPL/2     TOPL      
T1084A_T1084B   71    71    100.0000   100.0000   100.0000   100.0000   94.2857    91.5493 
```

Demo2: Evaluate the heterodimer target.
```
python ./lib/distmap_evaluate.py -p ./example/expection_output/H1017A_H1017B/predmap/H1017A_H1017B.htxt -t ./example/ground_truth/H1017A_H1017B.htxt -f1 ./example/ground_truth/H1017A.fasta -f2 ./example/ground_truth/H1017B.fasta
```
Expection output of Demo2:
```
NAME            LEN_A LEN_B TOP5       TOP10      TOPL/10    TOPL/5     TOPL/2     TOPL      
H1017A_H1017B   110   125   60.0000    60.0000    54.5455    50.0000    41.8182    36.3636 
```

## License
This project is covered under the **MIT License**.
