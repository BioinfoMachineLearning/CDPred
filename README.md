![header](image/CDPred1.gif)

# CDPred
## Function of CDPred
CDPred is a deep transformer tool for predicting interchain residue-residue distances of protein dimers. For a homodimer consiting of two identical units, it takes the tertiary structure of a unit of the homodimer and the multiple sequence alignment (MSA) of the unit as input to predict the residue-residue distances between the two identical units of the homodimer. For a heterodimer consisting of two different units, it takes the tertiary structures of the two units of the heterodimer and the MSA of the two units as input to predict residue-residue distances between the two units. If no MSA is available for dimer, users can use a script in this package to generate a MSA as input. The input is converted into numerical features that are used by 2D attention-based transformer networks to predict residue-residue distances between two units in a dimer. 

## System Requirements
### OS Requirements
This package is supported for Linux. The package has been tested on the following systems:  
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
### Parameter Description of CDPred prediction script.

Script File: `.lib/Model_predict.py -n [name] -p [pdb_file_list] -a [a3m_file] -m [model_option] -o [out_path]`
Parameters:
	`-n` – The name of the protein complex, can be protein ID or custom name.
	`-p` – The prediction monomer structure file or files with ".pdb" suffix. For homodimer inter-chain distance prediction one 
		prediction monomer structure file is enough. For heterodimer inter-chain distance prediction, both chain's prediction 
		monomer structure files are required and need to seperate by one space (Check the detail in [Demo](##Demo) section).
	`-a` – Multiple sequence file in ".a3m" format. You can use any third-party tools to generate MSA file, or you can follow the 
		instruction in [ZComplexMSA](https://github.com/BioinfoMachineLearning/CDPred/tree/main/external_tool/ZComplexMSA) to 
		install our custom MSA generation tool (Require large disk space and long time for dataset downloading).
	`-m` – Model option for different type prediction. Use "homodimer" for homodimer inter-chain distance prediction. 
		Use "heterodimer" for heterodimer inter-chain distance prediction
	`-o` – The custom output folder. Will automatic create if not exist.

## Demo
### Instructions to run on example data

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

## License

This project is covered under the **MIT License**.