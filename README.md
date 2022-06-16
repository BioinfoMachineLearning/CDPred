# CDPred
## Function of CDPred
CDPred is a deep transformer tool for predicting interchain residue-residue distances of protein dimers. For a homodimer consiting of two identical units, it takes the tertiary structure of a unit of the homodimer and the multiple sequence alignment (MSA) of the unit as input to predict the residue-residue distances between the two identical units of the homodimer. For a heterodimer consisting of two different units, it takes the tertiary structures of the two units of the heterodimer and the MSA of the two units as input to predict residue-residue distances between the two units. If no MSA is available for dimer, users can use a script in this package to generate a MSA as input. The input is converted into numerical features that are used by 2D attention-based transformer networks to predict residue-residue distances between two units in a dimer. 

## Installation
**(1) Download CDPred package (a short path for the package is recommended)**

```
git clone https://github.com/BioinfoMachineLearning/CDPred.git

cd CDPred
```

**(2) Install and activate Python 3.6.x environment on Linux (required)**

The installation of Python 3.6.x may be different for different Linux systems. 

**Note**: The system is developed and tested under Python 3.6.x. 
```
mkdir env
python3 -m venv env/CDPred_virenv
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

## Run CDPred and Example 
Case 1: Use CDPred to predict inter-protein homodimeric distance

```
python lib/Model_predict.py -n ProteinID -p pdb_file_list -a MSA_file(end in .a3m) -m homodimer -o output_path 
```
Example (Lasts about 5 minutes):

```
python lib/Model_predict.py -n T1084A_T1084B -p ./example/T1084A_T1084B.pdb -a ./example/T1084A_T1084B.a3m -m homodimer -o ./output/T1084A_T1084B/
```

Case 2: Use CDPred to predict inter-protein heterodimers distance

```
python lib/Model_predict.py -n ProteinID -p pdb_file_list -a MSA_file(end in .a3m) -m homodimer -o output_path 
```
Example (Lasts about 5 minutes):

```
python lib/Model_predict.py -n H1017A_H1017B -p ./example/H1017A.pdb ./example/H1017B.pdb -a ./example/H1017A_H1017B.a3m -m heterodimer -o ./output/H1017A_H1017B/
```

## Output files
For both homodimer and heterodimer cases, the final prediction results will be stored in folder ./output/ProteinID/predmap/.
The inter-protein prediction distance map end in ProteinID.dist, the inter-protein prediction contact map end in ProteinID.htxt.

