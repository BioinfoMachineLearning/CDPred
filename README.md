# CDPred
Deep transformer for predicting interchain residue-residue distances of protein complexes

**(1) Download CDPred package (a short path for the package is recommended)**

```
git clone git@github.com:BioinfoMachineLearning/CDPred.git

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

**(3) Run CDPred (required)**
<h4>Case 1: Use CDPred to predict inter-protein homodimeric distance</h4>

```
python lib/Model_predict.py -n ProteinID -p pdb_file_list -a MSA_file(end in .a3m) -m homodimer -o output_path 
```
Example

```
python lib/Model_predict.py -n T1084A_T1084B -p ./example/T1084A_T1084B.pdb -a ./example/T1084A_T1084B.a3m -m homodimer -o ./output/T1084A_T1084B/
```

<h4>Case 2: Use CDPred to predict inter-protein heterodimers distance</h4>

```
python lib/Model_predict.py -n ProteinID -p pdb_file_list -a MSA_file(end in .a3m) -m homodimer -o output_path 
```
Example

```
python lib/Model_predict.py -n H1017A_H1017B -p ./example/H1017A.pdb ./example/H1017B.pdb -a ./example/H1017A_H1017B.a3m -m heterodimer -o ./output/H1017A_H1017B/
```
For both cases, the final prediction results in folder ./output/ProteinID/predmap/
The inter-protein prediction distance map end in ProteinID.dist, the inter-protein prediction contact map end in ProteinID.htxt.
