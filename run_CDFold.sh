#!/bin/bash

usage() {
        echo ""
        echo "Usage: $0 <OPTIONS>"
        echo "Required Parameters:"
        echo "-n <protein id>               Name of target"
        echo "-p <monomer pdb file list>    Path list of monomer pdb file, could be one file for the homodimer
                              Use \"\" for the list input, i.e. -p \"./fileA ./fileB\""
        echo "-a <a3m file>                 Multiple sequence alignment file end with .a3m"
        echo "-m <model option>             Model option for the CDPred, e.g. homodimer, heterodimer"
        echo "-o <output folder>            Output folder for the project"
        echo "Optional Parameters:"
        echo "-d <dist threshold>           Distance threshold for the GDFold"
        echo ""
        exit 1
}

while getopts ":n:p:a:m:o:d" i; do
        case "${i}" in
        n)
                name=$OPTARG
        ;;
        p)
                pdb_file_list=$OPTARG
        ;;
        a)
                a3m_file=$OPTARG
        ;;
        m)
                model_option=$OPTARG
        ;;
        o)
                output_dir=$OPTARG
        ;;
        d)
                dist_thred=$OPTARG
        ;;
        esac
done

# Parse input and set defaults
if [[ "$name" == "" || "$pdb_file_list" == "" || "$a3m_file" == "" || "$model_option" == ""  || "$output_dir" == "" ]]
then
    usage
fi

if [[ "$dist_thred" == "" ]]
then
    dist_thred=12
fi

workdir=$(dirname $(readlink -f "$0"))

# Activate virtual environment
virenv_dir=$workdir/env/CDPred_virenv/
if [[ ! -x "$virenv_dir" ]]
then
	echo "$virenv_dir not exit, please create the virtual environment follow README!"
	exit 1
else
	echo "### Activet the virtual environment"
	source $virenv_dir/bin/activate
fi

# Run CDPred
echo "### Runing CDPred"
python lib/Model_predict.py -n $name -p $pdb_file_list -a $a3m_file -m $model_option -o $output_dir
echo "The prediction map at: $output_dir/predmap/"

# Check GDFold existence
echo "### Runing GDFold"
if [[ ! -x "$workdir/external_tool/GDFold/" ]]
then
	echo "GDFold tool not exit"
	exit 1
fi

if [[ ! "$pdb_file_list" =~ \  ]]
then
	pdb_file_list="$pdb_file_list $pdb_file_list"
fi

rr_dir=$output_dir/predmap/
if [[ ! -x "$rr_dir" ]]
then
	echo "Contrain file not exit!"
	exit 1
else
	rr_file=$rr_dir/$name"_dist.rr"
fi
weight_file=$workdir/external_tool/GDFold/scripts/talaris2013.wts
fold_outdir=$output_dir/models/

python external_tool/GDFold/scripts/run_dock.py dimer dist $name $pdb_file_list $rr_file $fold_outdir $weight_file $dist_thred
echo "Final model at: $output_dirmodels/top_5_models/"