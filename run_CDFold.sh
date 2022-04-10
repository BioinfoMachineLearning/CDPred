#!/bin/bash -l

usage() {
        echo ""
        echo "Usage: $0 <OPTIONS>"
        echo "Required Parameters:"
        echo "-n <protein id>               Name of target i.e. T1084A_T1084B"
        echo "-p <monomer pdb file list>    Path list of monomer pdb file, could be one file for the homodimer
                              Use \"\" for the list input, i.e. -p \"./fileA ./fileB\""
        echo "-m <model option>             Model option for the CDPred, e.g. homodimer, heterodimer"
        echo "-o <output folder>            Output folder for the project"
        echo "Optional Parameters:"
        echo "-a <a3m file>                 Multiple sequence alignment file end with .a3m"
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
if [[ "$name" == "" || "$pdb_file_list" == "" || "$model_option" == ""  || "$output_dir" == "" ]]
then
    usage
fi

if [[ "$dist_thred" == "" ]]
then
    dist_thred=12
fi

workdir=$(dirname $(readlink -f "$0"))
cd $workdir

pdb_file_arr=(`echo $pdb_file_list | tr ' ' ' '`)
count=0
for pdb_file in ${pdb_file_arr[@]}
do
        pdb_name_list[count]=$(basename $pdb_file .pdb)
        let count++
done

if [[ "$a3m_file" == "" ]]
then
        if [[ ! -x "$workdir/external_tool/ZComplexMSA/" ]]
        then
                echo 'Complex MSA geneation tool not exist!'
                exit 1
        else
                __conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
                eval "$__conda_setup"
                unset __conda_setup
                conda activate $workdir/external_tool/ZComplexMSA/env/
                if [[ "$model_option" =~ 'homo' ]]
                then
                        python lib/pdb_process.py -p ${pdb_file_arr[0]} -o $output_dir/MSA/ -op get_sequence_from_pdb
                        python $workdir/external_tool/ZComplexMSA/run_zcomplexmsa.py --option_file $workdir/external_tool/ZComplexMSA/bin/db_option --fasta1 $output_dir/MSA/${pdb_name_list[0]}.fasta --outdir $output_dir/MSA/ --option homodimer
                else
                        for pdb_file in ${pdb_file_arr[@]}
                        do
                                python lib/pdb_process.py -p $pdb_file -o $output_dir/MSA/ -op get_sequence_from_pdb
                        done

                        python $workdir/external_tool/ZComplexMSA/run_zcomplexmsa.py --option_file $workdir/external_tool/ZComplexMSA/bin/db_option --fasta1 $output_dir/MSA/${pdb_name_list[0]}.fasta --fasta2 $output_dir/MSA/${pdb_name_list[1]}.fasta --outdir $output_dir/MSA/ --option heterodimer

                fi
                conda deactivate
        fi
fi

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
if [[ "$a3m_file" == "" ]] 
then
        if [[ "$model_option" =~ 'homo' ]] 
        then
                python lib/Model_predict.py -n $name -p $pdb_file_list -a $output_dir/MSA/${pdb_name_list[0]}.a3m -m $model_option -o $output_dir
        else
                tmp_name="${pdb_name_list[0]}"_"${pdb_name_list[1]}"
                python lib/Model_predict.py -n $name -p $pdb_file_list -a $output_dir/MSA/$tmp_name.a3m -m $model_option -o $output_dir
        fi
else
        python lib/Model_predict.py -n $name -p $pdb_file_list -a $a3m_file -m $model_option -o $output_dir
fi
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

python ./external_tool/GDFold/scripts/docking_new_dist.py $name $pdb_file_list $rr_file $fold_outdir $weight_file $dist_thred
echo "Final model at: $output_dir/models/top_5_models/"
