#!/bin/bash -l

usage() {
        echo ""
        echo "Usage: $0 <OPTIONS>"
        echo "Required Parameters:"
        echo "-n <protein id>               Name of target"
        echo "-p <monomer pdb file list>    Path list of monomer pdb file, could be one file for the homodimer
                              Use \"\" for the list input, i.e. -p \"./T1032A.pdb ./T1032B.pdb\""
        echo "-s <stocihiometry>            Complex stocihiometry, e.g. T1032A:2/T1032B:2 means have T1032A and 
                              T1032B form heterodimer and both of them have one homodimer copy"
        echo "-o <output folder>            Output folder for the project"
        echo "Optional Parameters:"
        echo "-d <dist threshold>           Distance threshold for the GDFold"
        echo ""
        exit 1
}

while getopts ":n:p:s:o:d" i; do
        case "${i}" in
        n)
                name=$OPTARG
        ;;
        p)
                pdb_file_list=$OPTARG
        ;;
        s)
                stocihiometry=$OPTARG
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
if [[ "$name" == " " || "$pdb_file_list" == " " || "$stocihiometry" == " "  || "$output_dir" == " " ]]
then
    usage
fi

if [[ "$dist_thred" == " " ]]
then
    dist_thred=12
fi

workdir=$(dirname $(readlink -f "$0"))
cd $workdir


if [[ ! -x "$workdir/external_tool/ZComplexMSA/" ]]
then
        echo 'Complex MSA geneation tool not exist!'
        exit 1
else
        __conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
        eval "$__conda_setup"
        unset __conda_setup
        conda activate $workdir/external_tool/ZComplexMSA/env/

                python $workdir/external_tool/ZComplexMSA/run_zcomplexmsa.py --option_file $workdir/external_tool/ZComplexMSA/bin/db_option --fasta1 $output_dir/MSA/${pdb_name_list[0]}.fasta --outdir $output_dir/MSA/ --option homodimer
        else
                for pdb_file in ${pdb_file_arr[@]}
                do
                        python lib/pdb_process.py -p $pdb_file -o $output_dir/MSA/ -op get_sequence_from_pdb
                done
                # compare the fasta sequence to make sure is heterodimer
                for pdb_pair in ${pdb_pair_list[@]}
                do
                {        
                        tmp=(`echo $pdb_pair | tr '_' ' '`)
                        python $workdir/external_tool/ZComplexMSA/run_zcomplexmsa.py --option_file $workdir/external_tool/ZComplexMSA/bin/db_option --fasta1 $output_dir/MSA/${pdb_name_list[${tmp[0]}]}.fasta --fasta2 $output_dir/MSA/${pdb_name_list[${tmp[1]}]}.fasta --outdir $output_dir/MSA/ --option heterodimer
                }&
                done
                wait
        fi
        conda deactivate
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
chain_num=${#pdb_name_list[@]}
if [[ "${#pdb_name_list[@]}" -le 2 && "$a3m_file" == "" ]] 
then
        echo "### Start dimer prediction."
        python lib/Model_predict.py -n $name -p $pdb_file_list -a $a3m_file -m $model_option -o $output_dir
        echo "The prediction map at: $output_dir/predmap/"
else
        echo "### Start multimer prediction."
        if [[ "$model_option" =~ 'homo' ]]
        then
                python lib/Model_predict.py -n $name -p $pdb_file_list -a $output_dir/MSA/${pdb_name_list[0]}.a3m -m $model_option -o $output_dir
        else
                for pdb_pair in ${pdb_pair_list[@]}
                do
                        tmp=(`echo $pdb_pair | tr '_' ' '`)
                        sub_pdb_file_list="${pdb_file_arr[${tmp[0]}]} ${pdb_file_arr[${tmp[1]}]}"
                        sub_pdb_pair="${pdb_name_list[${tmp[0]}]}_${pdb_name_list[${tmp[1]}]}"
                        python lib/Model_predict.py -n $sub_pdb_pair -p $sub_pdb_file_list -a $output_dir/MSA/$sub_pdb_pair.a3m -m $model_option -o $output_dir
                done
        fi
        rrfile_list
        python lib/generate_multimer_rr.py -n -r -cn -d -o 
fi
exit 1
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

python external_tool/GDFold/scripts/docking_new_dist.py $name $pdb_file_list $rr_file $fold_outdir $weight_file $dist_thred
echo "Final model at: $output_dir/models/top_5_models/"

