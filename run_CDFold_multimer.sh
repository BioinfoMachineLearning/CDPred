#!/bin/bash -l

usage() {
        echo ""
        echo "Usage: $0 <OPTIONS>"
        echo "Required Parameters:"
        echo "-n <protein id>               Name of target"
        echo "-p <monomer pdb file list>    Path list of monomer pdb file, could be one file for the homodimer Use \"\" for the list input, i.e. -p \"./T1032A.pdb ./T1032B.pdb\""
        echo "-s <stocihiometry>            Complex stocihiometry, e.g. T1032A:2/T1032B:2 means have T1032A and T1032B form heterodimer and both of them have one homodimer copy"
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
if [[ "$name" == "" || "$pdb_file_list" == "" || "$stocihiometry" == ""  || "$output_dir" == "" ]]
then
    usage
fi

if [[ "$dist_thred" == "" ]]
then
    dist_thred=12
fi

workdir=$(dirname $(readlink -f "$0"))
cd $workdir

paras=`python ./lib/multimer_preprocess.py  -p $pdb_file_list -s $stocihiometry -o $output_dir/PrePro/`
paras_list=(`echo $paras | tr ' ' ' '`)
homomeric_list=${paras_list[0]}
heteromeric_pairs=${paras_list[1]}
all_inter_paris=${paras_list[2]}
new_pdb_file_list=${paras_list[3]}
echo "homomeric_list   : "$homomeric_list
echo "heteromeric_pairs: "$heteromeric_pairs
echo "all_inter_paris  : "$all_inter_paris
echo "new_pdb_file_list: "$new_pdb_file_list

if [[ ! -x "$workdir/external_tool/ZComplexMSA/" ]]
then
        echo 'Complex MSA geneation tool not exist!'
        exit 1
else
        mkdir "$output_dir"/MSA/
        __conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
        eval "$__conda_setup"
        unset __conda_setup
        conda activate $workdir/external_tool/ZComplexMSA/env/
        tmp_list=(`echo $homomeric_list | tr '|' ' '`)
        for tmp_id in ${tmp_list[@]}
        do
        {
                if [[ ! -f $output_dir/MSA/$tmp_id.a3m ]]
                then
                        python $workdir/external_tool/ZComplexMSA/run_zcomplexmsa.py \
                        --option_file $workdir/external_tool/ZComplexMSA/bin/db_option \
                        --fasta1 $output_dir/PrePro/$tmp_id.fasta \
                        --outdir $output_dir/MSA/ \
                        --option homodimer
                fi

        }&
        done
        tmp_list=(`echo $heteromeric_pairs | tr '|' ' '`)
        for tmp_pair in ${tmp_list[@]}
        do
        {       
                if [[ ! -f $output_dir/MSA/$tmp_pair.a3m ]]
                then 
                        tmp_id=(`echo $tmp_pair | tr '_' ' '`)
                        python $workdir/external_tool/ZComplexMSA/run_zcomplexmsa.py \
                        --option_file $workdir/external_tool/ZComplexMSA/bin/db_option \
                        --fasta1 $output_dir/PrePro/${tmp_id[0]}.fasta \
                        --fasta2 $output_dir/PrePro/${tmp_id[1]}.fasta \
                        --outdir $output_dir/MSA/ \
                        --option heterodimer
                fi
        }&
        done
        wait
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
tmp_list=(`echo $homomeric_list | tr '|' ' '`)
for tmp_id in ${tmp_list[@]}
do
{
        if [[ ! -f $output_dir/predmap/"$tmp_id"_dist.rr ]]
        then 
                tmp_pdb_file_list="$output_dir/PrePro/$tmp_id.pdb"
                echo $tmp_pdb_file_list
                python ./lib/Model_predict.py -n $tmp_id -p $tmp_pdb_file_list -a $output_dir/MSA/$tmp_id.a3m -m homodimer -o $output_dir/
        fi
}&
done
tmp_list=(`echo $heteromeric_pairs | tr '|' ' '`)
for tmp_pair in ${tmp_list[@]}
do
{ 
        if [[ ! -f $output_dir/predmap/"$tmp_pair"_dist.rr ]]
        then 
                tmp_id=(`echo $tmp_pair | tr '_' ' '`)  
                tmp_pdb_file_list="$output_dir/PrePro/${tmp_id[0]}.pdb $output_dir/PrePro/${tmp_id[1]}.pdb"   
                echo $tmp_pdb_file_list  
                python ./lib/Model_predict.py -n $tmp_pair -p $tmp_pdb_file_list -a $output_dir/MSA/$tmp_pair.a3m -m heterodimer -o $output_dir/
        fi
}&
done
wait
python ./lib/generate_multimer_rr.py -n $name -r $output_dir/predmap/ -i $all_inter_paris -d $dist_thred -o $output_dir/predmap

# Check GDFold existence
echo "### Runing GDFold"
if [[ ! -x "$workdir/external_tool/GDFold/" ]]
then
	echo "GDFold tool not exit"
	exit 1
fi

rr_dir=$output_dir/predmap/
if [[ ! -x "$rr_dir" ]]
then
	echo "Contrain file not exit!"
	exit 1
else
	rr_file=$rr_dir/$name"_dist.rr"
fi


pdb_file_arr=(`echo $new_pdb_file_list | sed 's/|/ /g'`)
pdb_file_list=$(echo "${pdb_file_arr[*]}")

echo "New pdb file list:"$pdb_file_list
echo "Chain number:"${#pdb_file_arr[@]}

weight_file=$workdir/external_tool/GDFold/scripts/talaris2013.wts
fold_outdir=$output_dir/models/

python external_tool/GDFold/scripts/docking_gd_parallel_multi_dist.py $name ${#pdb_file_arr[@]} $pdb_file_list $rr_file $fold_outdir $weight_file $dist_thred
echo "Final model at: $output_dir/models/"$name"_GD.pdb"

