import argparse
import re
import subprocess
import datetime
import json
import os
import sys
import shutil
import time
from multiprocessing import cpu_count
from module.candidate_Util import Logger, file_exist, read_fasta

def parse_args():
    parser = argparse.ArgumentParser(description='Find candidate helitrons')
    parser.add_argument('--out_dir', default="./f_oxy", help='Temporary output directory')
    parser.add_argument('--genome', default="../data/f_oxy.fa", help='Reference genome file')
    parser.add_argument('--threads', type=int, default=104, help='Number of threads')
    parser.add_argument('--edge_threshold',default="20")
    parser.add_argument('--middle_zero_threshold',default="0.4")
    parser.add_argument('--sp',default="oxy")
    #parser.add_argument('--debug', type=int, default=0, help='Debug mode')
    return parser.parse_args()





def main():
    args = parse_args()
    genome = args.genome
    out_dir = args.out_dir
    abs_out_dir = os.path.abspath(out_dir)
    sp = args.sp
    threads = str(args.threads)
    new_genome = f"{out_dir}/genome.fa"
    if not os.path.exists(out_dir):
       os.makedirs(out_dir, exist_ok=True)
    cp_genome_command = f"cp {args.genome} {new_genome}"
    script_path = os.path.dirname(os.path.abspath(__file__))
    device = "cuda:0"
    tools_dir = script_path + "/tools"
    os.system(cp_genome_command)
    
    
    #######FEMA candidate################
    FEMA_out_dir = out_dir + "/FEMA_out"
    if not os.path.exists(FEMA_out_dir):
       os.makedirs(FEMA_out_dir, exist_ok=True)
    

    command1 = f"python module/candidate.py --tmp_output_dir {FEMA_out_dir} --reference {new_genome} --threads {threads}"
    os.system(command1)
    
    
    ##############HLE1 candidate ##############################
    
    
    command2 = f"python module/HLE1_find.py --input_dir {FEMA_out_dir} --head_pattern {script_path}/head.lcvs --tail_pattern {script_path}/tail.lcvs --output {out_dir}/HLE1.fa --threads {threads}"
    os.system(command2)
    
    ##############HLE2 candidate ##############################
    
    
    command3 = f"python module/HLE2_find.py --input_dir {FEMA_out_dir} --RepHel {script_path}/RepHel.hmm --output {out_dir}/HLE2.fa --threads {threads} --genome {new_genome} --primary_dir {script_path}"
    
    os.system(command3)
    
    HLE1_path = f"{out_dir}/HLE1.fa"
    HLE2_path = f"{out_dir}/HLE2.fa"
    HLE_path = f"{out_dir}/HLE_candidate.fa"
    if os.path.exists(HLE1_path) and os.path.exists(HLE2_path):
       command4 = f"cat {HLE1_path} {HLE2_path} > {HLE_path}"
    elif os.path.exists(HLE1_path):
       command4 = f"cat {HLE1_path} > {HLE_path}"
    elif os.path.exists(HLE2_path):
       command4 = f"cat {HLE2_path} > {HLE_path}"
    command4 = f"cat {HLE1_path} {HLE2_path} > {out_dir}/HLE_candidate.fa"
     
    os.system(command4)
    os.remove(HLE1_path)
    os.remove(HLE2_path)
    ######################Intact TE module #################################
    
    
    command5 = f"python module/homo_v2.py --input_file {abs_out_dir}/HLE_candidate.fa --genome {new_genome} --input_dir {abs_out_dir} --MSA_script {script_path}/tools/ready_for_MSA.sh --threads {threads}"
    
    os.system(command5)
    
    command6 = f"python module/Intact.py --device {device} --input_dir {out_dir} --input_MSA_dir {out_dir}/MSA --model_path {script_path}/model_state/{sp}_sort_binary_classifier.pth " 
    
    os.system(command6)
    
    #####################Boundary identify module############################
    
    command7 = f"python module/Window.py --device {device} --model_path {script_path}/model_state/seq_labeler_{sp}.pth --input_dir {out_dir} --edge_threshold {args.edge_threshold} --middle_zero_threshold {args.middle_zero_threshold} --homo_col_threshold 0.5 --homo_region_threshold 0.5 --homo_window_threshold 0.5"    
    os.system(command7)
    
    
    
    #####################Helitron classification module###########################
    command8 = f"python module/Classify.py --input_dir {out_dir}  --model_path {script_path}/model_state/best_combined_model_{sp}.pth --threads {threads} --device {device}"
    os.system(command8)
    
    current_dir = os.path.abspath(out_dir)
    command9 = f"python module/post_filter.py --input_dir {current_dir} --threads {threads} --tools_dir {tools_dir} --cur_dir {script_path}"   
    

    os.system(command9)

    os.remove(out_dir+"/class1_files.txt")
    os.remove(out_dir+"/predict_record.txt")
    ####################remove nest Helitron###########################################
    command10 = f"python module/cluster.py --input_dir {out_dir}"
    os.system(command10)
    command11 = f"python module/remove_nest.py --input_dir {out_dir} --threads {threads} --genome {new_genome}"
    
    
    os.system(command11)
    shutil.rmtree(FEMA_out_dir)
    shutil.rmtree(f"{out_dir}/temp_struc")
    shutil.rmtree(f"{out_dir}/window_log_v2")
    shutil.rmtree("./GenomeDB")
    shutil.rmtree('./BedtoolsTMP')
    for ext in ['.nhr', '.nin', '.nsq','.ndb','.njs','.nog','.nos','.not','.ntf','.nto']:
        temp_file = f"{out_dir}/genome.fa{ext}"
        if os.path.exists(temp_file):
            os.remove(temp_file)
    os.remove(new_genome)
    os.remove(new_genome+".tmp")
    
if __name__ == "__main__":
    main()

