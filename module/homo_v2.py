import os
import re
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy_Util import read_fasta, store_fasta, getReverseSequence,get_full_length_copies
import subprocess

def build_blast_database(fasta_path, db_path):
    print(f"\nmake db: {db_path}")
    try:
        subprocess.run([
            'makeblastdb',
            '-in', fasta_path,
            '-dbtype', 'nucl',
            '-parse_seqids',
            '-out', db_path
        ], check=True)
        print("succees make db")
    except subprocess.CalledProcessError as e:
        print(f"fail make db: {e}")
        raise

def process_TE_alignments(genome_dir, TE_file, output_dir, subset_script_path, threads=4, batch_size=10):
    os.makedirs(output_dir, exist_ok=True)
    names, contigs = read_fasta(TE_file)
    build_blast_database(genome_dir,genome_dir)
    batch_id = 0
    split_files = []
    temp_dir = os.path.join(output_dir, "batch_process")
    os.makedirs(temp_dir, exist_ok=True)
    cur_contigs = {}
    for i, name in enumerate(names):
        cur_contigs[name] = contigs[name]
        if len(cur_contigs) == batch_size:
            batch_file = os.path.join(temp_dir, f"batch_{batch_id}.fa")
            store_fasta(cur_contigs, batch_file)
            split_files.append(batch_file)
            cur_contigs = {}
            batch_id += 1
    if cur_contigs:
        batch_file = os.path.join(temp_dir, f"batch_{batch_id}.fa")
        store_fasta(cur_contigs, batch_file)
        split_files.append(batch_file)
    ref_contigs = {}
    file_path = genome_dir
    cur_names, cur_contigs = read_fasta(file_path)
    ref_contigs.update(cur_contigs)
    ex = ProcessPoolExecutor(threads)
    batch_member_files = []

    future_to_file = {}
    for batch_file in split_files:
        future = ex.submit(get_full_length_copies, batch_file, genome_dir, False)
        future_to_file[future] = batch_file

    for future in as_completed(future_to_file):
        all_copies = future.result()
        for query_name in all_copies:
            copies = all_copies[query_name]
            member_contigs = {}
            for copy in copies:
                #print(copy)
                ref_name, start, end, _, direct = copy
                start = int(start)
                end = int(end)
                flanking_len = 500 
                if start - flanking_len < 0 or end + flanking_len > len(ref_contigs[ref_name]):
                    continue
                seq = ref_contigs[ref_name][start-50: end+50]
                if direct == '-':
                    seq = getReverseSequence(seq)
                member_name = f"{ref_name}:{start}-{end}({direct})"
                member_contigs[member_name] = seq

            # 保存成员文件
            if member_contigs:
                safe_name = re.sub(r'[<>:"/\\|?*]', '_', query_name)
                member_file = os.path.join(output_dir, f"{safe_name}_members.fa")
                store_fasta(member_contigs, member_file)
                batch_member_files.append((query_name, contigs[query_name], member_file))

    ex = ProcessPoolExecutor(threads)
    total_file = len(batch_member_files)
    
    for index,batch in enumerate(batch_member_files):
        ex.submit(run_mafft_only, batch, output_dir, subset_script_path,total_file,index)

    ex.shutdown(wait=True)


def run_mafft_only(batch_member_file, output_dir, subset_script_path,total_file,index):
    
    print(f"\rMafft sequence file {index+1}/{total_file}", end="")
    query_name, query_seq, member_file = batch_member_file

    # 处理成员过多的情况（抽样）
    member_names, member_contigs = read_fasta(member_file)
    if len(member_names) > 100:
        temp_dir = os.path.dirname(member_file)

        cmd = f"cd {temp_dir} && sh {subset_script_path} {member_file} 100 100 > /dev/null 2>&1"
        os.system(cmd)
        member_file += ".rdmSubset.fa"

    if not os.path.exists(member_file):
        print("skip")
        return

    # 运行MAFFT
    align_file = member_file + ".aln.fa"
    cmd = f"mafft --preservecase --quiet --thread 1 {member_file} > {align_file}"
    os.system(cmd)


def parse_args():
    parser = argparse.ArgumentParser(description='Find candidate helitrons')
    parser.add_argument('--genome', default="sample.fa", help='Reference genome file')
    parser.add_argument('--threads', type=int, default=40, help='Number of threads')
    parser.add_argument('--input_file')
    parser.add_argument('--input_dir')
    parser.add_argument('--MSA_script')
    #parser.add_argument('--debug', type=int, default=0, help='Debug mode')
    return parser.parse_args()
    
def main():
    args = parse_args()
    genome = args.genome
    #out_dir = args.tmp_output_dir
    threads = args.threads
    input_file = args.input_file
    process_TE_alignments(
      genome_dir=genome,
      TE_file=input_file,
      output_dir=args.input_dir + "/MSA",
      subset_script_path=args.MSA_script,
      threads=args.threads
    )
      

if __name__ == "__main__":
    main()


