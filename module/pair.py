import re
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
import argparse

def cluster_sequences(input_file, output_prefix):
    cmd = [
        "cd-hit-est",
        "-i", input_file,
        "-o", output_prefix,
        "-c", "0.95", 
        "-aS","0.95",
        "-aL","0.95",
        "-G","1",
        "-g","1",
        "-d", "0",  
        "-M", "16000",  
        "-T", "8" 
    ]

    try:
        subprocess.run(cmd, check=True)
        
    except Exception as e:
        print(f"cluster head/tail fail: {e}")

def parse_cluster_file(cluster_filename):
    id_to_representative = {}
    current_representative = None
    with open(cluster_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>Cluster'):
                current_representative = None
            else:
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                desc_part = parts[1]
                match = re.search(r'>(\S+)', desc_part)
                if match:
                    seq_id = match.group(1).strip("...")
                    if line.endswith('*'):
                        current_representative = seq_id.strip("...")
                        id_to_representative[seq_id] = seq_id.strip("...")
                    else:
                        if current_representative is not None:
                            id_to_representative[seq_id] = current_representative
    return id_to_representative


def parse_fasta(fasta_filename):
    id_to_rows = {}  
    current_row = 0
    with open(fasta_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                seq_id = line[1:].split()[0] 
                if seq_id not in id_to_rows:
                    id_to_rows[seq_id] = []  
                id_to_rows[seq_id].append(current_row) 
                current_row += 1
    return id_to_rows

def parse_fasta_row_to_id(fasta_filename):

    row_to_id = {}
    current_row = 0
    with open(fasta_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                seq_id = line[1:].split()[0]
                row_to_id[current_row] = seq_id
                current_row += 1
    return row_to_id

result_pair = []

def parse_args():
    parser = argparse.ArgumentParser(description='Find candidate helitrons')
    parser.add_argument('--input_dir', default="./f_oxy_out", help='Temporary output directory')
    return parser.parse_args()
    
def main():
    args = parse_args()
    input_dir = args.input_dir
    head_file = f"{input_dir}/left_ORF.fa"
    tail_file = f"{input_dir}/right_ORF.fa"
    cluster_sequences(head_file,f'{input_dir}/head_clustered.fa')
    cluster_sequences(tail_file,f'{input_dir}/tail_clustered.fa')

    head_clustered = f'{input_dir}/head_clustered.fa.clstr'
    head_id_to_rep = parse_cluster_file(head_clustered)

    head_id_to_rows = parse_fasta(head_file)  

    tail_row_to_id = parse_fasta_row_to_id(tail_file)

    tail_clustered = f'{input_dir}/tail_clustered.fa.clstr'
    tail_id_to_rep = parse_cluster_file(tail_clustered)

    output_file = f'{input_dir}/head_tail_pair.txt'
    result_pair = []
    with open(output_file, 'w') as outf:
        for seq_id, rows in head_id_to_rows.items(): 
            for row in rows: 
                tail_id = tail_row_to_id.get(row)
                if tail_id is None:
                    continue 
                tail_rep = tail_id_to_rep.get(tail_id)
                if tail_rep is None:
                    continue 
                head_rep = head_id_to_rep.get(seq_id)
                if head_rep is None:
                    continue 
                pair = f"{head_rep},{tail_rep}"
                if pair not in result_pair:
                    result_pair.append(pair)
                    outf.write(f"{pair}\n")


if __name__ == "__main__":
    main()

