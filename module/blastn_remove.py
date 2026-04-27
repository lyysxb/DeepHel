#!/usr/bin/env python3
import subprocess
import tempfile
import os
from pathlib import Path

def self_blast_and_filter(fasta_file, output_file="filtered.fasta"):

    db_name = "self_blast_db"
    subprocess.run(f"makeblastdb -in {fasta_file} -dbtype nucl -out {db_name} -parse_seqids", 
                   shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    blast_output = "blast_results.tsv"
    subprocess.run(
        f"blastn -query {fasta_file} -db {db_name} -out {blast_output} "
        f"-outfmt '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen' "
        f"-evalue 1e-5 -num_threads 4 -max_hsps 1",
        shell=True, check=True
    )
    
    sequences = {}
    with open(fasta_file) as f:
        seq_id = ""
        seq_lines = []
        for line in f:
            if line.startswith(">"):
                if seq_id:
                    sequences[seq_id] = "".join(seq_lines)
                seq_id = line[1:].strip().split()[0]
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if seq_id:
            sequences[seq_id] = "".join(seq_lines)
    

    to_remove = set()
    with open(blast_output) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 13:
                continue
                
            qseqid, sseqid = fields[0], fields[1]
            if qseqid == sseqid:  
                continue
                
            length = int(fields[3])  
            qlen = int(fields[12])   
            sstart = int(fields[8])  
            send = int(fields[9])    
            slen = int(fields[13])   
            

            target_coverage = abs(send - sstart) / slen if slen > 0 else 0
            

            if (length / qlen > 0.9) and (sstart == 1 or (qlen - send) == 0) and (target_coverage < 0.25) and qlen > 1000:
                #print(f"删除序列: {qseqid}")
                to_remove.add(qseqid)
    

    with open(output_file, "w") as out:
        for seq_id, seq in sequences.items():
            if seq_id not in to_remove:
                out.write(f">{seq_id}\n{seq}\n")
    

    for ext in ['.nhr', '.nin', '.nsq','.ndb','.njs','.nog','.nos','.not','.ntf','.nto']:
        temp_file = f"{db_name}{ext}"
        if os.path.exists(temp_file):
            os.remove(temp_file)
    if os.path.exists(blast_output):
        os.remove(blast_output)
    


if __name__ == "__main__":
    self_blast_and_filter("../maize_out_try/nest/final2.fa", "../maize_out_try/nest/final2s.fa")

