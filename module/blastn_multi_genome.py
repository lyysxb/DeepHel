import os
import subprocess
import argparse
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import shutil

def reverse_complement(sequence):
    return str(Seq(sequence).reverse_complement())

def run_command(cmd, log_file=None):
    """Run system command"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if log_file:
            with open(log_file, "w") as f:
                f.write(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e.cmd}")
        return False

def prepare_blast_db(genome_file):
    """Prepare BLAST database"""
    if not os.path.exists(genome_file):
        raise FileNotFoundError(f"Genome file {genome_file} not found")

    cmd = f"makeblastdb -in {genome_file} -dbtype nucl"
    return run_command(cmd)

def run_blast(query, db, threads,output, task='blastn-short'):
    """Run BLAST alignment"""
    cmd = (
        f"blastn -db {db} -query {query} -task {task} -max_target_seqs 999999999 "
        f"-outfmt '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore stitle sstrand' "
        f"-out {output} -num_threads {threads} -evalue 1e-3"
    )
    return run_command(cmd)

def count_strands(input_file, output_file):
    """Count plus/minus strands"""
    counts = defaultdict(lambda: {'plus': 0, 'minus': 0})

    with open(input_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 14:
                continue
            qseqid = parts[0]
            strand = parts[13]
            alen = int(parts[3])
            if alen / 50 < 0.8:
                continue
            counts[qseqid][strand] += 1

    with open(output_file, 'w') as f:
        for id, data in counts.items():
            f.write(f"{id}\t{data['plus']}\t{data['minus']}\t{data['minus'] + data['plus']}\n")

def load_counts(count_file):
    """Load count file to dictionary"""
    counts = {}
    with open(count_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                qseqid, plus, minus, total = parts
                counts[qseqid] = int(total)
    return counts

def blast_to_bed(blast_file, bed_file, count_file):
    """Convert BLAST results to BED format"""
    counts = load_counts(count_file)

    with open(blast_file) as fin, open(bed_file, 'w') as fout:
        for line in fin:
            parts = line.strip().split('\t')
            qseqid = parts[0]
            chrom = parts[1].split('|')[0]
            start = int(parts[8])
            end = int(parts[9])
            strand = parts[13]
            alen = int(parts[3])

            if start > end:
                start, end = end, start

            fout.write(f"{chrom}\t{start - 1}\t{end}\t{qseqid}\t.\t{strand}\n")

def process_pairs(head_bed, tail_bed, forward_bed, reverse_bed):
    """Process pairs"""
    cmd_forward = f"bedtools window -a {head_bed} -b {tail_bed} -sm -l 0 -r 30000 > {forward_bed}"
    run_command(cmd_forward)

    cmd_reverse = f"bedtools window -a {tail_bed} -b {head_bed} -sm -l 0 -r 30000 > {reverse_bed}"
    run_command(cmd_reverse)
    return True

def filter_valid_pairs(valid_pair_file, forward_bed, reverse_bed, output_bed_file):
    """Filter results based on valid pairs"""
    valid_pairs = set()
    with open(valid_pair_file) as f:
        for line in f:
            h, t = line.strip().split(',')
            valid_pairs.add((h, t))
    
    with open(output_bed_file, 'w') as fout:
        with open(forward_bed) as fin:
            for line in fin:
                parts = line.strip().split('\t')
                if len(parts) >= 10 and parts[5] == "plus":
                    head_id = parts[3]
                    tail_id = parts[9]
                    if (head_id, tail_id) in valid_pairs:
                        fout.write(line)
        
        with open(reverse_bed) as fin:
            for line in fin:
                parts = line.strip().split('\t')
                if len(parts) >= 10 and parts[5] == "minus":
                    head_id = parts[9]
                    tail_id = parts[3]
                    if (head_id, tail_id) in valid_pairs:
                        fout.write(line)

def write_id(HLE_ids, id_file):
    """Write ID file"""
    with open(id_file, 'w') as f_i:
        for hle_id in HLE_ids:
            f_i.write(hle_id + '\n')

def convert_to_gene_bed(input_file, output_file, out_fa_file, genome_dict, id_file):
    """Convert to gene BED format"""
    bed_record = []
    chrom_list = []
    sequence_records = []
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            fields = line.strip().split('\t')
            if len(fields) < 10:
                continue
                
            strand = "+" if fields[5] == "plus" else "-"
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[8])
            
            if chrom not in genome_dict:
                continue
                
            seqLen = len(genome_dict[chrom])
            
            if start < 500 and (seqLen - end) < 500:
                if (chrom, start, end) not in bed_record:
                    bed_record.append((chrom, start, end))
                    outfile.write(f"{chrom}\t{start}\t{end}\t{strand}\n")
                    
                    if chrom not in chrom_list:
                        chrom_list.append(chrom)
                        if strand == '+':
                            sequence = genome_dict[chrom]
                        else:
                            sequence = reverse_complement(genome_dict[chrom])
                        
                        seq_record = SeqRecord(
                            Seq(sequence),
                            id=chrom,
                        )
                        sequence_records.append(seq_record)
    
    if sequence_records:
        with open(out_fa_file, 'w') as f_w:
            SeqIO.write(sequence_records, f_w, "fasta")
    
    if chrom_list:
        write_id(chrom_list, id_file)
    
    return len(sequence_records) > 0, chrom_list

def process_single_genome(genome_file, input_dir, threads,output_prefix):
    """Process single genome"""
    genome_name = os.path.splitext(os.path.basename(genome_file))[0]
    genome_output_dir = os.path.join(input_dir, genome_name)
    
    os.makedirs(genome_output_dir, exist_ok=True)
    
    print(f"Processing genome: {genome_name}")
    
    # Load genome sequences
    genome_dict = {}
    try:
        for record in SeqIO.parse(genome_file, 'fasta'):
            genome_dict[record.id] = str(record.seq).upper()
    except Exception as e:
        print(f"Failed to load genome file: {e}")
        return None, None
    
    # Prepare BLAST database
    if not prepare_blast_db(genome_file):
        return None, None
    
    # Run BLAST alignment
    blast_head = os.path.join(genome_output_dir, "head_blast.txt")
    if not run_blast(os.path.join(input_dir, "head_clustered.fa"), genome_file, threads, blast_head):
        return None, None
    
    blast_tail = os.path.join(genome_output_dir, "tail_blast.txt")
    if not run_blast(os.path.join(input_dir, "tail_clustered.fa"), genome_file, threads, blast_tail):
        return None, None
    
    # Generate count files
    count_strands(blast_head, os.path.join(genome_output_dir, "head_count.txt"))
    count_strands(blast_tail, os.path.join(genome_output_dir, "tail_count.txt"))
    
    # Convert to BED format
    blast_to_bed(blast_head, os.path.join(genome_output_dir, "head.bed"), 
                 os.path.join(genome_output_dir, "head_count.txt"))
    blast_to_bed(blast_tail, os.path.join(genome_output_dir, "tail.bed"), 
                 os.path.join(genome_output_dir, "tail_count.txt"))
    
    # Sort BED files
    run_command(f"sort -k1,1 -k2,2n {genome_output_dir}/head.bed > {genome_output_dir}/head.sorted.bed")
    run_command(f"sort -k1,1 -k2,2n {genome_output_dir}/tail.bed > {genome_output_dir}/tail.sorted.bed")
    
    head_bed = os.path.join(genome_output_dir, "head.sorted.bed")
    tail_bed = os.path.join(genome_output_dir, "tail.sorted.bed")
    forward_bed = os.path.join(genome_output_dir, "forward_pairs.bed")
    reverse_bed = os.path.join(genome_output_dir, "reverse_pairs.bed")
    
    # Filter pairs
    if not process_pairs(head_bed, tail_bed, forward_bed, reverse_bed):
        return None, None
    
    # Final filtering
    pairs = os.path.join(input_dir, "head_tail_pair.txt")
    all_pairs_bed = os.path.join(genome_output_dir, "all_HLE2_pairs.bed")
    
    filter_valid_pairs(pairs, forward_bed, reverse_bed, all_pairs_bed)
    
    # Convert to gene BED format
    simple_bed = os.path.join(genome_output_dir, "all_simple_HLE2_pairs.bed")
    output_fasta = f"{output_prefix}_{genome_name}.fa"
    id_file = os.path.join(genome_output_dir, "hle2_ids.txt")
    
    has_output, chrom_list = convert_to_gene_bed(all_pairs_bed, simple_bed, output_fasta, genome_dict, id_file)
    
    if has_output:
        return output_fasta, id_file
    else:
        print(f"No output for genome: {genome_name}")
        return None, None

def merge_output_files(output_files, id_files, final_output, merged_id_file):
    """Merge all output files"""
    all_records = []
    all_ids = []
    
    # Merge FASTA files
    for file in output_files:
        if file and os.path.exists(file):
            try:
                records = list(SeqIO.parse(file, "fasta"))
                all_records.extend(records)
            except Exception as e:
                print(f"Error reading file {file}: {e}")
    
    # Merge ID files
    for id_file in id_files:
        if id_file and os.path.exists(id_file):
            try:
                with open(id_file, 'r') as f:
                    ids = [line.strip() for line in f if line.strip()]
                    all_ids.extend(ids)
            except Exception as e:
                print(f"Error reading ID file {id_file}: {e}")
    
    # Write merged FASTA file
    if all_records:
        with open(final_output, "w") as output_handle:
            SeqIO.write(all_records, output_handle, "fasta")
    
    # Write merged ID file
    if all_ids:
        with open(merged_id_file, "w") as id_handle:
            for hle_id in all_ids:
                id_handle.write(hle_id + '\n')
    
    return len(all_records), len(all_ids)

def main():
    parser = argparse.ArgumentParser(description="Genome alignment analysis pipeline")
    parser.add_argument("--genome", required=True, help="Genome FASTA file paths, comma-separated")
    parser.add_argument("--out", required=True, help="Final merged output file path")
    parser.add_argument("--input_dir", required=True, help="Input directory path")
    parser.add_argument("--threads")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    final_output = args.out
    
    os.makedirs(input_dir, exist_ok=True)
    
    # Parse genome file list
    genome_files = [g.strip() for g in args.genome.split(',') if g.strip()]
    
    if not genome_files:
        print("Error: No valid genome files provided")
        return
    
    # Check if all genome files exist
    missing_files = []
    for genome_file in genome_files:
        if not os.path.exists(genome_file):
            missing_files.append(genome_file)
    
    if missing_files:
        print(f"Error: Missing genome files: {missing_files}")
        return
    
    print(f"Processing {len(genome_files)} genome files")
    
    # Process each genome
    output_files = []
    id_files = []
    successful_genomes = 0
    
    for i, genome_file in enumerate(genome_files, 1):
        print(f"[{i}/{len(genome_files)}] {os.path.basename(genome_file)}")
        
        genome_name = os.path.splitext(os.path.basename(genome_file))[0]
        output_prefix = os.path.join(input_dir, f"{genome_name}_output")
        
        output_file, id_file = process_single_genome(genome_file, input_dir, args.threads,output_prefix)
        if output_file and id_file:
            output_files.append(output_file)
            id_files.append(id_file)
            successful_genomes += 1
    
    # Merge all outputs
    if output_files:
        merged_id_file = os.path.join(input_dir, "hle2_ids.txt")
        
        num_sequences, num_ids = merge_output_files(output_files, id_files, final_output, merged_id_file)
        
        print(f"\nProcessing completed:")
        print(f"  Successfully processed genomes: {successful_genomes}/{len(genome_files)}")
        print(f"  Total sequences: {num_sequences}")
        print(f"  Total IDs: {num_ids}")
        print(f"  Merged FASTA: {final_output}")
        print(f"  Merged ID file: {merged_id_file}")
    else:
        print("\nNo output files generated")

if __name__ == "__main__":
    main()
