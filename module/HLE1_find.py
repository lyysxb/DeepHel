import re
import sys
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import time
import argparse
from Bio.Seq import Seq

def parse_args():
    parser = argparse.ArgumentParser(description='Process flanked files to find helitrons')
    parser.add_argument('--input_dir', default="./f_oxy_out", help='Directory containing flanked files')
    parser.add_argument('--head_pattern', default="head.lcvs", help='Head pattern file')
    parser.add_argument('--tail_pattern', default="tail.lcvs", help='Tail pattern file')
    parser.add_argument('--output', default="f_oxy_hel.fa", help='Output file')
    parser.add_argument('--threads', type=int, default=48, help='Number of threads to use')
    return parser.parse_args()

def clean_sequence(sequence):

    sequence = sequence.upper()
    return re.sub(r'[^ATCG]', '', sequence)

def reverse_complement(sequence):

    return str(Seq(sequence).reverse_complement())

def compile_patterns(pattern_file):

    with open(pattern_file, 'r') as f:
        patterns = [line.strip() for line in f if line.strip()]
    return [re.compile(pattern) for pattern in patterns]

def find_matches(sequence, patterns):

    matches = []
    for pattern in patterns:
        for match in pattern.finditer(sequence):
            matches.append((match.start(), match.end()))
            break
    return matches

def process_sequence_variants(header, sequence, head_patterns, tail_patterns, seq_index, total_sequences):

    results = []
    

    forward_results = process_single_sequence(header, sequence, head_patterns, tail_patterns, '+')
    results.extend(forward_results)
    

    rev_seq = reverse_complement(sequence)
    reverse_results = process_single_sequence(header, rev_seq, head_patterns, tail_patterns, '-')
    results.extend(reverse_results)
    
    print(f"\rProcessing sequence {seq_index+1}/{total_sequences} (found {len(results)} fragments)", end="")
    return results

def process_single_sequence(header, sequence, head_patterns, tail_patterns, direction):

    parts = header.split(':')
    gene_id = parts[0]
    start_end = parts[1].split('-')
    original_start = int(start_end[0])
    original_end = int(start_end[1])

    cleaned_seq = clean_sequence(sequence)
    seq_len = len(cleaned_seq)

    if seq_len < 200:
        return []

    head_part = cleaned_seq[0:60]
    tail_part = cleaned_seq[-60:]
    tail_matches = find_matches(tail_part, tail_patterns)
    if not tail_matches:
        return []
    head_matches = find_matches(head_part, head_patterns)
    if not head_matches:
        return []
    
    sub_seq = cleaned_seq
    results = []
    results.append((f"{gene_id}:{original_start}-{original_end}", sub_seq))
    return results

def process_fasta(input_file, head_patterns, tail_patterns, output_file, num_threads, file_index, total_files):
    print(f"\nProcessing file {file_index+1}/{total_files}: {os.path.basename(input_file)}")
    sequences = []
    with open(input_file, 'r') as f_in:
        current_header = None
        current_sequence = []
        
        for line in f_in:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_sequence)))
                current_header = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        

        if current_header is not None:
            sequences.append((current_header, ''.join(current_sequence)))
    
    total_sequences = len(sequences)
    print(f"Found {total_sequences} sequences in {os.path.basename(input_file)}")
    
    all_results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i, (header, seq) in enumerate(sequences):
            futures.append(executor.submit(
                process_sequence_variants, header, seq, head_patterns, tail_patterns, i, total_sequences
            ))
        
        for future in futures:
            all_results.extend(future.result())
    
    with open(output_file, 'a') as f_out:
        for header, seq in all_results:
            f_out.write(f">{header}\n")
            f_out.write(f"{seq}\n")
    
    return len(all_results)

def find_all_flanked_files(input_dir="./out"):
    pattern = os.path.join(input_dir, "longest_repeats_*.flanked.fa")
    files = glob.glob(pattern)
    def extract_number(filename):
        match = re.search(r'longest_repeats_(\d+)\.flanked\.fa', filename)
        return int(match.group(1)) if match else 0
    
    files.sort(key=extract_number)
    return files

def main():
    args = parse_args()
    
    input_dir = args.input_dir
    head_pattern_file = args.head_pattern
    tail_pattern_file = args.tail_pattern
    output_file = args.output
    num_threads = args.threads
    
    print(f"Starting processing with {num_threads} threads...")
    start_time = time.time()
    
    print("Compiling pattern files...")
    head_patterns = compile_patterns(head_pattern_file)
    tail_patterns = compile_patterns(tail_pattern_file)
    print(f"Loaded {len(head_patterns)} head patterns")
    print(f"Loaded {len(tail_patterns)} tail patterns")
    

    input_files = find_all_flanked_files(input_dir)
    if not input_files:
        print(f"No longest_repeats_*.flanked.fa files found in {input_dir}")
        return
    
    total_files = len(input_files)
    print(f"Found {total_files} input files:")
    for i, file_path in enumerate(input_files):
        print(f"  {i+1}. {os.path.basename(file_path)}")
    

    with open(output_file, 'w') as f:
        pass
    
    total_fragments = 0
    for i, input_file in enumerate(input_files):
        fragments_count = process_fasta(
            input_file, head_patterns, tail_patterns, output_file, 
            num_threads, i, total_files
        )
        total_fragments += fragments_count
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*50)
    print("Processing completed!")
    print(f"Total files processed: {total_files}")
    print(f"Total fragments generated: {total_fragments}")
    print(f"Output file: {output_file}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    main()

