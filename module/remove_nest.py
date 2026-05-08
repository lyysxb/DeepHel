import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import subprocess
import argparse
import csv
from blastn_remove import self_blast_and_filter

def extract_sequence_id(filename):
        base_name = os.path.basename(filename)
        clean_name = re.sub(r'(_members)?(\.fa)?(\.rdmSubset)?\.fa\.aln\.fa$', '', base_name)
        match = re.search(r'(\d+[_-])', clean_name)
        if match:
            prefix = clean_name[:match.start(1)]
            suffix = clean_name[match.start(1):]
            return prefix + suffix.replace('_', ':', 1)
            
def make_blast_database(fasta_file, db_name):
    """
    Build BLAST database
    """
    cmd = ['makeblastdb', '-in', fasta_file, '-dbtype', 'nucl', '-out', db_name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Failed to build BLAST database: {result.stderr}")
        return False
    
    print(f"Successfully built BLAST database: {db_name}")
    return True

def run_blastn(query_file, db_file, thread, output_file, evalue=1e-5):
    """
    Run BLASTN alignment
    """
    cmd = [
        'blastn', '-query', query_file, '-db', db_file, 
        '-out', output_file, '-outfmt', '6', 
        #'-evalue', str(evalue), '-num_threads', '4'
        '-num_threads',thread
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"BLASTN failed: {result.stderr}")
        return False
    
    return True

def parse_blastn_output(blast_file):
    """
    Parse BLASTN output file
    """
    hits_dict = defaultdict(list)
    
    with open(blast_file, 'r') as f:
        for line_num, line in enumerate(f,1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 12:
                continue
                
            try:
                qseqid = parts[0]
                qstart = int(parts[6])
                qend = int(parts[7])
                
                if qstart > qend:
                    qstart, qend = qend, qstart
                
                hits_dict[qseqid].append((qstart, qend))
                
            except (ValueError, IndexError):
                continue
    
    return hits_dict

def calculate_coverage_from_blast(hits_dict, sequences):
    """
    Calculate base coverage from BLASTN results
    """
    coverage_dict = {}
    
    for seq_id, hits in hits_dict.items():
        if seq_id not in sequences:
            continue
            
        seq_len = sequences[seq_id]['length']
        if seq_len <= 0:
            continue
        
        coverage = np.zeros(seq_len, dtype=int)
        
        for start, end in hits:
            start_idx = max(0, start - 1)
            end_idx = min(seq_len, end)
            
            if start_idx < end_idx:
                coverage[start_idx:end_idx] += 1
        
        coverage_dict[seq_id] = {
            'length': seq_len,
            'coverage': coverage,
            'hits_count': len(hits),
            'sequence': sequences[seq_id]['seq'],
            'description': sequences[seq_id]['description'],
            'original_id': sequences[seq_id]['original_id']
        }
    
    return coverage_dict

def extract_sequence_id(filename):
    base_name = os.path.basename(filename)
    clean_name = re.sub(r'(_members)?(\.fa)?(\.rdmSubset)?\.fa\.aln\.fa$', '', base_name)
    match = re.search(r'(\d+[_-])', clean_name)
    if match:
       prefix = clean_name[:match.start(1)]
       suffix = clean_name[match.start(1):]
       return prefix + suffix.replace('_', ':', 1)
            
def parse_json_and_filter_sequences(json_file, fasta_file):
    """
    Parse JSON file and filter sequences with copy_num=2
    Return mapping from file_name to id
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    filtered_entries = [entry for entry in data if entry.get('copy_num') <= 5]

    id_mapping = {}
    for entry in filtered_entries:
        file_name = entry['file_name']
        file_id = extract_sequence_id(file_name)
        original_id = extract_sequence_id(entry['id'])
        id_mapping[file_id] = original_id
    

    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        reduce_id = extract_sequence_id(record.id)
        if reduce_id in id_mapping:
            sequences[reduce_id] = {
                'seq': str(record.seq),
                'length': len(record.seq),
                'description': record.description,
                'original_id': reduce_id 
            }
    
    return sequences, id_mapping
    
def merge_segments(segments, gap_threshold=20):

    if not segments:
        return []

    segments.sort(key=lambda x: x[0])
    
    merged = []
    current_start, current_end = segments[0]
    
    for i in range(1, len(segments)):
        seg_start, seg_end = segments[i]
        

        if seg_start - current_end <= gap_threshold:
            current_end = max(current_end, seg_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = seg_start, seg_end
    
    merged.append((current_start, current_end))
    return merged

def filter_segments_by_position(segments, seq_length, distance_threshold=150):

    filtered = []
    for seg_start, seg_end in segments:
        if (seg_start >= distance_threshold and 
            seg_end <= (seq_length - distance_threshold)):
            filtered.append((seg_start, seg_end))
    return filtered

def filter_segments_by_length(segments, min_length=100, max_length=2000):

    filtered = []
    for seg_start, seg_end in segments:
        seg_length = seg_end - seg_start
        if min_length <= seg_length <= max_length:
            filtered.append((seg_start, seg_end))
    return filtered

def calculate_average_coverage(coverage, start, end, window_size=50):

    if start < 0:
        start = 0
    if end > len(coverage):
        end = len(coverage)
    
    if start >= end:
        return 0
    
    segment = coverage[start:end]
    return np.mean(segment) if len(segment) > 0 else 0

def filter_segments_by_coverage_pattern(segments, coverage, threshold_ratio=0.3):

    filtered = []
    discarded = []
    
    for seg_start, seg_end in segments:
        # caculate mean coverage of start pos
        start_left_start = max(0, seg_start - 50)
        start_left_end = seg_start
        start_right_start = seg_start
        start_right_end = min(len(coverage), seg_start + 50)
        
        start_left_avg = calculate_average_coverage(coverage, start_left_start, start_left_end)
        start_right_avg = calculate_average_coverage(coverage, start_right_start, start_right_end)
        
        # caculate mean coverage of end pos
        end_left_start = max(0, seg_end - 50)
        end_left_end = seg_end
        end_right_start = seg_end
        end_right_end = min(len(coverage), seg_end + 50)
        
        end_left_avg = calculate_average_coverage(coverage, end_left_start, end_left_end)
        end_right_avg = calculate_average_coverage(coverage, end_right_start, end_right_end)
        

        condition1 = start_left_avg <= threshold_ratio * start_right_avg
        condition2 = threshold_ratio * end_left_avg >= end_right_avg
        
        if condition1 and condition2:
            filtered.append((seg_start, seg_end, {
                'start_left_avg': start_left_avg,
                'start_right_avg': start_right_avg,
                'end_left_avg': end_left_avg,
                'end_right_avg': end_right_avg
            }))
        else:
            if start_left_avg/start_right_avg < 0.1 and end_right_avg/end_left_avg < 0.55:
               # On one side of the alignment curve, there may be some minor fluctuations, but the overall trend is a rapid decline. Therefore, if one end shows a drop of more than 90% while the other end only drops by more than 45%, we choose to recalculate an additional 50 base pairs outward to determine whether the decline meets the set threshold.
               end_right_avg_true =  calculate_average_coverage(coverage, end_right_start+50, end_right_end+50)
               if end_right_avg_true/end_left_avg <= min(threshold_ratio,0.3):
                  filtered.append((seg_start, seg_end, {
                      'start_left_avg': start_left_avg,
                      'start_right_avg': start_right_avg,
                      'end_left_avg': end_left_avg,
                      'end_right_avg': end_right_avg
                  }))
                  continue
              if end_right_avg/end_left_avg < 0.1 and start_left_avg/start_right_avg < 0.55:
               start_left_avg_true = calculate_average_coverage(coverage,start_left_start-50, end_left_end-50)
               if start_left_avg_true/start_right_avg <= min(threshold_ratio,0.3) and length >= 150:
                  filtered.append((seg_start, seg_end, {
                      'start_left_avg': start_left_avg,
                      'start_right_avg': start_right_avg,
                      'end_left_avg': end_left_avg,
                      'end_right_avg': end_right_avg
                  }))
                  continue

            discarded.append((seg_start, seg_end, {
                'start_left_avg': start_left_avg,
                'start_right_avg': start_right_avg,
                'end_left_avg': end_left_avg,
                'end_right_avg': end_right_avg,
                'condition1': condition1,
                'condition2': condition2
            }))
    
    return filtered, discarded

def identify_peak_regions_to_remove(coverage,count_thre):
    """
    Identify peak regions to be removed (return region coordinates, not coverage array)
    """
    if len(coverage) == 0:
        return [], []
    
    mean_coverage = np.mean(coverage)
    max_coverage = np.max(coverage)
    seq_length = len(coverage)
    
    #print(f"Coverage stats - Max: {max_coverage}, Mean: {mean_coverage:.2f}, Length: {seq_length}")
    
    regions_to_remove = []
    discarded_regions = []
    
    # Check if sequence length > 2500 bp
    if seq_length <= 2500:
        #print("Sequence length <= 2500 bp - No peak removal applied")
        return regions_to_remove, discarded_regions
    
    # Criterion 1: Extreme peaks
    if max_coverage > 10000 and max_coverage > 5 * mean_coverage:
        #print("Applying Criterion 1: Extreme peak removal")
        threshold = 2 * mean_coverage
        
        # Find all positions above threshold
        above_threshold = coverage > threshold
        regions_to_remove = [(i, i+1) for i in range(len(above_threshold)) if above_threshold[i]]
        regions_to_remove = merge_segments(regions_to_remove, gap_threshold=20)
        
        #print(f"Will remove {len(regions_to_remove)} bases with coverage > {threshold:.1f}")
    
    # Criterion 2: Moderate peaks
    elif max_coverage > 50 and max_coverage > count_thre * mean_coverage:
        #print("Applying Criterion 2: Moderate peak removal")
        #threshold = 0.9 * count_thre * mean_coverage
        threshold = count_thre * mean_coverage
        # Find bases above threshold
        above_threshold = coverage > threshold
        
        # Find continuous segments of high coverage
        segments = []
        in_segment = False
        segment_start = 0
        
        for i, is_high in enumerate(above_threshold):
            if is_high and not in_segment:
                in_segment = True
                segment_start = i
            elif not is_high and in_segment:
                in_segment = False
                segment_end = i
                segments.append((segment_start, segment_end))
        
        if in_segment:
            segments.append((segment_start, len(coverage)))
        

        
        if segments:

            segments = merge_segments(segments, gap_threshold=20)
            #print(f"After merging (gap<=20): {len(segments)} segments")

            segments = filter_segments_by_position(segments, seq_length, distance_threshold=150)
            #print(f"After position filtering: {len(segments)} segments")

            segments = filter_segments_by_length(segments, min_length=100, max_length=2000)
            #print(f"After length filtering: {len(segments)} segments")

            regions_to_remove, discarded_regions = filter_segments_by_coverage_pattern(
                segments, coverage, threshold_ratio=0.3)
            
            #print(f"After coverage pattern filtering: {len(regions_to_remove)} segments to remove")
            #print(f"Discarded {len(discarded_regions)} segments due to coverage pattern")

            regions_to_remove = [(start, end) for start, end, _ in regions_to_remove]
    
    return regions_to_remove, discarded_regions

def create_filtered_sequence(original_seq, peak_regions):
    """
    Create new sequence by removing only the identified peak regions
    Keep all other bases (including those with coverage = 0 that are not in peak regions)
    """
    # Create a mask of positions to keep (True = keep, False = remove)
    keep_mask = np.ones(len(original_seq), dtype=bool)
    
    # Mark peak regions for removal
    for start, end in peak_regions:
        keep_mask[start:end] = False
    
    # Build new sequence from kept positions
    kept_bases = []
    for i, keep in enumerate(keep_mask):
        if keep:
            kept_bases.append(original_seq[i])
    
    filtered_sequence = ''.join(kept_bases)
    
    removed_count = len(original_seq) - len(filtered_sequence)
    #print(f"Original sequence length: {len(original_seq)} bp")
    #print(f"Filtered sequence length: {len(filtered_sequence)} bp")
    #print(f"Removed {removed_count} bases from peak regions")
    #print(f"Kept {len(filtered_sequence)} bases (including zero-coverage regions not in peaks)")
    
    return filtered_sequence, keep_mask

def create_filtered_coverage_for_plotting(original_coverage, peak_regions):
    """
    Create filtered coverage array for plotting (set peak regions to 0, keep others as is)
    """
    filtered_coverage = original_coverage.copy()
    
    for start, end in peak_regions:
        filtered_coverage[start:end] = 0
    
    return filtered_coverage

def plot_coverage_comparison(seq_id, original_coverage, filtered_coverage, 
                           peak_regions, discarded_regions, output_dir):
    """
    Plot coverage comparison before and after peak removal (English version)
    """
    seq_len = len(original_coverage)
    positions = np.arange(1, seq_len + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Original coverage
    ax1.plot(positions, original_coverage, linewidth=1.5, color='red', alpha=0.8)
    ax1.set_ylabel('Coverage Count', fontsize=12)
    ax1.set_title(f'Original Coverage: {seq_id}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, seq_len)
    
    y_max = max(np.max(original_coverage), 1)
    ax1.set_ylim(0, y_max * 1.1)
    ax2.set_ylim(0, y_max * 1.1)
    
    # Highlight regions that will be removed (green)
    for start, end in peak_regions:
        ax1.axvspan(start, end, alpha=0.3, color='green', label='To be removed' if start == peak_regions[0][0] else "")
        ax1.text((start + end) / 2, y_max * 0.7, 
                f'Remove\n{start}-{end}', 
                ha='center', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Highlight discarded regions (orange)
    for i, (start, end, stats) in enumerate(discarded_regions):
        ax1.axvspan(start, end, alpha=0.3, color='orange', label='Discarded' if i == 0 else "")
        
        info_text = f'Discarded\n{start}-{end}\n'
        info_text += f'SL:{stats["start_left_avg"]:.1f} SR:{stats["start_right_avg"]:.1f}\n'
        info_text += f'EL:{stats["end_left_avg"]:.1f} ER:{stats["end_right_avg"]:.1f}\n'
        info_text += f'C1:{stats["condition1"]} C2:{stats["condition2"]}'
        
        ax1.text((start + end) / 2, y_max * 0.5, 
                info_text, 
                ha='center', fontsize=6, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
    
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles[:2], labels[:2], loc='upper right')
    
    # Plot 2: Filtered coverage (for visualization only)
    ax2.plot(positions, filtered_coverage, linewidth=1.5, color='blue', alpha=0.8)
    ax2.set_xlabel('Position in Query Sequence (bp)', fontsize=12)
    ax2.set_ylabel('Coverage Count', fontsize=12)
    ax2.set_title(f'Coverage After Peak Removal (Visualization): {seq_id}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, seq_len)
    
    # Calculate statistics
    original_max = np.max(original_coverage)
    original_mean = np.mean(original_coverage)
    filtered_max = np.max(filtered_coverage)
    filtered_mean = np.mean(filtered_coverage)
    
    total_removed = sum(end - start for start, end in peak_regions)
    removed_percentage = (total_removed / seq_len) * 100 if seq_len > 0 else 0
    
    stats_text = f'Sequence Length: {seq_len:,} bp\n'
    stats_text += f'Original - Max: {original_max}, Mean: {original_mean:.2f}\n'
    stats_text += f'Visualization - Max: {filtered_max}, Mean: {filtered_mean:.2f}\n'
    stats_text += f'Bases to remove: {total_removed:,} ({removed_percentage:.1f}%)\n'
    stats_text += f'Peak regions: {len(peak_regions)}\n'
    stats_text += f'Discarded regions: {len(discarded_regions)}'
    
    if seq_len <= 2500:
        stats_text += f'\n*** Length constraint: No removal (≤2500 bp) ***'
    
    ax2.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    safe_seq_id = re.sub(r'[<>:"/\\|?*]', '_', seq_id)
    filename = os.path.join(output_dir, f"{safe_seq_id}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated plot: {filename}")
    return filename

def write_complete_fasta_with_filtered_sequences(original_fasta, filtered_sequences_data, output_dir):

    output_file = os.path.join(output_dir, "Final.fa")
    
    records = []
    filtered_count = 0
    total_count = 0
    

    for record in SeqIO.parse(original_fasta, "fasta"):
        total_count += 1
        seq_id = record.id
        

        if seq_id in filtered_sequences_data and 'filtered_sequence' in filtered_sequences_data[seq_id]:
            filtered_seq = filtered_sequences_data[seq_id]['filtered_sequence']
            if filtered_seq:  
                new_record = SeqRecord(
                    Seq(filtered_seq),
                    id=record.id,
                    description=record.description
                )
                records.append(new_record)
                filtered_count += 1
                continue
        
        records.append(record)
    
    SeqIO.write(records, output_file, "fasta")
    
    print(f"output_file: {output_file}")
    
    return output_file, filtered_count

def write_filtered_fasta(sequences_data, output_dir, id_mapping):
    """
    Write filtered sequences to new FASTA file (using original IDs)
    """
    output_file = os.path.join(output_dir, "filter_helitron.fa")
    
    records = []
    for seq_id, data in sequences_data.items():
        if 'filtered_sequence' in data and data['filtered_sequence']:
            # Use original ID (from JSON)
            original_id = data.get('original_id', seq_id)
            record = SeqRecord(
                Seq(data['filtered_sequence']),
                id=original_id,  # Use original ID
                description=""  # No additional description
            )
            records.append(record)
    
    if records:
        SeqIO.write(records, output_file, "fasta")
        #print(f"Written {len(records)} filtered sequences to: {output_file}")
        return output_file
    else:
        print("No filtered sequences to write")
        return None

def split_sequences_by_hle_type(final_seq, hle2_ids, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    hle1_file = os.path.join(out_dir, "hle1.fa")
    hle2_file = os.path.join(out_dir, "hle2.fa")

    hle2_ids_set = set(hle2_ids)
    

    hle1_count = 0
    hle2_count = 0
    skipped_count = 0
    hle2_matched_ids = []

    with open(hle1_file, 'w') as f_hle1, open(hle2_file, 'w') as f_hle2:

        for record in SeqIO.parse(final_seq, 'fasta'):
            original_id = record.id
            try:

                extracted_id = extract_sequence_id(original_id)
                
                if extracted_id in hle2_ids_set:

                    SeqIO.write(record, f_hle2, "fasta")
                    hle2_count += 1
                    hle2_matched_ids.append((original_id, extracted_id))
                else:

                    SeqIO.write(record, f_hle1, "fasta")
                    hle1_count += 1
                    
            except Exception as e:
                skipped_count += 1

                SeqIO.write(record, f_hle1, "fasta")
                hle1_count += 1
             
def write_removed_regions_csv(filtered_sequences_data, output_dir):

    csv_file = os.path.join(output_dir, "remove_record.csv")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_id', 'removed_region', 'region_mean_coverage', 'sequence_mean_coverage'])
        
        for seq_id, data in filtered_sequences_data.items():
            original_coverage = data['original_coverage']
            peak_regions = data['peak_regions']
            seq_mean_coverage = np.mean(original_coverage) if len(original_coverage) > 0 else 0
            
            for start, end in peak_regions:

                region_coverage = original_coverage[start:end]
                region_mean = np.mean(region_coverage) if len(region_coverage) > 0 else 0
                

                writer.writerow([
                    data.get('original_id', seq_id), 
                    f"{start}-{end}",
                    f"{region_mean:.2f}",
                    f"{seq_mean_coverage:.2f}"
                ])
    

    return csv_file
       
def parse_args():
    parser = argparse.ArgumentParser(description='Find candidate helitrons')
    parser.add_argument('--input_dir')
    parser.add_argument('--genome', default="sample.fa", help='Reference genome file')
    parser.add_argument('--threads', help='Number of threads')
    #parser.add_argument('--debug', type=int, default=0, help='Debug mode')
    return parser.parse_args()
    
def main():
    # File paths
    args = parse_args()
    input_dir = args.input_dir
    json_file = input_dir + "/window_log_v2/stats.json"  # Replace with your JSON file path
    fasta_file = input_dir + "/cl.fa"
    output_dir = input_dir + "/nest"
    gene_fasta = args.genome 
    count_thre = 1.8
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=== Nested Sequence Cleaning Analysis ===")
    print("Features:")
    print("- Use file_name as FASTA sequence ID")
    print("- Build BLAST database and perform self-alignment")
    print("- Identify and remove high-coverage peak regions")
    print("- Generate complete FASTA file with filtered sequences")
    print()
    
    # Step 1: Parse JSON and filter sequences
    fasta_file_clean = input_dir + "/confident_cl_clean.fa"
    reduce_fasta_file = input_dir + "/confident_cl_reduce.fa"
    os.system(f"python module/clean.py {fasta_file} {fasta_file_clean}")
    self_blast_and_filter(fasta_file_clean,reduce_fasta_file)
    sequences, id_mapping = parse_json_and_filter_sequences(json_file, reduce_fasta_file)
    
    if not sequences:
        print("Error: No eligible sequences found")
        return
    
    # Step 2: Create temporary FASTA file (only containing filtered sequences)
    temp_fasta = os.path.join(output_dir, "filtered_sequences_temp.fa")
    records = []
    for seq_id, data in sequences.items():
        record = SeqRecord(
            Seq(data['seq']),
            id=seq_id
        )
        records.append(record)
    
    SeqIO.write(records, temp_fasta, "fasta")
    print(f"Created temporary FASTA file: {temp_fasta}")
    
    # Step 3: Build BLAST database
    db_name = os.path.join(output_dir, "blast_db")
    if not make_blast_database(gene_fasta, db_name):
        print("Error: Failed to build BLAST database")
        return
    
    threads = str(args.threads)
    # Step 4: Run BLASTN self-alignment
    blast_output = os.path.join(output_dir, "blastn_results.txt")
    if not run_blastn(temp_fasta, db_name,threads, blast_output):
        print("Error: BLASTN alignment failed")
        return
    
    # Step 5: Parse BLASTN output
    hits_dict = parse_blastn_output(blast_output)
    
    # Step 6: Calculate coverage
    coverage_dict = calculate_coverage_from_blast(hits_dict, sequences)
    
    if not coverage_dict:
        print("Error: No coverage data calculated")
        return
    seq_ids = [] 
    # Step 7: Process each sequence
    filtered_sequences_data = {}
    count = 0
    for seq_id, data in coverage_dict.items():
        original_coverage = data['coverage']
        seq_len = data['length']
        original_sequence = data['sequence']
        
        #print(f"\n{'='*60}")
        #print(f"Processing sequence: {seq_id} (Original ID: {data['original_id']})")
        #print(f"Sequence length: {seq_len:,} bp")
        #print(f"Alignment hits: {data['hits_count']}")
        
        # Identify peak regions to remove
        peak_regions, discarded_regions = identify_peak_regions_to_remove(original_coverage,count_thre)
        
        # Create filtered sequence (only remove peak regions, keep everything else)
        if seq_len > 2500 and peak_regions:
            #print(f"*** APPLYING PEAK REMOVAL ***")
            filtered_sequence, keep_mask = create_filtered_sequence(original_sequence, peak_regions)
            count += 1
            seq_ids.append(seq_id)
            # Create filtered coverage for visualization only
            filtered_coverage_viz = create_filtered_coverage_for_plotting(original_coverage, peak_regions)
        else:
            #print(f"*** NO PEAK REMOVAL APPLIED ***")
            filtered_sequence = original_sequence
            filtered_coverage_viz = original_coverage.copy()
            peak_regions = []
            discarded_regions = []
        
        # Generate plot 
        #plot_coverage_comparison(seq_id, original_coverage, filtered_coverage_viz, 
        #                       peak_regions, discarded_regions, output_dir)
        
        # Store data for FASTA output
        filtered_sequences_data[seq_id] = {
            'filtered_sequence': filtered_sequence,
            'length': seq_len,
            'original_coverage': original_coverage,
            'peak_regions': peak_regions,
            'discarded_regions': discarded_regions,
            'original_id': data['original_id']
        }
    
    print(f"\nTotal sequences with peak removal applied: {count}")
    if count!=0:
       print("seq ids :")
       print(seq_ids)
       print(f"\n{'='*60}")
    
    
    complete_fasta, filtered_count = write_complete_fasta_with_filtered_sequences(reduce_fasta_file, filtered_sequences_data, output_dir)
    
    #filtered_only_fasta = write_filtered_fasta(filtered_sequences_data, output_dir, id_mapping)

    #print(f"trim seq total: {filtered_count}")
    
    for seq_id, data in filtered_sequences_data.items():
        filtered_length = len(data['filtered_sequence'])
        removed_bases = data['length'] - filtered_length
        if data['discarded_regions']:
            print(f"Discarded regions details:")
            for i, (start, end, stats) in enumerate(data['discarded_regions']):
                print(seq_id)
                print(f"  Region {i+1}: {start}-{end} bp")
                print(f"    Start: L{stats['start_left_avg']:.1f} vs R{stats['start_right_avg']:.1f} (C1: {stats['condition1']})")
                print(f"    End: L{stats['end_left_avg']:.1f} vs R{stats['end_right_avg']:.1f} (C2: {stats['condition2']})")
    
    # Clean up temporary files
    if os.path.exists(temp_fasta):
        os.remove(temp_fasta)
    for ext in ['.nhr', '.nin', '.nsq','.ndb','.njs','.nog','.nos','.not','.ntf','.nto']:
        db_file = db_name + ext
        if os.path.exists(db_file):
            os.remove(db_file)
    
    final_seq = input_dir + "/nest/Final.fa"
 
       
    detect_seqs = SeqIO.parse(final_seq, 'fasta')
    detect_SeqIDs = {k.id for k in detect_seqs}
    hle2_file = input_dir + "/FEMA_out/hle2_ids.txt"
    hle2_ids = []
    if os.path.exists(hle2_file):
      f_r = open(hle2_file,"r")
      
      for line in f_r.readlines():
          hle2_ids.append(line.strip("\n"))
    
      f_r.close()
    os.remove(fasta_file_clean)
    os.remove(reduce_fasta_file)
    #split_sequences_by_hle_type(final_seq,hle2_ids,input_dir)
    #write_removed_regions_csv(filtered_sequences_data, output_dir)
    #os.system(f"python module/clean.py {final_result} {final_clean_result}")
if __name__ == "__main__":
    main()


