import json
import numpy as np
import os
import re
from Bio import SeqIO
from collections import defaultdict
from typing import List, Tuple, Union
import sys
import subprocess
import tempfile
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from find_small_tir import find_common_substrings_dp
from exclude_ltr import ex_ltr
from collections import defaultdict
def read_fasta(fasta_file):
    sequences = {}
    current_header = None
    current_sequence = []
    
    try:
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('>'):

                    if current_header is not None:
                        sequences[current_header] = ''.join(current_sequence)

                    current_header = line[1:].strip()
                    current_sequence = []
                else:
                    current_sequence.append(line)
            

            if current_header is not None:
                sequences[current_header] = ''.join(current_sequence)
                
    except FileNotFoundError:
        print(f"{fasta_file} not exist")
        sys.exit(1)
    except Exception as e:
        print(f"fail: {e}")
        sys.exit(1)
    
    return sequences

def safe_delete(path):
    try:
       if os.path.isfile(path) or os.path.islink(path):
          #print("remove file",path)
          os.remove(path)
       elif os.path.isdir(path):
          #print("remove dir",path)
          shutil.rmtree(path)
    except Exception as e:
       print(f"{path} | {str(e)}")


def remove_trf_regions(sequence, trf_regions):
    if not trf_regions:
        return sequence
    trf_regions.sort(key=lambda x: x[0])
    result = list(sequence)
    for start, end in reversed(trf_regions):
        start_idx = start - 1
        end_idx = end
        del result[start_idx:end_idx]  
    return ''.join(result)

def write_fasta(sequences, output_file):
    with open(output_file, 'w') as f:
        for header, sequence in sequences.items():
            f.write(f">{header}\n")
            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i+60] + "\n")
                

def clean_sequence(seq, max_length=25000):
    seq = ''.join([b for b in seq.upper() if b in 'ATCG'])
    return seq[:max_length] if len(seq) > max_length else seq

class StructureSearch:
    def __init__(self, genome, START=0):
        self.START = int(START)
        self.genome = genome
        self.records = {rec.id: rec.seq for rec in SeqIO.parse(genome, 'fasta')}

    def stem_loop(self, stem_loop_description_file,tailLen,output_dir):
        basename = os.path.basename(self.genome)
        rnabobopt = os.path.join(output_dir, f"{basename}.stemloop.txt")
        
        with open(rnabobopt, 'w') as rnabf:
            rnabob_program = subprocess.Popen(
                ["rnabob", "-c", "-q", "-F", "-s", stem_loop_description_file, self.genome],
                stderr=subprocess.DEVNULL, stdout=rnabf)
            rnabob_program.wait()

        complement_dict = {
            'A': "T", "T": "A", "G": "C", "C": "G",
            "K": "M", "M": "K", "Y": "R", "R": "Y", "S": "S", "W": "W",
            "B": "V", "V": "B", "H": "D", "D": "H", "N": "N", "X": "X"
        }
        stem_loop_loc = []
        if os.path.exists(rnabobopt):
            with open(rnabobopt, 'r') as F:
                for line in F:
                    line = line.strip()
                    if re.match('\d', line):
                        splitline = re.split('\s+', line)[:3]
                        chrid = splitline[2]
                        if int(splitline[1]) < 0:
                            continue
                        if int(splitline[0]) < int(splitline[1]):
                            strand = '+'
                            length = int(splitline[1]) - int(splitline[0]) + 1
                            start = int(splitline[0]) + self.START
                            end = start + length - 1
                        else:
                            strand = '-'
                            continue
                    else:
                        if strand == '-':
                            continue
                        seq = line.strip('|').split('|')
                        if len(seq) == 3:
                            continue
                        helix_seq1, loop_seq, helix_seq2, tail_seq = seq
                        stem_len = len(helix_seq1)
                        loop_len = len(loop_seq)
                        midpoint = int(len(loop_seq)/2)
                        for i in range(midpoint):
                            if loop_seq[i] == complement_dict.get(loop_seq[-i - 1], 'N'):
                                stem_len += 1
                                loop_len -= 2
                        if loop_len >= 1:
                            if abs(end-tailLen) < 10: 
                                stem_loop_loc.append([chrid, str(start), str(end), str(stem_len), str(loop_len), strand])
            stem_loop_loc = sorted(stem_loop_loc, key=lambda x: -int(x[2])) 
            #os.remove(rnabobopt)
        return stem_loop_loc

    def inverted_detection(self, sequencefile, minitirlen, maxtirlen, mintirdist, maxtirdist, seed, find_type, left_n,right_n,total_len):
        output_dir = os.path.dirname(sequencefile)
        basename = os.path.basename(sequencefile)
        
        dbname = os.path.join(output_dir, f"{basename}.invdb")
        invttirfile = os.path.join(output_dir, f"{basename}.inv.txt")
        
        mkinvdb = subprocess.Popen(
            ['gt', 'suffixerator', '-db', sequencefile, '-indexname', dbname, 
             '-mirrored', '-dna', '-suf', '-lcp', '-bck'],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        mkinvdb.wait()

        with open(invttirfile, 'w') as invf:
            runinvsearch = subprocess.Popen(
                ['gt', 'tirvish', '-index', dbname, '-mintirlen', str(minitirlen),
                 '-maxtirlen', str(maxtirlen), '-similar', '80', '-mintirdist', str(mintirdist),
                 '-maxtirdist', str(maxtirdist), '-mintsd', '0', '-seed', str(seed),
                 '-vic', '1', '-overlaps', 'all', '-xdrop', '0'],
                stderr=subprocess.DEVNULL, stdout=invf)
            runinvsearch.wait()
        
        invt_list = []
        with open(invttirfile, 'r') as F:
            for line in F:
                if line.startswith('#'):
                    continue
                splitlines = line.rstrip().split('\t')
                if splitlines[2] == 'repeat_region':
                    chrmid = splitlines[0]
                    id = splitlines[8].replace('ID=', '')
                    t = 1
                elif splitlines[2] == 'terminal_inverted_repeat_element':
                    sim = re.findall('tir_similarity=(\d+\.\d+)', splitlines[8])[0]
                elif splitlines[2] == 'terminal_inverted_repeat':
                    if t == 1:
                        left_start = int(splitlines[3]) + self.START
                        left_end = int(splitlines[4]) + self.START
                        left_expand = '-'.join([str(left_start), str(left_end)])
                        invt_length_left = int(splitlines[4]) - int(splitlines[3]) + 1
                        t += 1
                    else:
                        right_start = int(splitlines[3]) + self.START
                        right_end = int(splitlines[4]) + self.START
                        right_expand = '-'.join([str(right_start), str(right_end)])
                        invt_length_right = int(splitlines[4]) - int(splitlines[3]) + 1
                        valid_3 = (
                            invt_length_left >= 9 and 
                            invt_length_right >= 9 and 
                            invt_length_left <= 20 and 
                            invt_length_right <= 20
                        )
                        valid_2 = (
                            invt_length_left >= 12 and
                            invt_length_right >= 12 and
                            invt_length_left <= 20 and
                            invt_length_right <= 20
                        )
                        
                        if right_start >= right_n and left_end <= left_n and invt_length_left >= 4 and invt_length_right >= 4 and left_start < 5 and (total_len - right_end) < 5:
                            invt_list.append([
                                chrmid, 
                                str(left_start), 
                                str(left_end),
                                str(right_start),
                                str(right_end),
                                left_expand,
                                right_expand,
                                (invt_length_right + invt_length_left)/2, 
                                'pattern1',
                                sim
                             ])
                            
        invt_list = sorted(invt_list, key=lambda x: int(x[1]))
        #os.remove(invttirfile)
        #subprocess.run(['rm', f'{dbname}*'], stderr=subprocess.DEVNULL)
        return invt_list

def create_tir_test_sequence(record, output_dir):
    seq = clean_sequence(str(record.seq))
    head = seq[:50] if len(seq) >= 50 else seq
    tail = seq[-80:] if len(seq) >= 80 else seq
    test_seq = head + 'N' * 10 + tail
    
    output_file = os.path.join(output_dir, f"{record.id}_TIRtest.fa")
    with open(output_file, 'w') as f:
        f.write(f">{record.id}_TIRtest\n{test_seq}\n")
    return output_file

def reverse_complement(sequence):
    
    return str(Seq(sequence).reverse_complement())

def create_tail_50bp_sequence(record, output_dir):
    seq = clean_sequence(str(record.seq))
    tail = seq[-50:] if len(seq) >= 50 else seq
    
    output_file = os.path.join(output_dir, f"{record.id}_tail50bp.fa")
    with open(output_file, 'w') as f:
        f.write(f">{record.id}_tail50bp\n{tail}\n")
    return output_file

def create_tir_mode3_sequence(record, output_dir):
    head = record.seq[:50] if len(record.seq) >= 50 else record.seq
    tail = record.seq[-50:] if len(record.seq) >= 50 else record.seq
    test_seq = head + 'N' * 10 + tail
    n_connector_pos = len(head)
    
    output_file = os.path.join(output_dir, f"{record.id}_TIRmode3.fa")
    with open(output_file, 'w') as f:
        f.write(f">{record.id}_TIRmode3\n{test_seq}\n")
    return output_file, n_connector_pos

def is_cacta_tagtg(sequence):
    seq = clean_sequence(str(sequence))
    if len(seq) < 10:
        return False
    
    starts_with_cacta = seq.startswith('CACTA')
    ends_with_tagtg = seq.endswith('TAGTG')
    
    return starts_with_cacta and ends_with_tagtg

def run_itrsearch(input_dir,tools_dir,input_fasta,cur_dir):
    log_file = f"{input_dir}/{os.path.basename(input_fasta)}.itr.log"
    itr_file = f"{input_dir}/{os.path.basename(input_fasta)}.itr"
    print("Running ITRsearch detection...")
    

    with open(log_file, 'w') as log_f:
        itrsearch_cmd = [f'{tools_dir}/itrsearch', '-i', '0.7', '-l', '7', input_fasta]
        result = subprocess.run(itrsearch_cmd, stdout=log_f, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"ITRsearch ex fail: {result.stderr}")
            return set()
        os.system(f"mv {cur_dir}/confident_struc_helitrons.fa.itr {input_dir}")
    itr_ids = set()
    if os.path.exists(itr_file):
        with open(itr_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    parts = line.split()
                    if len(parts) > 0:
                        full_id = parts[0][1:] 
                        clean_id = full_id
                        itr_ids.add(clean_id)
    else:
        print(f"ITR{itr_file}")
    
    return itr_ids

def run_ltrsearch(input_dir,tools_dir,input_fasta,cur_dir):
    log_file = f"{input_dir}/{os.path.basename(input_fasta)}.ltr.log"
    ltr_file = f"{input_dir}/{os.path.basename(input_fasta)}.ltr"
    
    print("Running LTRsearch detection...")
    
    with open(log_file, 'w') as log_f:
        ltrsearch_cmd = [f'{tools_dir}/ltrsearch', '-i', '0.7', input_fasta]
        result = subprocess.run(ltrsearch_cmd, stdout=log_f, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"LTRsearch exec fail: {result.stderr}")
            return set()
        os.system(f"mv {cur_dir}/confident_struc_helitrons.fa.ltr {input_dir}")
        
    ltr_ids = set()
    if os.path.exists(ltr_file):
        with open(ltr_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    parts = line.split()
                    if len(parts) > 0:
                        full_id = parts[0][1:] 
                        clean_id = full_id
                        ltr_ids.add(clean_id)

    else:
        print(f"LTR not exists: {ltr_file}")
    
    return ltr_ids

def has_poly_at(sequence, min_consecutive=4, window_size=15, long_consecutive=8):

    seq = clean_sequence(str(sequence))
    if len(seq) < max(min_consecutive, window_size):
        return False
    
    pattern_short_a = 'A{' + str(min_consecutive) + ',}'
    pattern_short_t = 'T{' + str(min_consecutive) + ',}'
    pattern_long_a = 'A{' + str(long_consecutive) + ',}'
    pattern_long_t = 'T{' + str(long_consecutive) + ',}'

    start_seq = seq[:min_consecutive]
    if re.search(pattern_short_a, start_seq) or re.search(pattern_short_t, start_seq):
        return True
    

    end_seq = seq[-min_consecutive:]
    if re.search(pattern_short_a, end_seq) or re.search(pattern_short_t, end_seq):
        return True
    
    return False


def process_single_sequence(record, output_dir, desc1, desc2, hle2_ids):
    result = {}

    
    if is_cacta_tagtg(record.seq):
        result = {
            'type': 'pattern_cacta',
            'source': 'CACTA_start_TAGTG_end',
            'description': 'CACTA_start_TAGTG_end'
        }
        return (record.id, result)
    

    strLen = len(clean_sequence(str(record.seq)))
    tir_test_file = create_tir_test_sequence(record, output_dir)
    tir_searcher = StructureSearch(tir_test_file)
    tir_result = tir_searcher.inverted_detection(
        tir_test_file,
        minitirlen=5,
        maxtirlen=50,
        mintirdist=10,
        maxtirdist=140,
        seed=8,
        find_type="pattern5",
        left_n = 49,
        right_n = 59,
        total_len = 140
    )
    if tir_result:
        tir_right_end_dis = 140 - int(tir_result[0][4])
        actual_tir_left_start = int(tir_result[0][1])
        actual_tir_left_end = int(tir_result[0][2])
        actual_tir_right_start = strLen -(140 - int(tir_result[0][3]))
        actual_tir_right_end = strLen - (140 - int(tir_result[0][4]))
        if actual_tir_left_start < 5 and tir_right_end_dis < 5:
            result = {
                'type': 'pattern5',
                'tir': f"{tir_result[0][1]}-{tir_result[0][4]}",
                'tir_left_start': str(actual_tir_left_start),
                'tir_left_end': str(actual_tir_left_end),
                'tir_right_start':str(actual_tir_right_start),
                'tir_right_end': str(actual_tir_right_end),
                'tir_similarity': tir_result[0][8],
                'source': 'tail50bp_for_stemloop + extended_for_TIR'
            }
            os.remove(tir_test_file)
            return (record.id, result)
            
    if extract_sequence_id(record.id) in hle2_ids:
       return (record.id, None)
       
    if has_poly_at(record.seq, min_consecutive=6, window_size=15, long_consecutive=10):
        result = {
            'type': 'pattern_poly_at',
            'source': 'poly_A_T_detected',
            'description': 'poly_A_T_detected'
        }
        return (record.id, result)
    seq_str = str(record.seq)
    if "TG" in seq_str[:3]:
        result = {
            'type': 'TG',
        }
        return (record.id,result)

    first_20 = seq_str[:20]
    if (all(base in "AT" for base in first_20) and re.search(r'^(A{5,}|T{5,})', seq_str)) or re.search(r'(TA){6,}$', seq_str):
        result = {
            'type': 'cons_AT',
        }
        return (record.id,result)
    # for f in [tail50bp_file, tir_test_file]:
    for f in [tir_test_file]:
        if os.path.exists(f):
            os.remove(f)
            
    seq1 = clean_sequence(str(record.seq))[:8]
    seq2 = reverse_complement(clean_sequence(str(record.seq))[-8:]) 
    small_result = find_common_substrings_dp(seq1,seq2,5)
    #print(record.id,small_result)
    if small_result != []:
        #print(record.id,small_result)
        result = {
                  'type': 'pattern6',
                  'source': 'smallTIR'
        }
    return (record.id, result) if result else (record.id, None)

def filter_results_from_json(results_file, itr_ids, ltr_ids, prev_filtered_ids):
    filtered_ids_set = set(itr_ids) | set(ltr_ids) | set(prev_filtered_ids or [])
    filtered_ids = list(filtered_ids_set)
    
    if not os.path.exists(results_file):
        print(f"file {results_file} not exists")
        return list(filtered_ids)
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        pattern_counts = {
            'itrsearch': len(itr_ids),
            'ltrsearch': len(ltr_ids),
            'pattern_cacta': 0,
            'pattern_poly_at': 0,
            'pattern5': 0,
            'pattern6': 0
        }
        
        for seq_id, result_data in results.items():
            if result_data['type'] == 'pattern_cacta':
                filtered_ids.append(seq_id)
                pattern_counts['pattern_cacta'] += 1
            elif result_data['type'] == 'pattern_poly_at':
                filtered_ids.append(seq_id)
                pattern_counts['pattern_poly_at'] += 1
            elif result_data['type'] == 'pattern6':
                filtered_ids.append(seq_id)
                pattern_counts['pattern6'] += 1
            elif result_data['type'] == 'pattern5':
                tir_left_start = int(result_data.get('tir_left_start', 999))
                if tir_left_start <= 2:
                    filtered_ids.append(seq_id)
                    pattern_counts['pattern5'] += 1
            else:
                filtered_ids.append(seq_id)
        filtered_ids = set(filtered_ids)
        
    except Exception as e:
        print(f"fail read: {e}")
    
    return list(filtered_ids)

def write_filtered_fasta(input_fasta, filtered_ids,out_fasta):
    basename = os.path.basename(input_fasta)
    #name, ext = os.path.splitext(basename)
    output_file = out_fasta
    
    filtered_count = 0
    total_count = 0
    
    with open(output_file, 'w') as out_f:
        for record in SeqIO.parse(input_fasta, 'fasta'):
            total_count += 1
            if record.id not in filtered_ids:
                SeqIO.write(record, out_f, 'fasta')
                filtered_count += 1
    
    print(f"from {total_count} sequences select {filtered_count} 个序列")
    print(f"filter file save at: {output_file}")
    
    return output_file, filtered_count  

def load_json_data(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"fail read json: {e}")
        return []


def get_sequence_id_info(json_data, target_id):
    for record in json_data:
        if record.get('id') == target_id:
            return record
    return None


def extract_region_from_aln(alignment_file, actual_start, actual_end):
    try:
        records = list(SeqIO.parse(alignment_file, 'fasta'))
        if not records:
            return [], actual_start


        seq_length = len(records[0].seq)
        if actual_start < 0 or actual_end >= seq_length or actual_start > actual_end:
            return [], actual_start


        region_sequences = []
        for record in records:
            sequence = str(record.seq)
            region_seq = list(sequence[actual_start:actual_end + 1])
            region_sequences.append(region_seq)

        return region_sequences, actual_start

    except Exception as e:
        print(f"fail ex region: {e}")
        return [], actual_start


def extract_region_from_aln_v1(alignment_file, actual_start, base_num):
    try:

        records = list(SeqIO.parse(alignment_file, 'fasta'))
        if not records:
            return [], actual_start


        seq_length = len(records[0].seq)
        if actual_start < 0 or actual_start >= seq_length:
            return [], actual_start

        num_sequences = len(records)
        region_sequences = [[] for _ in range(num_sequences)]

        current_pos = actual_start
        collected_columns = 0
        region_start = actual_start  

        while collected_columns < base_num and current_pos < seq_length:

            column = [str(records[i].seq[current_pos]) for i in range(num_sequences)]


            valid_bases = [base for base in column if base.upper() in ['A', 'T', 'C', 'G']]

            if len(valid_bases) > 1:
                for i in range(num_sequences):
                    region_sequences[i].append(column[i])
                collected_columns += 1

            current_pos += 1


        if collected_columns < base_num:
            return [], actual_start

        return region_sequences, region_start

    except Exception as e:
        print(f"fail ex region: {e}")
        return [], actual_start


def extract_region_from_aln_v2(alignment_file, actual_end, base_num):
    try:

        records = list(SeqIO.parse(alignment_file, 'fasta'))
        if not records:
            return [], actual_end


        seq_length = len(records[0].seq)
        if actual_end < 0 or actual_end >= seq_length:
            return [], actual_end

        num_sequences = len(records)
        region_sequences = [[] for _ in range(num_sequences)]

        current_pos = actual_end
        collected_columns = 0
        valid_positions = []  


        while collected_columns < base_num and current_pos >= 0:

            column = [str(records[i].seq[current_pos]) for i in range(num_sequences)]


            valid_bases = [base for base in column if base.upper() in ['A', 'T', 'C', 'G']]

            if len(valid_bases) > 1:
                valid_positions.append(current_pos)
                collected_columns += 1

            current_pos -= 1


        if collected_columns < base_num:
            return [], actual_end


        valid_positions.sort()
        region_start = valid_positions[0] if valid_positions else actual_end

        for pos in valid_positions:
            column = [str(records[i].seq[pos]) for i in range(num_sequences)]
            for i in range(num_sequences):
                region_sequences[i].append(column[i])

        return region_sequences, region_start

    except Exception as e:
        print(f"fail ex region: {e}")
        return [], actual_end


def count_gaps_in_single_sequence(sequence, start_idx: int = 0, end_idx: int = None) -> int:

    if end_idx is None:
        end_idx = len(sequence)

    if start_idx < 0 or end_idx > len(sequence) or start_idx >= end_idx:
        return 0

    gap_count = 0
    for i in range(start_idx, end_idx):
        if sequence[i] == '-':
            gap_count += 1

    return gap_count


def count_gaps_in_region(region_sequences):

    gap_counts = []

    for sequence in region_sequences:
        gap_count = sum(1 for base in sequence if base == '-')
        gap_counts.append(gap_count)

    return int(sum(gap_counts))


def find_motif_positions(region_sequences, motif, actual_start):

    motif_length = len(motif)
    positions = {}

    for i, sequence in enumerate(region_sequences):

        seq_str = ''.join(sequence)


        pos = seq_str.find(motif)

        if pos != -1:

            absolute_pos = actual_start + pos
            positions[i] = absolute_pos
        else:
            positions[i] = -1

    return positions


def find_motif_positions_v1(single_sequence, motif, actual_start, search_type="front"):


    seq_str = ''.join(single_sequence)

    if any(char in motif for char in ['[', ']', '(', ')', '{', '}', '*', '+', '?', '.', '|']):

        matches = list(re.finditer(motif, seq_str))
        positions = [match.start() for match in matches]
    else:

        positions = []
        start = 0
        while True:
            pos = seq_str.find(motif, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
    
    if not positions:
        return -1
    

    if search_type == "front":
        match_pos = min(positions)
    else:
        match_pos = max(positions)
    

    absolute_pos = actual_start + match_pos
    return absolute_pos


def find_motif_positions_v2(region_sequences, motif, actual_start, search_type="front"):

    positions = []

    for sequence in region_sequences:
        pos = find_motif_positions_v1(sequence, motif, actual_start, search_type)
        positions.append(pos) 

    if all(pos == -1 for pos in positions):
        return -1


    position_counts = defaultdict(int)
    for pos in positions:
        position_counts[pos] += 1


    non_negative_counts = {pos: count for pos, count in position_counts.items() if pos != -1}

    if not non_negative_counts:
        return -1

    max_count = max(non_negative_counts.values())
    max_positions = [pos for pos, count in non_negative_counts.items() if count == max_count]


    if len(max_positions) == 1 and len(positions) - max_count <= 2:
        return max_positions[0]
    else:
        return -1


def find_motif_positions_v3(region_sequences, motif, actual_start, search_type="front"):

    positions = []

    for sequence in region_sequences:
        pos = find_motif_positions_v1(sequence, motif, actual_start, search_type)
        positions.append(pos)
    if len(set(positions)) == 1:
        return positions[0]
    else:
        return -1



def _get_rep_base(region_sequences, col_idx):
    col_chars = [seq[col_idx] for seq in region_sequences]

    valid_chars = []
    for c in col_chars:
        if c in "ATCG":
            valid_chars.append(c)
    
    if not valid_chars:
        return "-"

    counts = {}
    for base in valid_chars:
        counts[base] = counts.get(base, 0) + 1

    max_base = None
    max_count = 0
    for base, count in counts.items():
        if count > max_count:
            max_base = base
            max_count = count
    
    return max_base if (max_count / len(valid_chars)) > 0.5 else "-"


def _get_max_ratio(region_sequences, col_idx):

    col_chars = [seq[col_idx] for seq in region_sequences]
    

    valid_chars = []
    for c in col_chars:
        if c in "ATCG":
            valid_chars.append(c)
    
    if not valid_chars:
        return 0.0
    

    counts = {}
    for base in valid_chars:
        counts[base] = counts.get(base, 0) + 1

    max_count = 0
    for count in counts.values():
        if count > max_count:
            max_count = count
    
    return max_count / len(valid_chars)
    

def calculate_homology_scores(region_sequences):

    if not region_sequences:
        return []

    num_sequences = len(region_sequences)
    region_length = len(region_sequences[0])

    homology_scores = []

    for col_idx in range(region_length):

        column_bases = [region_sequences[seq_idx][col_idx] for seq_idx in range(num_sequences)]

        valid_bases = [base for base in column_bases if base.upper() in ['A', 'T', 'C', 'G']]

        if not valid_bases:
            homology_scores.append(0.0)
            continue

        base_counts = defaultdict(int)
        for base in valid_bases:
            base_counts[base.upper()] += 1

        max_count = max(base_counts.values())

        homology_score = max_count / num_sequences
        homology_scores.append(homology_score)

    return homology_scores


def get_bases_at_position(region_sequences, actual_start, positions):
    result = []

    for pos in positions:
        relative_pos = pos - actual_start


        if relative_pos < 0 or relative_pos >= len(region_sequences[0]):
            result.append([]) 
            continue


        bases_at_pos = [sequence[relative_pos] for sequence in region_sequences]
        result.append(bases_at_pos)

    return result



def run_trf(fasta_file, trf_path, input_dir):

    cmd = [
        trf_path,
        fasta_file,
        "2", "7", "7", "80", "10", "50", "500",
        "-f", "-d", "-m","> /dev/null 2>&1"
    ]
    
    print(f"run trf: {' '.join(cmd)}")
    
    try:

        os.system(" ".join(cmd))
        print("TRF finish")
        

        dat_file = fasta_file + ".2.7.7.80.10.50.500.dat"
        if os.path.exists(dat_file):
            return dat_file
        else:
            raise FileNotFoundError(f"TRF doesn't produce .dat: {dat_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"TRF run err: {e}")
        print(f"trf err: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"error: {e}")
        sys.exit(1)

def run_trf_simple(fasta_file, trf_path, input_dir):
    original_dir = os.getcwd()
    
    try:
        os.chdir(input_dir)
        
        fasta_abs = os.path.abspath(fasta_file)
        fasta_basename = os.path.basename(fasta_file)
        
        cmd = f'{trf_path} {fasta_abs} 2 7 7 80 10 50 500 -f -d -m > /dev/null 2>&1'
        
    

        os.system(cmd)
        
        
        
        print("TRF finish")
        

        dat_filename = f"{fasta_basename}.2.7.7.80.10.50.500.dat"
        dat_file = os.path.join(input_dir, dat_filename)
        
        if os.path.exists(dat_file):
            #print(f"trf output: {dat_file}")
            return dat_file
        else:
            for file in os.listdir('.'):
                if file.endswith('.dat') and fasta_basename in file:
                    found_file = os.path.join(input_dir, file)
                    return found_file
            
            raise FileNotFoundError(f"trf has no output")
            
    except Exception as e:
        raise
    finally:

        os.chdir(original_dir)
    
def parse_dat_file(dat_file):

    sequences_to_process = defaultdict(list)
    current_sequence = None
    
    try:
        with open(dat_file, 'r') as f:
            for line in f:
                line = line.strip()
                

                if line.startswith('Sequence:'):
                    current_sequence = line.split(':', 1)[1].strip()
                    continue

                if not line or line.startswith('Parameters:'):
                    continue
                
                fields = line.split()
                if len(fields) < 13:
                    continue
                
                try:
                    start = int(fields[0])
                    end = int(fields[1])
                    copy_number = float(fields[3])
                    consensus_size = int(fields[4])
                    entropy = float(fields[12])
                    
                    if (copy_number > 20 and consensus_size > 20 and entropy > 1.9 and current_sequence):                        
                        sequences_to_process[current_sequence].append((start, end))
                        
                except (ValueError, IndexError) as e:
                    continue
                    
    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        sys.exit(1)
    
    return sequences_to_process
    
def remove_trf(trf_path,fasta_file,output_file,input_dir):
    if not os.path.exists(fasta_file):
        print(f"err: {fasta_file} not exisit")
        sys.exit(1)
    
    #print(f"input: {fasta_file}")
    #print(f"output: {output_file}")
    #print(f"trf path: {trf_path}")
 
    
    dat_file = run_trf_simple(fasta_file, trf_path,input_dir)
    #print(f"TRF output: {dat_file}")

    #remove_html_files(fasta_file)
    

    print("parse.dat...")
    sequences_to_process = parse_dat_file(dat_file)
    
    if not sequences_to_process:
        shutil.copy2(fasta_file, output_file)
        print("finish!")
        return
    
    

    fasta_sequences = read_fasta(fasta_file)

    modified_sequences = {}
    processed_count = 0
    
    for header, sequence in fasta_sequences.items():
        matched_seq_id = None
        for seq_id in sequences_to_process:
            if seq_id in header:
                matched_seq_id = seq_id
                break
        
        if matched_seq_id:
            modified_sequence = remove_trf_regions(sequence, sequences_to_process[matched_seq_id])
            modified_sequences[header] = modified_sequence
            processed_count += 1
            

            original_len = len(sequence)
            modified_len = len(modified_sequence)
            deleted_len = original_len - modified_len
            
        else:
            modified_sequences[header] = sequence
    
    write_fasta(modified_sequences, output_file)
    
    print("finish!")

def extract_sequence_id(filename):
        base_name = os.path.basename(filename)
        clean_name = re.sub(r'(_members)?(\.fa)?(\.rdmSubset)?\.fa\.aln\.fa$', '', base_name)
        match = re.search(r'(\d+[_-])', clean_name)
        if match:
            prefix = clean_name[:match.start(1)]
            suffix = clean_name[match.start(1):]
            return prefix + suffix.replace('_', ':', 1)

    
    
def remove_no_A_T_helitron(input_dir,hle2_file):
    filter_ids = []
    window_log_dir = input_dir + "/window_log_v2"
    
    consensus_file = input_dir + "/all_consensus.fa"
    
    hle2_ids = []
    if os.path.exists(hle2_file):
      f_r = open(hle2_file,"r")
      
      for line in f_r.readlines():
          hle2_ids.append(line.strip("\n"))
      f_r.close()
    data = load_json_data(f"{window_log_dir}/stats.json")
    for aln_result in data:
        copy_num = int(aln_result["copy_num"])
        actual_start = int(aln_result["actual_start"])
        actual_end = int(aln_result["actual_end"])
        file_name = f"{input_dir}/MSA/" + aln_result["file_name"]
        if extract_sequence_id(file_name) in hle2_ids:
           continue
        records = list(SeqIO.parse(file_name, 'fasta'))
        seq_length = len(records[0].seq)
        file_id = aln_result["id"]
        region_all, _ = extract_region_from_aln(file_name, 0, seq_length - 1)
        consecutive_zero = int(aln_result["max_consecutive_zero"])
        if copy_num == 2:
            region,_ = extract_region_from_aln(file_name, 0, 50)
            region_tail,_ = extract_region_from_aln(file_name, seq_length - 49, seq_length - 1)
            region_TC,TC_region_start = extract_region_from_aln_v1(file_name, actual_start, 20)
            region_CTRR,CTRR_region_start = extract_region_from_aln_v2(file_name, actual_end , 30)
            head_gap_count = count_gaps_in_single_sequence(records[0].seq,0,actual_start) 
            tail_gap_count = count_gaps_in_single_sequence(records[0].seq,actual_end,seq_length)
                                             
            if actual_start < 50 and (seq_length - actual_end) < 50:
                    TC_start = find_motif_positions_v3(region_TC,"TC",TC_region_start,"front")
                    CTRR_start = find_motif_positions_v3(region_CTRR,"CT[AG]{2}",CTRR_region_start,'back')
                       
                    if TC_start == -1 or CTRR_start == -1:
                        continue
                    CTRR_end = CTRR_start + 4
                    TC_out_region10,_ = extract_region_from_aln(file_name, TC_start - 10, TC_start)
                    CTRR_out_region10,_ = extract_region_from_aln(file_name, CTRR_end,CTRR_end + 10)
                    if TC_start != -1 and CTRR_end != -1:
                        TC_region_sc = calculate_homology_scores(TC_out_region10)
                        CTRR_region_sc = calculate_homology_scores(CTRR_out_region10)
                        tc_np10 = np.mean(np.array(TC_region_sc))
                        ctrr_np10 = np.mean(np.array(CTRR_region_sc))
                        if tc_np10 >= 0.8 and ctrr_np10 >= 0.8:
                            filter_ids.append(aln_result["id"])
            else:
                    TC_start = find_motif_positions_v3(region_TC, "TC", TC_region_start, "front")
                    CTRR_start = find_motif_positions_v3(region_CTRR, "CT[AG]{2}", CTRR_region_start, 'back')
                    if CTRR_start == -1:
                       continue
                    CTRR_end = CTRR_start + 4
                    insertion_site = get_bases_at_position(region_all,0,[CTRR_end])[0]
                    CTRR_out_region5,_ = extract_region_from_aln(file_name, CTRR_end, CTRR_end + 5)
                    CTRR_out_region15,_ = extract_region_from_aln(file_name, CTRR_end, CTRR_end + 15)
                    CTRR_region_15 = calculate_homology_scores(CTRR_out_region15)
                    CTRR_region_5 = calculate_homology_scores(CTRR_out_region5)
                    ctrr_np5 = np.mean(np.array(CTRR_region_5))
                    ctrr_np15 = np.mean(np.array(CTRR_region_15))
                    if insertion_site != "T" and ctrr_np5 == 1 and ctrr_np15 >= 0.9:
                       if "".join(region_all[0][CTRR_start:CTRR_end])=="CTAG":
                          filter_ids.append(aln_result["id"])
                    if TC_start == -1 or CTRR_start == -1:
                       continue
                    CTRR_end = CTRR_start + 4
                    TC_end = TC_start + 2
                    insertion_list = get_bases_at_position(region_all,0,[TC_start-1, CTRR_end])
                    TC_insertion_list = insertion_list[0]
                    CTRR_insertion_list = insertion_list[1]
                    TC_out_region5,_ = extract_region_from_aln(file_name, TC_start - 5, TC_start)
                    CTRR_out_region5,_ = extract_region_from_aln(file_name, CTRR_end, CTRR_end + 5)
                    if TC_start != -1 and CTRR_end != -1:
                       TC_region_sc = calculate_homology_scores(TC_out_region5)
                       CTRR_region_sc = calculate_homology_scores(CTRR_out_region5)
                       tc_np5 = np.mean(np.array(TC_region_sc))
                       ctrr_np5 = np.mean(np.array(CTRR_region_sc))
                       if tc_np5 == 1 and ctrr_np5 == 1:
                          if len(set(TC_insertion_list)) == 1 and len(set(CTRR_insertion_list)) == 1:
                             if TC_insertion_list[0] != "A" and CTRR_insertion_list[0] != "T":
                                filter_ids.append(aln_result["id"])                 
            
        if copy_num == 5 or copy_num == 4 or copy_num == 3:
            region,_ = extract_region_from_aln(file_name, 0, 50)
            region2,_ = extract_region_from_aln(file_name, seq_length - 50, seq_length - 1)
            if count_gaps_in_region(region) <= 15:
               seq_str_0 = ''.join(region[0])
               seq_str_1 = ''.join(region[1]) 
               seq_str_2 = ''.join(region[2])
               if seq_str_0 == seq_str_1 or seq_str_1 == seq_str_2 or seq_str_0 == seq_str_2:
                  filter_ids.append(aln_result["id"])
            elif count_gaps_in_region(region2) <= 15:
               seq_str_0 = ''.join(region2[0])
               seq_str_1 = ''.join(region2[1]) 
               seq_str_2 = ''.join(region2[2])
               if seq_str_0 == seq_str_1 or seq_str_1 == seq_str_2 or seq_str_0 == seq_str_2:
                  filter_ids.append(aln_result["id"])                  
            if True:
                zero_window_count = int(aln_result["max_zero_counts"])
                consecute_zero_count = int(aln_result["max_consecutive_zero"])
                if True:
                    region_TC, TC_region_start = extract_region_from_aln_v1(file_name, actual_start, 20)
                    region_CTRR, CTRR_region_start = extract_region_from_aln_v2(file_name, actual_end, 30)
                    if find_motif_positions_v3(region_TC[0],"TC",TC_region_start,"front") != -1:
                       TC_start = find_motif_positions_v3(region_TC,"TC",TC_region_start,"front")
                    else:
                       TC_start = find_motif_positions_v2(region_TC,"TC",TC_region_start,"front")
                       
                    if find_motif_positions_v3(region_CTRR,"CT[AG]{2}",CTRR_region_start,'back') != -1:
                       CTRR_start = find_motif_positions_v3(region_CTRR,"CT[AG]{2}",CTRR_region_start,'back')
                    else:
                       CTRR_start = find_motif_positions_v2(region_CTRR,"CT[AG]{2}",CTRR_region_start,'back')
                    if TC_start == -1 or CTRR_start == -1:
                            continue
                    CTRR_end = CTRR_start + 4
                    TC_end = TC_start + 2
                    insertion_list = get_bases_at_position(region_all, 0, [TC_start-1, CTRR_end])
                    if TC_start != -1 and CTRR_end != -1:
                        TC_insertion_list = insertion_list[0]
                        CTRR_insertion_list = insertion_list[1]
                        if len(set(TC_insertion_list)) == 1 and len(set(CTRR_insertion_list)) == 1:
                            if TC_insertion_list[0] != "A" and CTRR_insertion_list[0] != "T":
                                filter_ids.append(aln_result["id"])
                        else:
                            if len(TC_insertion_list) > 2 and len(CTRR_insertion_list) > 2:
                               TC_insertion2 = [base for base in TC_insertion_list if base != '-']
                               CTRR_insertion2 = [base for base in CTRR_insertion_list if base != '-']
                               if len(TC_insertion_list) - len(TC_insertion2) <= 1 and len(CTRR_insertion_list) -len(CTRR_insertion2) <= 1:
                                  if TC_insertion2[0] != "A" and CTRR_insertion2[0] != "T": 
                                     filter_ids.append(aln_result["id"])

    return filter_ids

def parse_args():
    parser = argparse.ArgumentParser(description='Find candidate helitrons')
    parser.add_argument('--input_dir', default="./out", help='Temporary output directory')
    parser.add_argument('--threads', type=int, default=40, help='Number of threads')
    parser.add_argument('--tools_dir')
    parser.add_argument('--cur_dir')
    #parser.add_argument('--debug', type=int, default=0, help='Debug mode')
    return parser.parse_args()
    




def main():
    args = parse_args()
    cur_dir = args.cur_dir
    num_workers = args.threads
    input_dir = args.input_dir
    tools_dir = args.tools_dir
    input_fasta = input_dir + "/confident_struc_helitrons.fa"
    out_fasta = input_dir + "/confident_middle_helitrons.fa"
    final_fasta = input_dir + "/confident_final_helitrons.fa"
    stats_json = input_dir + "/window_log_v2/stats.json"
    MSA_dir = input_dir + "/MSA"
    # itrsearch and ltrsearch
    
    itr_ids = list(run_itrsearch(input_dir,tools_dir,input_fasta,cur_dir))
    ltr1_ids = list(run_ltrsearch(input_dir,tools_dir,input_fasta,cur_dir))
    ltr2_ids = ex_ltr(stats_json,MSA_dir)
    ltr_ids = list(set(ltr1_ids + ltr2_ids))
    
    
    
    itr_file = f"{input_dir}/{os.path.basename(input_fasta)}.itr"
    itr_log = f"{input_dir}/{os.path.basename(input_fasta)}.itr.log"
    ltr_file = f"{input_dir}/{os.path.basename(input_fasta)}.ltr"
    ltr_log = f"{input_dir}/{os.path.basename(input_fasta)}.ltr.log"
    #ltr_ids = []
    
    
    desc1 = os.path.join(input_dir, "desc1.txt")
    desc2 = os.path.join(input_dir, "desc2.txt")
    hle2_file = args.input_dir + "/FEMA_out/hle2_ids.txt"

    hle2_ids = []
    if os.path.exists(hle2_file):
      f_r = open(hle2_file,"r")
      
      for line in f_r.readlines():
          hle2_ids.append(line.strip("\n"))
        
    with open(desc1, 'w') as f:
        f.write("r1 s1 r1' s2\nr1 1:1 NNNNN[10]:[10]NNNNN TGCA\ns1 0 N[15]\ns2 0 NNNNNN[2]\n")
    with open(desc2, 'w') as f:
        f.write("r1 s1 r1' s2\nr1 1:1 NNNNN[10]:[10]NNNNN TGCA\ns1 0 N[7]\ns2 0 N[15]CTRR\n")
    
    records = list(SeqIO.parse(input_fasta, 'fasta'))
    total_seqs = len(records)
    
    #remove no A/T insersion site
    filter_ids = remove_no_A_T_helitron(input_dir,hle2_file)
    results = {}
    out_temp_dir = input_dir + "/temp_struc_dir2"
    os.makedirs(out_temp_dir,exist_ok=True)
    # remove FP by tirvish and other non-Helittron structure
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_sequence, record, out_temp_dir, desc1, desc2, hle2_ids): record.id
            for record in records
        }
        
        with tqdm(total=total_seqs, desc="Processing sequences") as pbar:
            for future in as_completed(futures):
                seq_id, result = future.result()
                if result:
                    results[seq_id] = result
                pbar.update(1)
    
    os.remove(desc1)
    os.remove(desc2)
    
    result_file = os.path.join(input_dir, "results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    filtered_ids = filter_results_from_json(result_file, itr_ids, ltr_ids, filter_ids)
    safe_delete(result_file)
    filtered_file = os.path.join(input_dir, "filtered_ids.txt")
    with open(filtered_file, 'w') as f:
        for seq_id in filtered_ids:
            f.write(f"{seq_id}\n")

    filtered_fasta, filtered_count = write_filtered_fasta(input_fasta, filtered_ids,out_fasta)
    input_trf_dir = input_dir + "/trf"
    os.makedirs(input_trf_dir, exist_ok=True)
    #remove trf
    remove_trf("trf",out_fasta,final_fasta,input_trf_dir)
    safe_delete(itr_file)
    safe_delete(ltr_file)
    safe_delete(itr_log)
    safe_delete(ltr_log)
    safe_delete(filtered_file)
    safe_delete(result_file)
    safe_delete(input_fasta)
    safe_delete(out_fasta)
    safe_delete(input_trf_dir)
    safe_delete(out_temp_dir)
    
    


if __name__ == "__main__":

    
    main()
    










