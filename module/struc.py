#!/usr/bin/env python3


import os
import re
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

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
        return stem_loop_loc

    def inverted_detection(self, sequencefile, minitirlen, maxtirlen, mintirdist, maxtirdist, seed, find_type, left_n,right_n):
        output_dir = os.path.dirname(sequencefile)
        basename = os.path.basename(sequencefile)
        
        dbname = os.path.join(output_dir, f"{basename}.invdb")
        invttirfile = os.path.join(output_dir, f"{basename}.inv.txt")
        
        # 构建数据库
        mkinvdb = subprocess.Popen(
            ['gt', 'suffixerator', '-db', sequencefile, '-indexname', dbname, 
             '-mirrored', '-dna', '-suf', '-lcp', '-bck'],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        mkinvdb.wait()
        
        # 运行tirvish
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
                        
                        if right_start >= right_n and left_end <= left_n and invt_length_left >= 12 and invt_length_right >= 12 and invt_length_left <= 20 and invt_length_right <= 20:
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
                        if left_start >= right_n and invt_length_left >= 9 and invt_length_right >= 9 and invt_length_left <= 20 and invt_length_right <= 20 and float(sim) >= 85:
                            invt_list.append([
                                chrmid, 
                                str(left_start), 
                                str(left_end),
                                str(right_start),
                                str(right_end),
                                left_expand,
                                right_expand,
                                (invt_length_right + invt_length_left)/2, 
                                'pattern2',
                                sim
                             ])
        
        invt_list = sorted(invt_list, key=lambda x: int(x[1]))
        #if 
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

def process_single_sequence(record, output_dir, desc1, desc2,desc3):
    result = {}
    pattern2_result = None
    strLen = len(clean_sequence(str(record.seq)))
    tail50bp_file = create_tail_50bp_sequence(record, output_dir)
    searcher_tail = StructureSearch(tail50bp_file)
    stem_loop_result2 = searcher_tail.stem_loop(desc2, 50,output_dir)
    if stem_loop_result2:
        stem_loop_end_dis = 50 - int(stem_loop_result2[0][2])
        stem_loop_start_dis = 50 - int(stem_loop_result2[0][1])
        if stem_loop_end_dis < 5:
            pattern2_result = {
                 'type': 'pattern2',
                 'position': f"{stem_loop_result2[0][1]}-{stem_loop_result2[0][2]}",
                 'stemloop_start':str(strLen - stem_loop_start_dis),
                 'stemloop_end':str(strLen - stem_loop_end_dis),
                 'stem_len': stem_loop_result2[0][3],
                 'loop_len': stem_loop_result2[0][4],
                 'source': 'tail50bp'
             }
            os.remove(tail50bp_file)
            return (record.id, pattern2_result)
    

    stem_loop_result1 = searcher_tail.stem_loop(desc1, 50, output_dir)

    tir_test_file = create_tir_test_sequence(record, output_dir)
    tir_searcher = StructureSearch(tir_test_file)
    tir_result = tir_searcher.inverted_detection(
        tir_test_file,
        minitirlen=5,
        maxtirlen=50,
        mintirdist=10,
        maxtirdist=140,
        seed=8,
        find_type="pattern2",
        left_n = 49,
        right_n = 59
    )

    if stem_loop_result1 and tir_result:
        stem_loop_end_dis = 50 - int(stem_loop_result1[0][2])
        stem_loop_start_dis = 50 - int(stem_loop_result1[0][1])
        tir_right_end_dis = 140 - int(tir_result[0][4])
        dis_loop_tir = abs(stem_loop_end_dis-tir_right_end_dis)
        actual_tir_left_start = int(tir_result[0][1])
        actual_tir_left_end = int(tir_result[0][2])
        actual_tir_right_start = strLen -(140 - int(tir_result[0][3]))
        actual_tir_right_end = strLen - (140 - int(tir_result[0][4]))
        if dis_loop_tir > 10 and actual_tir_left_start < 5 and tir_right_end_dis > 15: 
            result = {
                'type': 'pattern1',
                'stemloop': f"{stem_loop_result1[0][1]}-{stem_loop_result1[0][2]}",
                'stemloop_start':str(strLen - stem_loop_start_dis),
                'stemloop_end':str(strLen - stem_loop_end_dis),
                'tir': f"{tir_result[0][1]}-{tir_result[0][4]}",
                'tir_left_start': str(actual_tir_left_start),
                'tir_left_end': str(actual_tir_left_end),
                'tir_right_start':str(actual_tir_right_start),
                'tir_right_end': str(actual_tir_right_end),
                'tir_similarity': tir_result[0][8],
                'source': 'tail50bp_for_stemloop + extended_for_TIR'
            }
            os.remove(tail50bp_file)
            os.remove(tir_test_file)
            return (record.id, result)
    if tir_result:

        if tir_result[0][-2] == 'pattern1':
           tir_right_end_dis = 140 - int(tir_result[0][4])
           actual_tir_left_start = int(tir_result[0][1])
           actual_tir_left_end = int(tir_result[0][2])
           actual_tir_right_start = strLen -(140 - int(tir_result[0][3]))
           actual_tir_right_end = strLen - (140 - int(tir_result[0][4]))
           if actual_tir_left_start < 5 and tir_right_end_dis > 20:
               result = {
                   'type': 'pattern3',
                   'tir': f"{tir_result[0][1]}-{tir_result[0][4]}",
                   'tir_left_start': str(actual_tir_left_start),
                   'tir_left_end': str(actual_tir_left_end),
                   'tir_right_start':str(actual_tir_right_start),
                   'tir_right_end': str(actual_tir_right_end),
                   'tir_similarity': tir_result[0][9],
                   'source': 'extended_for_TIR'
               }
               os.remove(tail50bp_file)
               os.remove(tir_test_file)
               return (record.id, result)
    stem_loop_result3 = searcher_tail.stem_loop(desc3, 50, output_dir)
    if stem_loop_result3:
        stem_loop_end_dis = 50 - int(stem_loop_result3[0][2])
        stem_loop_start_dis = 50 - int(stem_loop_result3[0][1])
        if stem_loop_end_dis < 5 and 20 >= int(stem_loop_start_dis) - int(stem_loop_end_dis) >= 16:
           result = {
                     'type': 'pattern4',
                     'position':f"{stem_loop_result3[0][1]}-{stem_loop_result3[0][2]}",
                     'stemloop_start':str(strLen - stem_loop_start_dis),
                     'stemloop_end':str(strLen - stem_loop_end_dis),
                     'stem_len': stem_loop_result3[0][3],
                     'loop_len': stem_loop_result3[0][4],
                     'source': 'tail50bp'
                  }
           return (record.id, result)

    for f in [tail50bp_file, tir_test_file]:
        if os.path.exists(f):
            os.remove(f)
    
    return (record.id, result) if result else (record.id, None)

def analyze_helitrons(input_fasta, output_dir, num_workers=48):
    os.makedirs(output_dir, exist_ok=True)

    desc1 = os.path.join(output_dir, "desc1.txt")
    desc2 = os.path.join(output_dir, "desc2.txt")
    desc3 = os.path.join(output_dir, "desc3.txt") 
    with open(desc1, 'w') as f:
        f.write("r1 s1 r1' s2\nr1 1:1 NNNNN[10]:[10]NNNNN TGCA\ns1 0 N[15]\ns2 0 NNNNNN[2]\n")
    with open(desc2, 'w') as f:
        f.write("r1 s1 r1' s2\nr1 1:1 NNNNN[10]:[10]NNNNN TGCA\ns1 0 N[7]\ns2 0 N[15]CTRR\n")
    with open(desc3, 'w') as f:
        f.write("r1 s1 r1' s2\nr1 0:0 NNNNN[10]:[10]NNNNN TGCA\ns1 0 N[15]\ns2 0 NNNNNN[2]\n") 

    records = list(SeqIO.parse(input_fasta, 'fasta'))
    total_seqs = len(records)
    
    print(f"Starting analysis of {total_seqs} sequences with {num_workers} workers...")
    
    results = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        futures = {
            executor.submit(process_single_sequence, record, output_dir, desc1, desc2,desc3): record.id
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
    

    result_file = os.path.join(output_dir, "results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    total = len(results)
    p1 = sum(1 for r in results.values() if r['type'] == 'pattern1')
    p2 = sum(1 for r in results.values() if r['type'] == 'pattern2')
    p3 = sum(1 for r in results.values() if r['type'] == 'pattern3')
    p4 = sum(1 for r in results.values() if r['type'] == 'pattern4')
    print("\n=== Analysis Summary ===")
    print(f"Total sequences processed: {total_seqs}")
    print(f"Pattern1 (stemloop+TIR) hits: {p1}")
    print(f"Pattern2 (stemloop only) hits: {p2}")
    print(f"Pattern3 (TIR only) hits: {p3}")
    print(f"Pattern4 (stemloop2 only) hits: {p4}")
    print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Helitron analyse")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default="helitron_results2")
    parser.add_argument("-w", "--workers", type=int, default=4)
    
    args = parser.parse_args()
    analyze_helitrons(args.input, args.output, args.workers)




