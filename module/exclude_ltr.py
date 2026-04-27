import json
import os
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict

def parse_sequence_id(seq_id: str) -> Dict[str, str]:
    pattern = r'(.+):(\d+)-(\d+)\(([+-])\)'
    match = re.match(pattern, seq_id)
    if match:
        return {
            'gene_id': match.group(1),
            'start': int(match.group(2)),
            'end': int(match.group(3)),
            'strand': match.group(4)
        }
    else:
        raise ValueError(f"can't parse: {seq_id}")

def analyze_file(file_path):

    sequences = []
    current_seq_id = None
    current_seq = ""
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq_id:
                    sequences.append(parse_sequence_id(current_seq_id))
                current_seq_id = line[1:] 
                current_seq = ""
            else:
                current_seq += line

        if current_seq_id:
            sequences.append(parse_sequence_id(current_seq_id))
    
    if not sequences:
        return 0, 0
    

    groups = defaultdict(list)
    for seq in sequences:
        key = (seq['gene_id'], seq['strand'])
        groups[key].append(seq)

    for key in groups:
        groups[key].sort(key=lambda x: x['start'])


    pairs = set()
    used_sequences = set()
    
    for key, seq_list in groups.items():
        i = 0

        while i < len(seq_list) - 1: 
            current_item = seq_list[i]
            next_item = seq_list[i + 1]
            
           
            distance = next_item['start'] - current_item['end']

            if 100 < distance < 20000:

                pair_id1 = f"{current_item['gene_id']}:{current_item['start']}-{current_item['end']}({current_item['strand']})"
                pair_id2 = f"{next_item['gene_id']}:{next_item['start']}-{next_item['end']}({next_item['strand']})"
                pairs.add((pair_id1, pair_id2))
                used_sequences.add(i)
                used_sequences.add(i + 1)
                i += 2  
            else:
                i += 1
    
    total_sequences = len(sequences)
    pair_count = len(pairs)
    
    return pair_count, total_sequences


def ex_ltr(json_file_path: str, folder_path: str):
    """主函数"""
    ltr_ids = []

    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    

    filtered_items = [item for item in json_data if item.get('copy_num', 0) > 5]
    


    results = []
    
    for item in filtered_items:
        file_name = item['file_name']
        file_path = os.path.join(folder_path, file_name)
        
        if not os.path.exists(file_path):
        
            continue
        

        try:
            pair_count, total_sequences = analyze_file(file_path)
            
            
            if total_sequences > 0:
                ratio = (pair_count * 2) / total_sequences
                
                if ratio >= 0.8:  # 80%
                    results.append({
                        'id': item['id'],
                        'file_name': file_name,
                        'pair_count': pair_count,
                        'total_sequences': total_sequences,
                        'ratio': ratio
                    })
                    ltr_ids.append(item['id'])


                
        except Exception as e:
            print(f"fail read {e}")

    return ltr_ids

if __name__ == "__main__":
    json_file_path = "crassgigas_stats.json"
    folder_path = "crassgigas"
    
    ltr_ids = ex_ltr(json_file_path, folder_path)

