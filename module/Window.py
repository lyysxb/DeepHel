import os
import re
import numpy as np
import torch
import torch.nn as nn
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import json
import argparse


class Config:
    window_size = 50                      # window size
    slide_step = 10                       # slide step
    num_copies = 100                      # max copies
    input_channels = 5                    # A,T,C,G,- 

    cnn_channels = [32, 64, 128]       
    lstm_hidden = 64
    lstm_layers = 2
    dropout = 0.3
    

processing_stats = {
    'total_processed': 0,
    'kept': 0,
}


BASE_ENCODING = {
    'A': [1, 0, 0, 0, 0],
    'T': [0, 1, 0, 0, 0],
    'C': [0, 0, 1, 0, 0],
    'G': [0, 0, 0, 1, 0],
    '-': [0, 0, 0, 0, 1],
    'N': [0, 0, 0, 0, 0],
    'default': [0, 0, 0, 0, 0]
}

class SequenceLabeler(nn.Module):
    def __init__(self):
        super(SequenceLabeler, self).__init__()
        
        # CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(Config.input_channels, Config.cnn_channels[0], 
                     kernel_size=3, padding=1),
            nn.BatchNorm1d(Config.cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(Config.dropout),
            
            nn.Conv1d(Config.cnn_channels[0], Config.cnn_channels[1], 
                     kernel_size=3, padding=1),
            nn.BatchNorm1d(Config.cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(Config.dropout),
            
            nn.Conv1d(Config.cnn_channels[1], Config.cnn_channels[2], 
                     kernel_size=3, padding=1),
            nn.BatchNorm1d(Config.cnn_channels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=Config.cnn_channels[-1],
            hidden_size=Config.lstm_hidden,
            num_layers=Config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=Config.dropout if Config.lstm_layers > 1 else 0
        )
        
        # FC
        self.fc = nn.Sequential(
            nn.Linear(Config.lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        

        x = x.view(batch_size * seq_len, Config.num_copies, Config.input_channels)
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x).squeeze(-1)
        

        lstm_in = cnn_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(lstm_in)
        

        output = self.fc(lstm_out).squeeze(-1)
        return output


def load_class1_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

    
def preprocess_sequence(sequence):
    processed = []
    for char in sequence:
        upper_char = char.upper()
        if upper_char in BASE_ENCODING:
            processed.append(upper_char)
        else:
            processed.append('N')
    return processed

def extract_sequence_id(filename):
    base_name = os.path.basename(filename)
    return base_name

def process_alignment_file(file_path):
    records = list(SeqIO.parse(file_path, 'fasta'))
    sequences = [preprocess_sequence(str(record.seq)) for record in records]
    
    if not sequences or len(sequences) == 1:
        return None
    
    seq_len = len(sequences[0])
    num_seqs = len(sequences)

    kept_columns = []
    valid_col_indices = []
    column_shapes = []  
    
    current_invalid_shape = None
    invalid_count = 0
    invalid_indices = set()  
    
    for col_idx in range(seq_len):
        col = [seq[col_idx] for seq in sequences]
        filtered_col = [c for c in col if c in {'A', 'T', 'C', 'G'}]
        total_chars = len([c for c in col if c in {'A', 'T', 'C', 'G', 'N', '-'}])
        total_valid = len(filtered_col)
        

        shape = [0] * num_seqs
        for i, char in enumerate(col):
            if char in {'A', 'T', 'C', 'G'}:
                shape[i] = 1        
        remove_col = False        
        if total_chars > 10:
            if total_valid <= int(0.1 * total_chars):
                remove_col = True
        elif total_chars <= 10:
            if total_valid == 1:
                if current_invalid_shape is None:
                    current_invalid_shape = shape
                    invalid_count = 1
                elif shape == current_invalid_shape:
                    invalid_count += 1
                    if invalid_count >= 5:
                        for k in range(max(0, col_idx - invalid_count + 1), col_idx + 1):
                            invalid_indices.add(k)
                else:
                    current_invalid_shape = None
                    invalid_count = 0
        
        if not remove_col and total_valid > 0 and col_idx not in invalid_indices:
            kept_columns.append(col)
            valid_col_indices.append(col_idx)
            column_shapes.append(shape)
    
    if len(kept_columns) < Config.window_size:
        return None
    
    return {
        'columns': kept_columns,
        'valid_indices': valid_col_indices,
        'original_sequences': sequences, 
        'file_name': os.path.basename(file_path),
        'num_copies': len(records)
    }

def generate_forward_windows(processed_data):
    #generate 5' slide windows
    columns = processed_data['columns']
    valid_indices = processed_data['valid_indices']
    num_columns = len(columns)
    
    windows = []
    window_info = []
    
    for i in range(20):
        start = i * Config.slide_step
        end = start + Config.window_size
        if end > num_columns:
            break
        
        # # encoding matrix (window_size x num_copies x input_channels)
        encoding_matrix = np.zeros((Config.window_size, Config.num_copies, 
                                  Config.input_channels), dtype=np.float32)
        for row in range(Config.window_size):
            for col in range(Config.num_copies):
                if col < len(columns[start + row]):
                    char = columns[start + row][col]
                    encoding_matrix[row, col] = BASE_ENCODING.get(char, BASE_ENCODING['default'])
        
        windows.append(encoding_matrix)
        window_info.append({
            'type': 'forward',
            'slide_num': i + 1,
            'valid_start': valid_indices[start],
            'valid_end': valid_indices[start + Config.window_size - 1],
            'actual_start': start,
            'actual_end': start + Config.window_size - 1
        })
    
    return np.array(windows), window_info

def generate_backward_windows(processed_data):
    #generate 3' slide windows
    columns = processed_data['columns']
    valid_indices = processed_data['valid_indices']
    num_columns = len(columns)
    
    windows = []
    window_info = []
    
    for i in range(20):
        end = num_columns - 1 - i * Config.slide_step
        start = end - Config.window_size + 1
        if start < 0:
            break
        
        # encoding matrix
        encoding_matrix = np.zeros((Config.window_size, Config.num_copies, 
                                  Config.input_channels), dtype=np.float32)
        for row in range(Config.window_size):
            for col in range(Config.num_copies):
                if col < len(columns[start + row]):
                    char = columns[start + row][col]
                    encoding_matrix[row, col] = BASE_ENCODING.get(char, BASE_ENCODING['default'])
        
        windows.append(encoding_matrix)
        window_info.append({
            'type': 'backward',
            'slide_num': i + 1,
            'valid_start': valid_indices[start],
            'valid_end': valid_indices[start + Config.window_size - 1],
            'actual_start': start,
            'actual_end': start + Config.window_size - 1
        })
    
    return np.array(windows), window_info

def generate_middle_windows(processed_data, start_idx, end_idx):
    #generate middle windows
    columns = processed_data['columns']
    valid_indices = processed_data['valid_indices']
    
    if start_idx is None or end_idx is None or start_idx >= end_idx:
        return None, None
    
    region_length = end_idx - start_idx + 1
    if region_length < Config.window_size:
        return None, None
    
    windows = []
    window_info = []
    
    
    num_windows = region_length // Config.window_size
    
    for i in range(num_windows):
        start = start_idx + i * Config.window_size
        end = start + Config.window_size - 1
        if end > end_idx:
            break
        
        
        encoding_matrix = np.zeros((Config.window_size, Config.num_copies, 
                                  Config.input_channels), dtype=np.float32)
        for row in range(Config.window_size):
            for col in range(Config.num_copies):
                if col < len(columns[start + row]):
                    char = columns[start + row][col]
                    encoding_matrix[row, col] = BASE_ENCODING.get(char, BASE_ENCODING['default'])
        
        windows.append(encoding_matrix)
        window_info.append({
            'type': 'middle',
            'window_num': i + 1,
            'valid_start': valid_indices[start],
            'valid_end': valid_indices[end],
            'actual_start': start,
            'actual_end': end
        })
    
    return np.array(windows), window_info


def find_homology_boundaries(columns, valid_indices, forward_predictions, backward_predictions, filename, homo_col_threshold, edge_threshold, homo_region_threshold):
    num_columns = len(columns)
    slide_step = Config.slide_step
    window_size = Config.window_size
    copy_num = len(columns[0]) if columns else 0
    
   
    start_homo_score = None
    end_homo_score = None
    
    
    candidate_start_idx = None
    candidate_end_idx = None

    all_forward_high_ones = True
    for i, window_preds in enumerate(forward_predictions):
        ones_ratio = np.mean(window_preds == 1)
        if ones_ratio < 0.95:  
            all_forward_high_ones = False
            break
    
    if all_forward_high_ones and len(forward_predictions) > 0:
        discard_reason = "Start position too close to beginning (1-5bp)"
        return None, None, discard_reason, start_homo_score, end_homo_score, candidate_start_idx, candidate_end_idx
    
    
    all_backward_high_ones = True
    for i, window_preds in enumerate(backward_predictions):
        ones_ratio = np.mean(window_preds == 1)
        if ones_ratio < 0.95:  
            all_backward_high_ones = False
            break
    
    if all_backward_high_ones and len(backward_predictions) > 0:
        discard_reason = "End position too close to beginning (1-5bp)"
        return None, None, discard_reason, start_homo_score, end_homo_score, candidate_start_idx, candidate_end_idx

    
    start_idx = None
    discard_reason = None
    
    #scanning from 5' ends
    for i in range(len(forward_predictions)):
        window_preds = forward_predictions[i]
        window_mean = np.mean(window_preds)
        
        
        condition1 = window_mean > 0.45
        
        
        condition2 = False
        consecutive_ones = 0
        for pred in window_preds:
            if pred == 1:
                consecutive_ones += 1
                if consecutive_ones >= 15:
                    condition2 = True
                    break
            else:
                consecutive_ones = 0
        # find the first predicted 1 label colum process
        if condition1 or condition2:
            window_start = i * slide_step
            
            
            transitions = []
            zero_count = 0
            one_count = 0
            transition_pos = -1
            
            for j in range(1, window_size):
                #find 0->1 
                if window_preds[j-1] == 0 and window_preds[j] == 1:
                    
                    transition_pos = j
                    
                    left_zero = 0
                    k = j - 1
                    while k >= 0 and window_preds[k] == 0:
                        left_zero += 1
                        k -= 1
                    
                    right_one = 0
                    k = j
                    while k < window_size and window_preds[k] == 1:
                        right_one += 1
                        k += 1
                    transitions.append({
                        'pos': j,
                        'left_zero': left_zero,
                        'right_one': right_one
                    })
            
        
            candidate_pos = None
            if transitions:
                
                strong_transitions = [t for t in transitions if t['left_zero'] >= 10]
                if strong_transitions:
                    strong_transitions.sort(key=lambda x: -x['pos'])  # if there has strong evidence ,namely having enough zero before transition point
                    candidate_pos = strong_transitions[0]['pos'] # then we'll find the first predicted 1 label colum.
                else:
                    
                    for j in range(window_size):
                        if window_preds[j] == 1:
                            candidate_pos = j   # find the first predicted 1 label colum
                            break
            
            if candidate_pos is not None:
                candidate_start = window_start + candidate_pos
                candidate_start_idx = candidate_start  
                
                
                if candidate_start <= edge_threshold:
                    discard_reason = "Start position too close to beginning (1-5bp)"
                    return None, None, discard_reason, start_homo_score, end_homo_score, candidate_start_idx, candidate_end_idx
                

                start_idx = candidate_start
                break
            
    
    
    end_idx = None
    
    #scanning from 3' ends
    for i in range(len(backward_predictions)):
        window_preds = backward_predictions[i]
        window_mean = np.mean(window_preds)
        
        
        condition1 = window_mean > 0.45
        
        
        condition2 = False
        consecutive_ones = 0
        for pred in window_preds:
            if pred == 1:
                consecutive_ones += 1
                if consecutive_ones >= 15:
                    condition2 = True
                    break
            else:
                consecutive_ones = 0
        
        if condition1 or condition2:
            window_start = num_columns - 1 - i * slide_step - window_size + 1
            
            
            transitions = []
            zero_count = 0
            one_count = 0
            transition_pos = -1
            # find the last predicted 1 label colum process
            for j in range(1, window_size):
                if window_preds[j-1] == 1 and window_preds[j] == 0:
                    
                    transition_pos = j-1
                    
                    left_one = 0
                    k = j - 1
                    while k >= 0 and window_preds[k] == 1:
                        left_one += 1
                        k -= 1
                    
                    right_zero = 0
                    k = j
                    while k < window_size and window_preds[k] == 0:
                        right_zero += 1
                        k += 1
                    transitions.append({
                        'pos': j-1,
                        'left_one': left_one,
                        'right_zero': right_zero
                    })
            
       
            candidate_pos = None
            if transitions:
                
                strong_transitions = [t for t in transitions if t['right_zero'] >= 10]
                if strong_transitions:
                    strong_transitions.sort(key=lambda x: x['pos']) # if there has strong evidence ,namely having enough zero after transition point
                    candidate_pos = strong_transitions[0]['pos']   # then we'll find the last predicted 1 label colum.
                else:
                    
                    for j in reversed(range(window_size)):
                        if window_preds[j] == 1:
                            candidate_pos = j     # find the first predicted 1 label colum
                            break
            
            if candidate_pos is not None:
                candidate_end = window_start + candidate_pos
                candidate_end_idx = candidate_end  
                
                
                if candidate_end >= num_columns - 1 - edge_threshold:
                    discard_reason = "End position too close to end (1-5bp)"
                    return None, None, discard_reason, start_homo_score, end_homo_score, candidate_start_idx, candidate_end_idx
            
                end_idx = candidate_end
                break

                
    
    
    if start_idx is not None and end_idx is not None:
        
        while start_idx > 0:
            prev_col = columns[start_idx - 1]
            copy_num = len(prev_col)
            filtered_col = [c for c in prev_col if c in {'A', 'T', 'C', 'G'}]
            total_valid = len(filtered_col)
            
            counts = {
                'A': filtered_col.count('A'),
                'T': filtered_col.count('T'),
                'C': filtered_col.count('C'),
                'G': filtered_col.count('G')
            }
            max_count = max(counts.values())
            max_ratio = max_count / total_valid
            prev_start_idx = start_idx - 14
            if prev_start_idx < 0:
               prev_start_idx = 0
                   
            
            if max_ratio > 0.9 and total_valid >= 10:
                start_idx -= 1
            elif float(total_valid/copy_num) > 0.7 and copy_num > 45 and max_ratio >= 0.75:
                start_idx -= 1
            elif copy_num > 25 and start_idx >= 5 and calculate_homology_score_for_region(columns, prev_start_idx, start_idx, homo_col_threshold)[1] > 0.85:
                #print(prev_start_idx, start_idx,calculate_homology_score_for_region(columns, prev_start_idx, start_idx, homo_col_threshold)[1])
                start_idx -= 1
            elif max_ratio == 1 and total_valid > 5:
                start_idx -= 1
            else:
                break
        

        while end_idx < num_columns - 1:
            next_col = columns[end_idx + 1]
            filtered_col = [c for c in next_col if c in {'A', 'T', 'C', 'G'}]
            total_valid = len(filtered_col)
                
            counts = {
                'A': filtered_col.count('A'),
                'T': filtered_col.count('T'),
                'C': filtered_col.count('C'),
                'G': filtered_col.count('G')
            }
            max_count = max(counts.values())
            max_ratio = max_count / total_valid
            forward_end_idx = end_idx + 15
            if forward_end_idx > num_columns:
               forward_end_idx = num_columns - 1
            
            if max_ratio > 0.9 and total_valid >= 10:
                end_idx += 1
            elif float(total_valid/copy_num) > 0.7 and copy_num > 45 and max_ratio >= 0.75:
                end_idx += 1
            elif copy_num > 25 and end_idx <= num_columns - 5 and calculate_homology_score_for_region(columns, end_idx, forward_end_idx, homo_col_threshold)[1] > 0.85:
                end_idx += 1
            elif max_ratio == 1 and total_valid > 5:
                end_idx += 1
            else:
                break
        #print(start_idx,end_idx,edge_threshold)        
        if end_idx >= num_columns - 1 - edge_threshold:
           discard_reason = "End position too close to end (1-5bp)"
           candidate_end_idx = end_idx
           return None, None, discard_reason, start_homo_score, end_homo_score, candidate_start_idx, candidate_end_idx
        if start_idx <= edge_threshold:
           discard_reason = "Start position too close to beginning (1-5bp)"
           candidate_start_idx = start_idx
           return None, None, discard_reason, start_homo_score, end_homo_score, candidate_start_idx, candidate_end_idx
    
    return start_idx, end_idx, discard_reason, start_homo_score, end_homo_score, candidate_start_idx, candidate_end_idx

def calculate_homology_score_for_region(columns, start, end, homo_col_threshold):
    total_homo = 0
    total_cols = 0
    count_list = []
    for col in columns[start:end+1]:
        filtered_col = [c for c in col if c in {'A', 'T', 'C', 'G'}]
        if not filtered_col:
            continue
        
        total_chars = len(filtered_col)
        count_list.append(total_chars)
        if total_chars > 5:
            homo_threshold = homo_col_threshold
        elif 2 < total_chars <= 5:
            homo_threshold = 0.75
        elif total_chars == 2:
            homo_threshold = 0.95
        else:  # total_chars == 1
            continue
        
        counts = {
            'A': filtered_col.count('A'),
            'T': filtered_col.count('T'),
            'C': filtered_col.count('C'),
            'G': filtered_col.count('G')
        }
        max_count = max(counts.values())
        ratio = max_count / len(filtered_col)
        
        if ratio >= homo_threshold:
            total_homo += 1
        total_cols += 1
    
    return count_list, total_homo / total_cols if total_cols > 0 else 0
    
def calculate_homology_score_for_region_v2(columns, start, end, homo_col_threshold):
    total_homo = 0
    total_cols = 0
    count_list = []
    ratio_list = []
    for col in columns[start:end+1]:
        filtered_col = [c for c in col if c in {'A', 'T', 'C', 'G'}]
        if not filtered_col:
            continue
        
        total_chars = len(filtered_col)
        count_list.append(total_chars)
        
        if total_chars > 5:
            homo_threshold = homo_col_threshold
        elif 2 < total_chars <= 5:
            homo_threshold = 0.75
        elif total_chars == 2:
            homo_threshold = 0.95
        else:  # total_chars == 1
            continue
        
        counts = {
            'A': filtered_col.count('A'),
            'T': filtered_col.count('T'),
            'C': filtered_col.count('C'),
            'G': filtered_col.count('G')
        }
        max_count = max(counts.values())
        ratio = max_count / len(filtered_col)
        ratio_list.append(ratio)
        #if ratio >= homo_threshold:
        #    total_homo += 1
        #total_cols += 1
    
    return count_list, sum(ratio_list) / len(ratio_list) if ratio_list else 0

def check_middle_region_v4(columns, start_idx, end_idx, middle_predictions_len, copy_num, seq_id, homo_col_threshold, homo_window_threshold, middle_zero_threshold):
    con_zero = 0
    zero_window_count = 0
    total_windows = middle_predictions_len
    current_consecutive_zero = 0
    max_consecutive_zero = 0
    for i in range(total_windows):
        window_start = start_idx + i * Config.window_size
        window_end = window_start + Config.window_size - 1
        count_list, homo_score = calculate_homology_score_for_region(columns, window_start, window_end, homo_col_threshold)
        total_chars = len(columns[window_start])
        if total_chars > 5:
            threshold = homo_window_threshold
        elif 2 < total_chars <= 5:
            threshold = 0.75
        else: 
            threshold = 0.8
        
        if homo_score <= threshold:
            zero_window_count += 1
            current_consecutive_zero += 1
            if current_consecutive_zero > max_consecutive_zero:
                max_consecutive_zero = current_consecutive_zero
        else:
            current_consecutive_zero = 0
    if zero_window_count >= 80:
       total_windows = 200 #when internal length exceeds 10000 ,we limit the max internal windows to better filter false positive
    if zero_window_count >= total_windows * middle_zero_threshold:
        return False, f"Too many zero windows in middle region: {zero_window_count} > {total_windows * middle_zero_threshold}", zero_window_count, total_windows, max_consecutive_zero

    return True, None, zero_window_count, total_windows, max_consecutive_zero  


def generate_consensus_sequence_from_original(processed_data, start_idx, end_idx):
    if start_idx is None or end_idx is None or start_idx >= end_idx:
        return None

    valid_indices = processed_data['valid_indices']
    original_start = valid_indices[start_idx]
    original_end = valid_indices[end_idx]
    
    original_sequences = processed_data['original_sequences']
    copy_num = len(original_sequences)
    consensus = []
    

    for col_idx in range(original_start, original_end + 1):
        col = [seq[col_idx] for seq in original_sequences]
        
      
        base_map = defaultdict(int)
        valid_bases = [] 
        
        for base in col:
            if base in {'A', 'T', 'C', 'G', '-'}:
                base_map[base] += 1
                if base != '-': 
                    valid_bases.append(base)

        if not valid_bases:
            continue
        
        gap_ratio = base_map['-'] / copy_num if '-' in base_map else 0
        

        valid_base_count = len(valid_bases)
        max_valid_count = 0
        max_valid_base = ''
        
        for base in {'A', 'T', 'C', 'G'}:
            if base in base_map and base_map[base] > max_valid_count:
                max_valid_count = base_map[base]
                max_valid_base = base
        
        max_valid_ratio = max_valid_count / valid_base_count if valid_base_count > 0 else 0
        
        max_base_count = 0
        max_base = ''
        
        for base, count in base_map.items():
            if count > max_base_count:
                max_base_count = count
                max_base = base
        
        max_base_ratio = max_base_count / copy_num
        

        if gap_ratio >= 0.5:
            #if copy_num >= 45 and col_idx < 500 and max_valid_ratio>=0.95 and valid_base_count > 5:
            #    consensus.append(max_valid_base)
            #    continue
            if copy_num >= 10 and gap_ratio >= 0.65:
                continue
            if max_valid_ratio > 0.75 and max_valid_count != 1:
                consensus.append(max_valid_base)
        else:
            if max_base_ratio > 0.5:
                consensus.append(max_base)
            elif max_valid_ratio > 0.75:
                consensus.append(max_valid_base)
            else:
                consensus.append('N')
    
    return ''.join(consensus) if consensus else None

class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx])

def predict_windows(model, windows, device):
    if windows is None or len(windows) == 0:
        return None
    
    dataset = WindowDataset(windows)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            preds = (outputs > 0.5).int().cpu().numpy()
            all_preds.append(preds)
    
    return np.concatenate(all_preds, axis=0)


def extract_sequence_from_intact(intact_file, seq_id):
    intact_path = intact_file
    if not os.path.exists(intact_path):
        print(f"  Warning: Intact file not found at {intact_path}")
        return None, 0
    
    try:
        base_name = seq_id
        match = re.search(r'([^_]+)_(\d+-\d+)_', base_name)
        if match:
            chromosome = match.group(1)
            coords = match.group(2)
            intact_seq_id = f"{chromosome}:{coords}"
        else:
            intact_seq_id = base_name.replace('_members.fa.rdmSubset.fa.aln.fa', '').replace('_members.fa.aln.fa', '')
            intact_seq_id = intact_seq_id.replace('_', ':')
        
        for record in SeqIO.parse(intact_path, 'fasta'):
            if record.id == intact_seq_id or intact_seq_id in record.id:
                return str(record.seq), len(record.seq)
        
        print(f"  Warning: Sequence {intact_seq_id} not found in {intact_path}")
        return None, 0
    except Exception as e:
        print(f"  Error extracting sequence from Intact.fa: {e}")
        return None, 0

def save_results(window_log_dir, all_consensus_file, filtered_consensus_file, 
                processed_data, start_idx, end_idx, candidate_start_idx, candidate_end_idx, middle_preds_len, 
                forward_info, backward_info, middle_info, forward_preds, backward_preds, 
                discard_reason, copy_num, seq_id, homo_col_threshold, homo_window_threshold, 
                middle_zero_threshold, start_homo_score, end_homo_score, intact_file):

    base_name = os.path.splitext(processed_data['file_name'])[0]
    output_filename = f"{base_name}_copies{processed_data['num_copies']}.txt"
    os.makedirs(window_log_dir, exist_ok=True)
    os.makedirs(window_log_dir + "/consensus_log", exist_ok=True)
    

    output_file = os.path.join(window_log_dir, "consensus_log", output_filename)
    with open(output_file, 'w') as f:
        f.write("Window\tType\tSlide#\tValidStart\tValidEnd\tActualStart\tActualEnd\tPredictions\n")
        
        for i, (info, pred) in enumerate(zip(forward_info, forward_preds)):
            pred_str = ','.join(map(str, pred))
            f.write(f"{i+1}\t{info['type']}\t{info['slide_num']}\t")
            f.write(f"{info['valid_start']}\t{info['valid_end']}\t")
            f.write(f"{info['actual_start']}\t{info['actual_end']}\t")
            f.write(f"{pred_str}\n")
        

        for i, (info, pred) in enumerate(zip(backward_info, backward_preds), 
                                       start=len(forward_info)):
            pred_str = ','.join(map(str, pred))
            f.write(f"{i+1}\t{info['type']}\t{info['slide_num']}\t")
            f.write(f"{info['valid_start']}\t{info['valid_end']}\t")
            f.write(f"{info['actual_start']}\t{info['actual_end']}\t")
            f.write(f"{pred_str}\n")


    final_start_idx = start_idx if start_idx is not None else candidate_start_idx
    final_end_idx = end_idx if end_idx is not None else candidate_end_idx
    

    result_data = {
        'start': start_idx,
        'end': end_idx,
        'should_filter': False, 
        'filter_reason': discard_reason, 
        'consensus': None,
        'intact_sequence': None, 
        'intact_seq_length': 0, 
        'zero_window_counts': 0,  
        'total_windows': 0, 
        'max_consecute_windows': 0, 
        'homo_col_threshold': homo_col_threshold,
        'homo_window_threshold': homo_window_threshold,
        'middle_zero_threshold': middle_zero_threshold
    }
    

    consensus_seq = None
    if start_idx is not None and end_idx is not None:
        consensus_seq = generate_consensus_sequence_from_original(processed_data, start_idx, end_idx)
        result_data['consensus'] = consensus_seq

    if start_idx is None or end_idx is None or consensus_seq is None:
        result_data['should_filter'] = True
        if discard_reason:
            result_data['filter_reason'] = discard_reason
        else:
            result_data['filter_reason'] = "Failed to generate consensus sequence"

        if (discard_reason is not None and intact_file):
            is_filtered_by_missing_boundary = (final_start_idx is None or final_end_idx is None)
            if is_filtered_by_missing_boundary:
                intact_seq, intact_seq_length = extract_sequence_from_intact(intact_file, processed_data['file_name'])
                if intact_seq:
                    result_data['intact_sequence'] = intact_seq
                    result_data['intact_seq_length'] = intact_seq_length
                    result_data['filter_reason'] = f"Missing boundary: {discard_reason} (using intact sequence)"
                    

        
        return result_data, discard_reason
    

    middle_valid, middle_discard_reason, zero_window_count, total_windows_count, max_consecutive_window = check_middle_region_v4(
        processed_data['columns'], final_start_idx, final_end_idx, middle_preds_len, copy_num, seq_id, homo_col_threshold, homo_window_threshold, middle_zero_threshold
    )
    

    result_data.update({
        'zero_window_counts': zero_window_count,
        'total_windows': total_windows_count,
        'max_consecute_windows': max_consecutive_window
    })
    
    if not middle_valid:
        result_data['should_filter'] = True
        result_data['filter_reason'] = middle_discard_reason
        intact_seq, intact_seq_length = extract_sequence_from_intact(intact_file, processed_data['file_name'])
        if intact_seq:
           result_data['intact_sequence'] = intact_seq
           result_data['intact_seq_length'] = intact_seq_length
        return result_data, middle_discard_reason
    

    with open(all_consensus_file, 'a') as f:
        f.write(f">{processed_data['file_name']}\n")
        f.write(f"{consensus_seq}\n")
    
    
    return result_data, None

def save_filtered_consensus(filtered_consensus_file, filename, consensus_seq, filter_reason, sequence_source="Consensus"):
    if consensus_seq:
        with open(filtered_consensus_file, 'a') as f:
            if sequence_source == "Intact.fa":
                f.write(f">{filename} [Filtered: {filter_reason}, Source: Intact.fa]\n")
            else:
                f.write(f">{filename} [Filtered: {filter_reason}]\n")
            f.write(f"{consensus_seq}\n")
        return True
    return False

def record_discarded_sequence(window_log_dir, discard_file, filename, reason):
    with open(os.path.join(window_log_dir, discard_file), 'a') as f:
        f.write(f"{filename}\t{reason}\n")

def save_processing_stats(window_log_dir, stats_file):
    with open(os.path.join(window_log_dir, stats_file), 'w') as f:
        f.write("Processing Statistics:\n")
        f.write(f"Total processed: {processing_stats['total_processed']}\n")
        f.write(f"Kept: {processing_stats['kept']}\n")

def save_kept_results_to_json(window_log_dir, kept_stats_json, kept_result_stats):
    json_file = os.path.join(window_log_dir, kept_stats_json)
    
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    
    serializable_stats = []
    for stat in kept_result_stats:
        serializable_stat = {}
        for key, value in stat.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                serializable_stat[key] = value
            else:
                serializable_stat[key] = str(value)
        serializable_stats.append(serializable_stat)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    
    print(f"Kept sequences statistics saved to: {json_file}")
    
        
def parse_args():
    parser = argparse.ArgumentParser(description='Find candidate helitrons')
    parser.add_argument('--input_dir', help='Temporary output directory')
    parser.add_argument('--device')
    parser.add_argument('--model_path')
    parser.add_argument('--homo_col_threshold')
    parser.add_argument('--homo_window_threshold')
    parser.add_argument('--edge_threshold')
    parser.add_argument('--middle_zero_threshold')
    parser.add_argument('--homo_region_threshold')
    return parser.parse_args()

# ==================== 主流程 ====================
def main():
    args = parse_args()


    window_log_dir = args.input_dir + "/window_log_v2"
    os.makedirs(window_log_dir, exist_ok=True)
    discard_file = "discarded_sequences.txt"
    with open(os.path.join(window_log_dir, discard_file), 'w') as f:
        f.write("Filename\tReason\n")

    
    filtered_consensus_file = os.path.join(window_log_dir,"filtered_consensus.fa")

    for file_path in [filtered_consensus_file]:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    input_dir = args.input_dir
    all_consensus_file = args.input_dir + "/all_consensus.fa"
    intact_file = os.path.join(args.input_dir, "Intact.fa")
    kept_stats_json = "stats.json"
    model_path = args.model_path
    homo_col_threshold = float(args.homo_col_threshold)
    homo_window_threshold = float(args.homo_window_threshold)
    edge_threshold = int(args.edge_threshold)
    middle_zero_threshold = float(args.middle_zero_threshold)
    homo_region_threshold = float(args.homo_region_threshold)

    if not os.path.exists(intact_file):
        print(f"Warning: Intact.fa file not found at {intact_file}")
        intact_file = None
    else:
        print(f"Found Intact.fa file at {intact_file}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SequenceLabeler().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")
    
    class1_file = input_dir + "/class1_files.txt"
    
    input_MSA_dir = input_dir + "/MSA"

    class1_files = load_class1_list(class1_file)
    print(f"Found {len(class1_files)} files to process")
    processing_stats['total_processed'] = len(class1_files)
    stats_file = "processing_stats.txt"
    kept_result_stats = []
    
    for filename in class1_files:
        file_path = input_MSA_dir + '/' + filename
        if not os.path.exists(file_path):
            print(f"Warning: File {filename} not found, skipping")
            continue
        
        print(f"\nProcessing: {filename}")
        
        processed_data = process_alignment_file(file_path)
        if processed_data is None:
            print("  Failed to process file (invalid data)")
            record_discarded_sequence(window_log_dir, discard_file, filename, "Invalid data")
            continue
        
        print(f"  Processed {len(processed_data['columns'])} valid columns from {processed_data['num_copies']} copies")
        
        forward_windows, forward_info = generate_forward_windows(processed_data)
        backward_windows, backward_info = generate_backward_windows(processed_data)
        print(f"  Generated {len(forward_windows)} forward and {len(backward_windows)} backward windows")
        
        forward_preds = predict_windows(model, forward_windows, device)
        backward_preds = predict_windows(model, backward_windows, device)
        
        start_idx, end_idx, discard_reason, start_homo_score, end_homo_score, candidate_start_idx, candidate_end_idx = find_homology_boundaries(
            processed_data['columns'], 
            processed_data['valid_indices'],
            forward_preds, 
            backward_preds,
            filename,
            homo_col_threshold,
            edge_threshold,
            homo_region_threshold
        )
        
        
        middle_preds = None
        middle_info = None
        middle_preds_len = 0 
        if (start_idx is not None and end_idx is not None) or (candidate_start_idx is not None and candidate_end_idx is not None):
            final_start_idx = start_idx if start_idx is not None else candidate_start_idx
            final_end_idx = end_idx if end_idx is not None else candidate_end_idx
            if final_start_idx is not None and final_end_idx is not None:
                middle_windows, middle_info = generate_middle_windows(processed_data, final_start_idx, final_end_idx)
                middle_preds_len = len(middle_windows) if middle_windows is not None else 0
        
        result, discard_reason = save_results(
            window_log_dir, all_consensus_file, filtered_consensus_file,
            processed_data, start_idx, end_idx, candidate_start_idx, candidate_end_idx, middle_preds_len,
            forward_info, backward_info, middle_info, forward_preds, backward_preds, 
            discard_reason, processed_data['num_copies'], file_path, homo_col_threshold, 
            homo_window_threshold, middle_zero_threshold, start_homo_score, end_homo_score, intact_file
        )
        
        zero_window_count = result.get('zero_window_counts', 0) if result else 0
        
        #print(result)

        if result is not None and result.get('should_filter', False):
            filter_reason = result.get('filter_reason', discard_reason)
            if result.get('intact_sequence'):
                filter_reason = result.get('filter_reason', discard_reason)
                if save_filtered_consensus(filtered_consensus_file, filename, result['intact_sequence'], filter_reason, "Intact.fa"):
                    print(f"  Saved filtered consensus sequence from Intact.fa (length={result['intact_seq_length']})")
            elif result.get('consensus'):
                filter_reason = result.get('filter_reason', discard_reason)
                if save_filtered_consensus(filtered_consensus_file, filename, result['consensus'], filter_reason, "Consensus"):
                    print(f"  Saved filtered consensus sequence (length={len(result['consensus'])})")
            
            #if "high homology score" in filter_reason.lower():
            #    processing_stats['discarded_homo_score'] += 1
            #elif "edge position" in filter_reason.lower():
            #    processing_stats['discarded_edge'] += 1
            #elif "middle zero" in filter_reason.lower():
            #    processing_stats['discarded_middle_zero'] += 1
   
            
            print(f"  Discarded sequence: {filter_reason}")
            record_discarded_sequence(window_log_dir, discard_file, filename, filter_reason)
        elif result is not None and not result.get('should_filter', False):
            print(f"  Found homology region: {result['start']}-{result['end']} (length={len(result['consensus'])})")
            print(f"  Consensus sequence length: {len(result['consensus'])}")
            processing_stats['kept'] += 1
            file_id = extract_sequence_id(filename)
            kept_result_stats.append({
                "id": file_id,
                "file_name": filename,
                "copy_num": processed_data['num_copies'],
                "max_zero_counts": result['zero_window_counts'],
                "total_zeros": result['total_windows'],
                "max_consecutive_zero": result['max_consecute_windows'],
                "actual_start": processed_data['valid_indices'][result['start']],
                "actual_end": processed_data['valid_indices'][result['end']]
            })
        else:
            if discard_reason:
            
                #if "edge position" in discard_reason.lower():
                #    processing_stats['discarded_edge'] += 1
                #elif "middle zero" in discard_reason.lower():
                #    processing_stats['discarded_middle_zero'] += 1
                print(f"  Discarded sequence: {discard_reason}")
                record_discarded_sequence(window_log_dir, discard_file, filename, discard_reason)
            else:
                record_discarded_sequence(window_log_dir, discard_file, filename, "No valid homology region")
    
    save_processing_stats(window_log_dir, stats_file)
    save_kept_results_to_json(window_log_dir, kept_stats_json, kept_result_stats)
    
    print("\nProcessing completed. Statistics:")
    print(f"Total processed: {processing_stats['total_processed']}")
    print(f"Kept: {processing_stats['kept']}")

if __name__ == '__main__':
    main()

