import os
import re
import torch
import torch.nn as nn
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import argparse

MAX_LEN = 1000
TARGET_CLASS = 1
BATCH_SIZE = 16
NUM_WORKERS = 104

class FrequencyDataset(Dataset):
    def __init__(self, input_dir):
        self.file_paths = []
        self.seq_ids = []
        self.freq_matrices = []
        self.file_names = []
        
        print(f"Scanning and processing alignment files using {NUM_WORKERS} threads...")
        aln_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.aln.fa'):
                    file_path = os.path.join(root, file)
                    aln_files.append(file_path)
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for file_path in aln_files:
                futures.append(executor.submit(self.process_file, file_path))
            
            for future in tqdm(as_completed(futures), total=len(aln_files), desc="Processing files"):
                result = future.result()
                if result is not None:
                    freq_matrix, seq_id, file_name = result
                    self.file_paths.append(file_path)
                    self.seq_ids.append(seq_id)
                    self.freq_matrices.append(freq_matrix)
                    self.file_names.append(file_name)
        
    def process_file(self, file_path):
        sequences = [str(record.seq).upper() for record in SeqIO.parse(file_path, 'fasta')]
        if len(sequences) < 2:  
            return None
            
        seq_len = len(sequences[0])
        num_seqs = len(sequences)
        
        base_freq_matrix = np.zeros((MAX_LEN, 4), dtype=np.float32)
        valid_col_indices = []
        
        # 第一遍：记录所有有效列的索引
        for col_idx in range(seq_len):
            col = [seq[col_idx] for seq in sequences]
            filtered_col = [c for c in col if c in {'A', 'T', 'C', 'G', '-'}]
            
            counts = {
                'A': filtered_col.count('A'),
                'T': filtered_col.count('T'),
                'C': filtered_col.count('C'),
                'G': filtered_col.count('G'),
            }
            total_valid = sum(counts.values())
            
            # 应用过滤规则
            original_col = [seq[col_idx] for seq in sequences]
            total_chars = len([c for c in original_col if c in {'A', 'T', 'C', 'G', 'N', '-'}])
            remove_col = False
            
            if total_chars >= 20:
                if total_valid <= int(0.1 * total_chars):
                    remove_col = True
            elif 5 < total_chars < 20:
                if total_valid <= 2:
                    remove_col = True
            elif total_chars <= 5:
                if total_valid <= 1:
                    remove_col = True
            
            if not remove_col and total_valid > 0:
                valid_col_indices.append(col_idx)
        
        if len(valid_col_indices) < 50:
            return None
        
        num_valid = len(valid_col_indices)
        if num_valid >= MAX_LEN:
            first_500_indices = valid_col_indices[:500]
            last_500_indices = valid_col_indices[-500:]
            selected_indices = first_500_indices + last_500_indices
        else:

            selected_indices = valid_col_indices

        for p_index, col_idx in enumerate(selected_indices):
            if p_index >= MAX_LEN:
                break
            
            col = [seq[col_idx] for seq in sequences]
            filtered_col = [c for c in col if c in {'A', 'T', 'C', 'G'}]
            total_valid = len(filtered_col)
            
            if total_valid > 0:
               counts = {
                   'A': filtered_col.count('A'),
                   'T': filtered_col.count('T'),
                   'C': filtered_col.count('C'),
                   'G': filtered_col.count('G'),
               }
               freqs = np.array([
                   counts['A'] / total_valid,
                   counts['T'] / total_valid,
                   counts['C'] / total_valid,
                   counts['G'] / total_valid
               ])
            

               sorted_indices = np.argsort(-freqs)
               base_freq_matrix[p_index] = freqs[sorted_indices]
        

        freq_matrix = base_freq_matrix.T
        file_name = os.path.basename(file_path)
        seq_id = self.extract_sequence_id(file_name)
        
        return freq_matrix, seq_id, file_name
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        return self.freq_matrices[idx], self.seq_ids[idx]
    
    @staticmethod
    def extract_sequence_id(filename):
        base_name = os.path.basename(filename)
        clean_name = re.sub(r'(_members)?(\.fa)?(\.rdmSubset)?\.fa\.aln\.fa$', '', base_name)
        match = re.search(r'(\d+[_-])', clean_name)
        if match:
            prefix = clean_name[:match.start(1)]
            suffix = clean_name[match.start(1):]
            return prefix + suffix.replace('_', ':', 1)
        return clean_name

class FreqCNN(nn.Module):
    def __init__(self):
        super(FreqCNN, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=30, padding=14),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=30, padding=14),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=30, padding=14),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=30, padding=14),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 1000)
            dummy_output = self.conv_blocks(dummy_input)
            self.output_size = int(torch.prod(torch.tensor(dummy_output.shape[1:])))
        

        self.classifier = nn.Sequential(
            nn.Linear(self.output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description='Find candidate helitrons')
    parser.add_argument('--input_dir')
    parser.add_argument('--input_MSA_dir')
    parser.add_argument('--model_path')
    parser.add_argument('--device')
    parser.add_argument('--threads', type=int, default=40, help='Number of threads')
    #parser.add_argument('--debug', type=int, default=0, help='Debug mode')
    return parser.parse_args()
    
def main():
    args = parse_args()
    input_dir = args.input_dir
    input_MSA_dir = args.input_MSA_dir

    model_path = args.model_path
    output_file = f"{input_dir}/class1_files.txt"
    device = torch.device(f"{args.device}")
    print(f"Using device: {device}")
    
    model = FreqCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("\nPreparing dataset...")
    dataset = FrequencyDataset(input_MSA_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class1_files = []
    stats = {
        'total': 0,
        'class1_count': 0,
    }
    initFile = args.input_dir + "/HLE_candidate.fa"
    genome_dict = SeqIO.parse(initFile, 'fasta')
    genome_dict = {k.id: k.seq.upper() for k in genome_dict}
    IdList = []
    print("\nStarting prediction...")
    with torch.no_grad():
        for batch_freq, batch_ids in tqdm(dataloader, desc="Predicting batches"):
            batch_freq = batch_freq.to(device)
            outputs = model(batch_freq)
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
            if preds.dim() == 0:  
                preds = preds.unsqueeze(0) 

            for i in range(len(batch_ids)):
                stats['total'] += 1
                if preds[i].item() == TARGET_CLASS:
                    idx = dataset.seq_ids.index(batch_ids[i])
                    IdList.append(batch_ids[i])
                    class1_files.append(dataset.file_names[idx])
                    stats['class1_count'] += 1
    f_w = open(f'{input_dir}/Intact.fa','w')
    for idx in IdList:
        if idx not in genome_dict:
           print(idx)
           continue
        predictSeq = genome_dict[idx]
        f_w.write(f">{idx}\n{predictSeq}\n")
    with open(output_file, 'w') as f:
        for file_name in class1_files:
            f.write(file_name + '\n')
    
    print(f"\nSaved {len(class1_files)} class 1 file names to {output_file}")
    print("\n=== Prediction Statistics ===")
    print(f"Total files processed: {stats['total']}")
    print(f"Predicted as class 1: {stats['class1_count']} ({stats['class1_count']/stats['total']:.2%})")


if __name__ == '__main__':
    main()


