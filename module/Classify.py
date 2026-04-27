import torch
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import argparse
from classify_model import CombinedModel, SequenceCNN, StructureCNN
import os
import json
from tqdm import tqdm
import tempfile
import re

def extract_sequence_id(filename):
        clean_name = re.sub(r'(_members)?(\.fa)?(\.rdmSubset)?\.fa\.aln\.fa$', '', filename)
        match = re.search(r'(\d+[_-])', clean_name)
        if match:
            prefix = clean_name[:match.start(1)]
            suffix = clean_name[match.start(1):]
            return prefix + suffix.replace('_', ':', 1)
    
class FastaProcessor:
    @staticmethod
    def clean_sequence(seq, max_length=25000):
        seq = ''.join([b for b in seq.upper() if b in 'ATCG'])
        return seq[:max_length] if len(seq) > max_length else seq
    
    @staticmethod
    def reverse_complement(seq):
        return str(Seq(seq).reverse_complement())
    
    @staticmethod
    def create_combined_fasta(input_fasta,hle2_ids):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.fasta')
        
        with open(input_fasta) as f:
            for record in SeqIO.parse(f, "fasta"):
                original_seq = FastaProcessor.clean_sequence(str(record.seq)) 
                temp_file.write(f">{record.id}_original\n{original_seq}\n".encode())
                
                if extract_sequence_id(record.id) in hle2_ids:
                   revcomp_seq = FastaProcessor.reverse_complement(original_seq)
                   temp_file.write(f">{record.id}_revcomp\n{revcomp_seq}\n".encode())
        
        temp_file.close()
        return temp_file.name

base2key_map = {'A':0, 'G':1, 'T':2, 'C':3}

class HelitronAnalyzer:
    @staticmethod
    def analyze_sequences(fasta_file, output_dir,threads):

        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nRunning structure analysis on {fasta_file}...")
        try:
            from struc import analyze_helitrons
            analyze_helitrons(fasta_file, output_dir,int(threads))
            return os.path.join(output_dir, "results.json")
        except ImportError:
            print("Warning: analyze_helitrons module not found, using empty structure features")
            return None
        except Exception as e:
            print(f"Warning: Structure analysis failed: {str(e)}")
            return None

class DNAPredictor:
    def __init__(self, model_path, device):
        self.device = device
        self.model = CombinedModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def predict_sequences(self, fasta_file, structure_file=None, batch_size=32):
        """
        预测流程：
        1. 加载结构信息
        2. 进行预测
        """
        # 加载结构信息
        structure_info = {}
        if structure_file and os.path.exists(structure_file):
            print(f"Loading structure information from {structure_file}...")
            with open(structure_file, 'r') as f:
                structure_info = json.load(f)
        
        # 处理FASTA文件并进行预测
        records = []
        seq_features = []
        struct_features = []
        original_ids = []  # 记录原始ID
        
        print(f"\nProcessing FASTA file: {fasta_file}")
        for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc="Reading sequences"):
            seq_type = record.id.split('_')[-1]  # original或revcomp
            original_id = '_'.join(record.id.split('_')[:-1])  # 原始ID
            
            cleaned_seq = FastaProcessor.clean_sequence(str(record.seq))
            if len(cleaned_seq) < 200:
                continue
            
            # 获取该序列的结构信息
            seq_structure = structure_info.get(record.id, None)
            
            # 编码序列和结构特征
            seq_encoded = self._sequence_one_hot_encode(cleaned_seq)
            struct_encoded = self._structure_one_hot_encode(cleaned_seq, seq_structure)
            
            seq_features.append(seq_encoded)
            struct_features.append(struct_encoded)
            records.append(record)
            original_ids.append(original_id)
        
        if not seq_features:
            raise ValueError("No valid sequences found in the FASTA file")
        
        # 分批预测
        predictions = []
        probabilities = []
        print(f"\nMaking predictions in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(seq_features), batch_size), desc="Predicting"):
            batch_seq = seq_features[i:i+batch_size]
            batch_struct = struct_features[i:i+batch_size]
            
            seq_inputs = torch.from_numpy(np.stack(batch_seq)).to(self.device)
            struct_inputs = torch.from_numpy(np.stack(batch_struct)).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(seq_inputs.float(), struct_inputs.float())
                batch_preds = (outputs > 0.5).float()
                batch_probs = outputs.cpu().numpy().flatten()
                
            predictions.extend(batch_preds.cpu().numpy().flatten())
            probabilities.extend(batch_probs)
        
        return records, original_ids, predictions, probabilities
    
    @staticmethod
    def _sequence_one_hot_encode(seq, max_length=25000):
        encoded = np.zeros((4, max_length), dtype=np.float32)
        for i, base in enumerate(seq[:max_length]):
            encoded[base2key_map[base], i] = 1
        return encoded
    
    @staticmethod
    def _structure_one_hot_encode(seq, structure_info=None, head_len=50, tail_len=80):

        total_len = head_len + tail_len
        encoded = np.zeros((4, total_len), dtype=np.float32)
        

        if structure_info is None:
            return encoded
        
        pattern_type = structure_info.get('type', '')
        seq_len = len(seq)
        
        if pattern_type == 'pattern1':
            stemloop_start = int(structure_info.get('stemloop_start', 0)) - 1
            stemloop_end = int(structure_info.get('stemloop_end', 0)) - 1
            
            if stemloop_start >= seq_len - tail_len and stemloop_end >= seq_len - tail_len:
                start = max(0, stemloop_start - (seq_len - tail_len)) + head_len
                end = max(0, stemloop_end - (seq_len - tail_len)) + head_len
                start, end = min(start, end), max(start, end)
                if end < total_len:
                    encoded[0, start:end+1] = 1
            
            tir_left_start = int(structure_info.get('tir_left_start', 0)) - 1
            tir_left_end = int(structure_info.get('tir_left_end', 0)) - 1
            
            if tir_left_start < head_len and tir_left_end < head_len:
                start = min(tir_left_start, tir_left_end)
                end = max(tir_left_start, tir_left_end)
                encoded[0, start:end+1] = 1
            
            tir_right_start = int(structure_info.get('tir_right_start', 0)) - 1
            tir_right_end = int(structure_info.get('tir_right_end', 0)) - 1
            
            if tir_right_start >= seq_len - tail_len and tir_right_end >= seq_len - tail_len:
                start = max(0, tir_right_start - (seq_len - tail_len)) + head_len
                end = max(0, tir_right_end - (seq_len - tail_len)) + head_len
                start, end = min(start, end), max(start, end)
                if end < total_len:
                    encoded[0, start:end+1] = 1
        
        elif pattern_type == 'pattern2':
            stemloop_start = int(structure_info.get('stemloop_start', 0)) - 1
            stemloop_end = int(structure_info.get('stemloop_end', 0)) - 1
            
            if stemloop_start >= seq_len - tail_len and stemloop_end >= seq_len - tail_len:
                start = max(0, stemloop_start - (seq_len - tail_len)) + head_len
                end = max(0, stemloop_end - (seq_len - tail_len)) + head_len
                start, end = min(start, end), max(start, end)
                if end < total_len:
                    encoded[1, start:end+1] = 1
        
        elif pattern_type == 'pattern3':
            tir_left_start = int(structure_info.get('tir_left_start', 0)) - 1
            tir_left_end = int(structure_info.get('tir_left_end', 0)) - 1
            
            if tir_left_start < head_len and tir_left_end < head_len:
                start = min(tir_left_start, tir_left_end)
                end = max(tir_left_start, tir_left_end)
                encoded[2, start:end+1] = 1
            
            tir_right_start = int(structure_info.get('tir_right_start', 0)) - 1
            tir_right_end = int(structure_info.get('tir_right_end', 0)) - 1
            
            if tir_right_start >= seq_len - tail_len and tir_right_end >= seq_len - tail_len:
                start = max(0, tir_right_start - (seq_len - tail_len)) + head_len
                end = max(0, tir_right_end - (seq_len - tail_len)) + head_len
                start, end = min(start, end), max(start, end)
                if end < total_len:
                    encoded[2, start:end+1] = 1
        
        elif pattern_type == 'pattern4':

            stemloop_start = int(structure_info.get('stemloop_start', 0)) - 1
            stemloop_end = int(structure_info.get('stemloop_end', 0)) - 1
            
            if stemloop_start >= seq_len - tail_len and stemloop_end >= seq_len - tail_len:
                start = max(0, stemloop_start - (seq_len - tail_len)) + head_len
                end = max(0, stemloop_end - (seq_len - tail_len)) + head_len
                start, end = min(start, end), max(start, end)
                if end < total_len:
                    encoded[3, start:end+1] = 1
        
        return encoded


    
def parse_args():
    parser = argparse.ArgumentParser(description='Find candidate helitrons')
    parser.add_argument('--input_dir', help='Temporary output directory')
    parser.add_argument('--device')
    parser.add_argument('--model_path')
    parser.add_argument('--structure')
    parser.add_argument('--threads')
    #parser.add_argument('--debug', type=int, default=0, help='Debug mode')
    return parser.parse_args()
    
def main():
    args = parse_args()
    threads = args.threads
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    input_file = args.input_dir + "/all_consensus.fa"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input FASTA file not found: {input_file}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    input_dir = args.input_dir
    hle2_file = args.input_dir + "/FEMA_out/hle2_ids.txt"
    hle2_ids = []
    if os.path.exists(hle2_file):
       f_r = open(hle2_file,"r")
       for line in f_r.readlines():
           hle2_ids.append(line.strip("\n"))
        

    print("\nCreating combined FASTA...")
    combined_fasta = FastaProcessor.create_combined_fasta(input_file,hle2_ids)

    print("\nInitializing combined model predictor...")
    predictor = DNAPredictor(args.model_path,device)
    output_record = args.input_dir + "/predict_record.txt"


    if args.structure is None:
        print("\nRunning structure analysis on combined sequences...")
        structure_file = HelitronAnalyzer.analyze_sequences(combined_fasta,input_dir + "/temp_struc",threads)
    else:
        structure_file = args.structure

    records, original_ids, predictions, probabilities = predictor.predict_sequences(
        combined_fasta,
        structure_file=structure_file,
        batch_size=32
    )

    results = {}
    for record, original_id, pred, prob in zip(records, original_ids, predictions, probabilities):
        seq_type = record.id.split('_')[-1]  
        if original_id not in results:
            results[original_id] = {'original': None, 'revcomp': None}
        
        if seq_type == 'original':
            results[original_id]['original'] = {
                'record': record,
                'prediction': pred,
                'probability': prob
            }
        else:
            results[original_id]['revcomp'] = {
                'record': record,
                'prediction': pred,
                'probability': prob
            }
    
    # 保存结果
    print(f"\nSaving results to {output_record}...")
    with open(output_record, 'w') as f:
        f.write("SequenceID\tPrediction\tProbability\tSequenceType\n")
        for original_id, variants in results.items():
            original = variants['original']
            revcomp = variants['revcomp']
            if original and original['prediction'] == 1:
                f.write(f"{original_id}\t1\t{original['probability']:.4f}\toriginal\n")
            elif revcomp and revcomp['prediction'] == 1:
                f.write(f"{original_id}\t1\t{revcomp['probability']:.4f}\treverse_complement\n")
            else:
                if original:
                    f.write(f"{original_id}\t0\t{original['probability']:.4f}\toriginal\n")
                if revcomp:
                    f.write(f"{original_id}\t0\t{revcomp['probability']:.4f}\treverse_complement\n")
    
    helitron_output = input_dir + "/confident_struc_helitrons.fa"
    helitrons = []
    
    for original_id, variants in results.items():
        original = variants.get('original')
        revcomp = variants.get('revcomp')
        
        if original and original['prediction'] == 1:
            record = original['record']
            record.id = original_id 
            record.description = ""
            helitrons.append(record)
        elif revcomp and revcomp['prediction'] == 1:
            record = revcomp['record']
            record.id = original_id
            record.description = ""
            helitrons.append(record)
    
    if helitrons:
        with open(helitron_output, 'w') as f:
            SeqIO.write(helitrons, f, 'fasta')
        print(f"Saved {len(helitrons)} predicted Helitrons to {helitron_output}")
    else:
        print("No sequences predicted as Helitron")
    
    total_original = len([v for v in results.values() if v['original']])
    total_revcomp = len([v for v in results.values() if v['revcomp']])
    helitron_count = len(helitrons)
    
    print("\nPrediction Summary:")
    #print(f"Total original sequences processed: {total_original}")
    #print(f"Total reverse complement sequences processed: {total_revcomp}")
    print(f"Predicted Helitrons: {helitron_count}")
    
    os.remove(combined_fasta)

if __name__ == '__main__':
    main()

