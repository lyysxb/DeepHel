import re
import sys
input_dir = sys.argv[1]
def read_fasta_ids(file_path, id_type):
    raw_id_list = []
    std_set = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('>'):
                continue
            raw_id = line[1:].strip()

            if id_type == 'HLE1':
                sid = raw_id
                raw_id_list.append(sid)
                std_set.add(sid)

            elif id_type == 'HLE2':
                sid = raw_id.split()[0]
                raw_id_list.append(sid)
                std_set.add(sid)

            elif id_type == 'hle_small':
                if "_members.fa.rdmSubset.fa.aln.fa" in raw_id:
                   core_part = raw_id.replace('_members.fa.rdmSubset.fa.aln.fa', '')
                else:
                   core_part = raw_id.replace('_members.fa.aln.fa', '')
                last_underscore_idx = core_part.rfind('_')
                if last_underscore_idx != -1:
                    chr_part = core_part[:last_underscore_idx]
                    pos_part = core_part[last_underscore_idx+1:]
                    sid = f"{chr_part}:{pos_part}"
                    raw_id_list.append(sid)
                    std_set.add(sid)
    return raw_id_list, std_set


hle1_raw_list, hle1_set = read_fasta_ids(f'{input_dir}/HLE1.fa', id_type='HLE1')
hle2_raw_list, hle2_set = read_fasta_ids(f'{input_dir}/HLE2.fa', id_type='HLE2')
_, small_hle1_set = read_fasta_ids(f'{input_dir}/hle1.fa', id_type='hle_small')
_, small_hle2_set = read_fasta_ids(f'{input_dir}/hle2.fa', id_type='hle_small')

common_hle_set = hle1_set & hle2_set

common_hle1_cnt = sum(1 for x in hle1_raw_list if x in common_hle_set)
common_hle2_cnt = sum(1 for x in hle2_raw_list if x in common_hle_set)

hle1_total_raw = len(hle1_raw_list)
hle2_total_raw = len(hle2_raw_list)

hle1_unique = hle1_total_raw - common_hle1_cnt
hle2_unique = hle2_total_raw - common_hle2_cnt

hle1_count = len(small_hle1_set)
hle2_count = len(small_hle2_set) - len(small_hle2_set & common_hle_set)
unclassified_final = len(small_hle2_set & common_hle_set)

print("=" * 55)
print("Statistics for HLE1.fa and HLE2.fa")
print(f"HLE1.fa unique count(without common): {hle1_unique}")
print(f"HLE2.fa unique count(without common): {hle2_unique}")
print(f"Common unique id count: {len(common_hle_set)}")
print(f"Common occurrence in HLE1.fa: {common_hle1_cnt}")
print(f"Common occurrence in HLE2.fa: {common_hle2_cnt}")
print("=" * 55)
print("unclassfied HLE", common_hle_set)
print("hle1 count", hle1_count)
print("hle2 count", hle2_count)
print("unclassifed HLE final", unclassified_final)
