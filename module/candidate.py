import argparse
import re
import subprocess
import datetime
import json
import os
import sys
import time
from multiprocessing import cpu_count
from candidate_Util import Logger, file_exist, read_fasta

def parse_args():
    parser = argparse.ArgumentParser(description='Find candidate helitrons')
    parser.add_argument('--tmp_output_dir', default="./ory_out", help='Temporary output directory')
    parser.add_argument('--reference', default="./data/ory_sativa.fa", help='Reference genome file')
    parser.add_argument('--flanking_len', type=int, default=0, help='Flanking length')
    parser.add_argument('--fixed_extend_base_threshold', type=int, default=4000, 
                       help='Fixed extend base threshold')
    parser.add_argument('--threads', type=int, default=40, help='Number of threads')
    parser.add_argument('--recover', type=int, default=0, help='Recover mode')
    parser.add_argument('--debug', type=int, default=0, help='Debug mode')
    parser.add_argument('--chrom_seg_length', type=int, default=100000, 
                       help='Chromosome segment length')
    parser.add_argument('--chunk_size', type=int, default=400, help='Chunk size')
    parser.add_argument('--max_repeat_len', type=int, default=30000, 
                       help='Maximum repeat length')
    parser.add_argument('--tandem_region_cutoff', type=float, default=0.5, 
                       help='Tandem region cutoff')
    return parser.parse_args()

def main():
    args = parse_args()
    
    tmp_output_dir = args.tmp_output_dir
    reference = args.reference
    flanking_len = args.flanking_len
    fixed_extend_base_threshold = args.fixed_extend_base_threshold
    threads = args.threads
    recover = args.recover
    debug = args.debug
    chrom_seg_length = args.chrom_seg_length
    chunk_size = args.chunk_size
    is_recover = bool(recover)
    max_repeat_len = args.max_repeat_len
    tandem_region_cutoff = args.tandem_region_cutoff

    log = Logger(tmp_output_dir+'/HiTE.log', level='debug')
    starttime = time.time()
    log.logger.info('Start step2.0: Splitting genome assembly into chunks')
    split_genome_command = 'python module/split_genome_chunks.py -g ' \
                         + reference + ' --tmp_output_dir ' + tmp_output_dir \
                         + ' --chrom_seg_length ' + str(chrom_seg_length) + ' --chunk_size ' + str(chunk_size)
    log.logger.info(split_genome_command)
    os.system(split_genome_command)
    endtime = time.time()
    dtime = endtime - starttime
    log.logger.info("Running time of step2.0: %.8s s" % (dtime))

    reg_str = 'genome.cut(\d+).fa$'
    cut_references = []
    for filename in os.listdir(tmp_output_dir):
        match = re.search(reg_str, filename)
        if match:
            ref_index = match.group(1)
            cut_references.append((ref_index, tmp_output_dir + '/' + filename))

    # Using identified TEs to mask the genome in order to reduce computational load in all-vs-all alignments.
    prev_TE = tmp_output_dir + '/prev_TE.fa'
    # if os.path.exists(confident_ltr_cut_path):
    #     os.system('cat ' + confident_ltr_cut_path + ' > ' + prev_TE)
    # if curated_lib is not None and os.path.exists(curated_lib):
    #     os.system('cat ' + curated_lib + ' >> ' + prev_TE)
    # The outcomes of homologous methods can only serve as supplementary information and should not be used as masks,
    # as this could potentially obscure many genuine non-LTR local masks, rendering the de novo method unable to identify them.
    # os.system('cat ' + confident_other_path + ' >> ' + prev_TE)

    split_ref_dir = tmp_output_dir + '/ref_chr'
    for cut_reference_item in cut_references:
        ref_index = cut_reference_item[0]
        cut_reference = cut_reference_item[1]
        log.logger.info('Current chunk: ' + str(ref_index))

        longest_repeats_flanked_path = tmp_output_dir + '/longest_repeats_' + str(ref_index) + '.flanked.fa'
        longest_repeats_path = tmp_output_dir + '/longest_repeats_' + str(ref_index) + '.fa'
        resut_file = longest_repeats_path
        if not is_recover or not file_exist(resut_file) or not file_exist(longest_repeats_flanked_path):
                starttime = time.time()
                log.logger.info('Start 2.1: Coarse-grained boundary mapping')
                coarse_boundary_command = 'python module/coarse_boundary.py ' \
                                       + ' -g ' + cut_reference + ' --tmp_output_dir ' + tmp_output_dir \
                                       + ' --prev_TE ' + str(prev_TE) \
                                       + ' --fixed_extend_base_threshold ' + str(fixed_extend_base_threshold) \
                                       + ' --max_repeat_len ' + str(max_repeat_len) \
                                       + ' --thread ' + str(threads) \
                                       + ' --flanking_len ' + str(flanking_len) \
                                       + ' --tandem_region_cutoff ' + str(tandem_region_cutoff) \
                                       + ' --ref_index ' + str(ref_index) \
                                       + ' -r ' + reference + ' --recover ' + str(recover) \
                                       + ' --debug ' + str(debug)
                log.logger.info(coarse_boundary_command)
                os.system(coarse_boundary_command)
                endtime = time.time()
                dtime = endtime - starttime
                log.logger.info("Running time of step2.1: %.8s s" % (dtime))
        else:
            log.logger.info(resut_file + ' exists, skip...')

        longest_repeats_flanked_path = tmp_output_dir + '/longest_repeats_' + str(ref_index) + '.flanked.fa'
        resut_file = tmp_output_dir + '/confident_tir_'+str(ref_index)+'.fa'

if __name__ == "__main__":
    main()

