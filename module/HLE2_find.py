import os, re, subprocess, sys, argparse, shutil, random, gc
from Bio import SeqIO
from Bio.Seq import Seq
from multiprocessing.pool import ThreadPool
from collections import defaultdict
import pybedtools as BT
from subprocess import Popen, DEVNULL

def reverse_complement(sequence):
    
    return str(Seq(sequence).reverse_complement())
        
class Homologous_search:
    def __init__(self, rep_hel_hmm, input_dir,genome,window, distance_domain, distance_na, pvalue, process_num, codetable):
        self.rep_hel_hmm = rep_hel_hmm
        self.genome = genome
        self.genome_dict = SeqIO.parse(genome, 'fasta')
        self.genome_dict = {k.id: k.seq.upper() for k in self.genome_dict}
        self.process_num = int(process_num)
        self.window = window
        self.distance_domain = distance_domain
        self.distance_na = defaultdict(lambda :int(distance_na))
        self.pvalue = float(pvalue)
        self.cutoff_flank = 0.9
        self.input_dir = input_dir
        IS1 = 0
        IS2 = 0
        self.terminalfile_dict = defaultdict(lambda: defaultdict(dict))
        self.prepair_dict = defaultdict(list)
        self.bedtoolstmp = os.path.abspath('BedtoolsTMP')
        if not os.path.exists(self.bedtoolstmp):
            os.mkdir(self.bedtoolstmp)
        BT.set_tempdir(self.bedtoolstmp)

        CWD = os.getcwd()
        self.genome_size = '%s/Genome.size' % CWD
        self.chrm_size = {i:len(self.genome_dict[i]) for i in self.genome_dict}
        genome_size = list(self.chrm_size.items())
        genome_size = sorted(genome_size, key=lambda x: x[0])
        #with open(self.genome_size, 'w') as F:
        #    F.writelines([''.join([i[0], '\t', str(i[1]), '\n']) for i in genome_size])

        ## To determine the evalue for short-sequence blastn, set the bit-score cutoff as 30, the evalue cutoff should follow the formula: m*n/(2**30)
        sum_genomesize = sum([i[1] for i in genome_size])
        self.evalue_blastn = sum_genomesize * 30 / (2 ** 32)

        # To define stem_loop structure ending with CTRR motif of Helitron
        self.CTRR_stem_loop_description = '%s/CTRR_stem_loop.descr' % CWD
        CTRR_description = """r1 s1 r1' s2\nr1 1:1 NNNNN[10]:[10]NNNNN TGCA\ns1 0 N[7]\ns2 0 N[15]CTRR%s\n"""
        # Add 'T' in the end if user limitted the 'A-T' insertion site for Helitron.
        CTRR_description = CTRR_description % 'T' if IS1 else CTRR_description % ''
        #with open(self.CTRR_stem_loop_description, 'w') as F:
        #    F.write(CTRR_description)

        # To define stem_loop structure of  HLE2
        self.subtir_description = '%s/subtir_stem_loop.descr' % CWD
        # Add 'T' in the end if user limitted the 'T-T' insertion site for HLE2.
        if IS2:
            subtir_description = """r1 s1 r1' s2\nr1 1:1 NNNNN[10]:[10]NNNNN TGCA\ns1 0 N[15]\ns2 0 NNNNN[10]T\n"""
        else:
            subtir_description = """r1 s1 r1' s2\nr1 1:1 NNNNN[10]:[10]NNNNN TGCA\ns1 0 N[15]\ns2 0 NNNNNN[2]\n"""
        #with open(self.subtir_description, 'w') as F:
        #    F.write(subtir_description)

        dbdir = 'GenomeDB/'
        if not os.path.exists(dbdir):
            os.mkdir(dbdir)
        self.genomedb = ''.join([CWD, '/', dbdir, os.path.basename(self.genome), '.blastndb'])
        makeblastndb = subprocess.Popen(['makeblastdb', '-dbtype', 'nucl', '-in', self.genome, '-out', self.genomedb],
                                        stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        makeblastndb.wait()

        ## code table
        code_table_dict = {0: (0, "Standard"), 1: (6, "Ciliate Macronuclear and Dasycladacean"),
                           2: (15, "Blepharisma Macronuclear"), 3: (22, "Scenedesmus obliquus")}
        self.codetable = code_table_dict[codetable]
        #sys.stdout.write('You are using the (%s) code to predict ORFs.\n' % self.codetable[1])
    
    def run_hmmsearch(self,hmm_model, orf_file, output_file, evalue_threshold='1e-5'):
        with Popen(['hmmsearch', '--domtblout', output_file, '--noali', '-E', evalue_threshold, hmm_model, orf_file],
               stdout=DEVNULL) as process:
             process.wait()

    def hmmsearch(self, subgenome):
        # Run hmmersearch program to search for Helitron-like transposase
        orf_file = ''.join([subgenome, '.orf'])
        hmm_opt = ''.join([subgenome, '.hmmsearch.out'])

        #The index of getorf output starts from 1, not 0
        # Use getorf to predicte open reading frames for a given genome
        command_list = ['getorf', '-sequence', subgenome, '-outseq', orf_file, '-minsize', '100']

        #os.system(" ".join(command_list))
        os.system(" ".join(command_list) + " > /dev/null 2>&1")

        Rep_dict, Hel_dict = defaultdict(list), defaultdict(list)
        Rep_opline, Hel_opline = [], []
        if os.path.getsize(orf_file):
            #run_hmmsearch = subprocess.Popen(
            #    ['hmmsearch', '--domtblout', hmm_opt, '--noali', '-E', '1e-3', self.rep_hel_hmm, orf_file],
            #    stdout=subprocess.DEVNULL)
            self.run_hmmsearch(self.rep_hel_hmm,orf_file,hmm_opt)
            #os.remove(orf_file)
        else:
            return Rep_opline, Hel_opline
        if not os.path.exists(hmm_opt):
            return Rep_opline, Hel_opline
        #print(1)
        # To parser hmmsearch output
        with open(hmm_opt, 'r') as F:
            for line in F:
                if line.startswith('#'):
                    continue
                #print(line)
                #print(2)
                splitlines = re.split('\s+', line.rstrip())
                domain, sub_class = splitlines[3].split('_')
                #print(splitlines[0])
                subchrname = "_".join(splitlines[0].split('_')[:-1])
                chrm_name, START = subchrname.split('startat')
                start, end = re.findall('\[(\d+)\s+-\s+(\d+)\]', line)[0]
                start = str(int(start) + int(START))
                end = str(int(end) + int(START))

                aa_start, aa_end = splitlines[19:21]
                score = splitlines[7]
                c_evalue, i_evalue = splitlines[11:13]
                if float(c_evalue) > 1e-5 or float(i_evalue) > 1e-5:
                    continue
                ## To transform amino acide coord to nucleotide coord
                if int(end) > int(start):
                    strand = '+'
                    nuc_start = int(aa_start) * 3 - 3 + int(start)
                    nuc_end = int(aa_end) * 3 + int(start) - 1
                    orf_loc = '-'.join([start, end])
                else:
                    nuc_end = int(start) - 3 * int(aa_start) + 3
                    nuc_start = int(start) - 3 * int(aa_end) + 1
                    strand = '-'
                    orf_loc = '-'.join([end, start])
                #print(nuc_start,nuc_end)
                
                if domain.startswith('Hel'):
                    Hel_dict[splitlines[0]].append(
                        [chrm_name, str(nuc_start), str(nuc_end), sub_class, score, strand, orf_loc])
                else:
                    Rep_dict[splitlines[0]].append(
                        [chrm_name, str(nuc_start), str(nuc_end), sub_class, score, strand, orf_loc])
        for key in Hel_dict:
            hel_candidate = sorted(Hel_dict[key], key=lambda x: float(x[4]))[-1]  ## Select the case with highest score.
            Hel_opline.append(hel_candidate)
        for key in Rep_dict:
            rep_candidate = sorted(Rep_dict[key], key=lambda x: float(x[4]))[-1]  ## Select the case with highest score.
            Rep_opline.append(rep_candidate)
        #print(Hel_opline,Rep_opline) 
        Hel_bed = BT.BedTool([BT.create_interval_from_list(line) for line in Hel_opline]).sort()
        Rep_bed = BT.BedTool([BT.create_interval_from_list(line) for line in Rep_opline]).sort()
        #print(3)
        #print(Hel_bed,Rep_bed)
        #return Rep_bed, Hel_bed
        try:
            Hel_bed = BT.BedTool([BT.create_interval_from_list(line) for line in Hel_opline]).sort()
            Rep_bed = BT.BedTool([BT.create_interval_from_list(line) for line in Rep_opline]).sort()
            #print(3)
            #print(Hel_bed,Rep_bed)
            return Rep_bed, Hel_bed
        except:
            return [], []

    def intersect(self, location1, location2, slip=0, lportion=0.0, rportion=0.0, bool_and=1):
        # Define intersect function to check if two intervals are intersected or not, similar to bedtools intersect
        location1 = sorted([int(i) for i in location1])
        location2 = sorted([int(i) for i in location2])
        if location1[0] - slip > location2[1] or location1[1] < location2[0] - slip:
            return False
        else:
            total_list = sorted([location1[0], location1[1], location2[0], location2[1]])
            portion1 = (total_list[2] - total_list[1] + 1) / (
                    location1[1] - location1[0] + 1)  ## how much proportion the intersected sequence occupiedonseq1
            portion2 = (total_list[2] - total_list[1] + 1) / (
                    location2[1] - location2[0] + 1)  ## how much proportion the intersected sequence occupiedonseq2
            if bool_and:
                if portion1 >= lportion and portion2 >= rportion:
                    return True
                else:
                    return False
            else:
                if portion1 >= lportion or portion2 >= rportion:
                    return True
                else:
                    return False

    def merge_bedfile(self, BedInput, window=1500):
        # Define function to merge two distance-close genomic features
        cluster_bed = BedInput.cluster(d=window, s=True)
        merge_dict = defaultdict(list)
        merge_list = []
        for line in cluster_bed:
            line = list(line)
            cluster = line[-1]
            merge_dict[cluster].append(line[:-1])

        for cluster in merge_dict:
            ## To record the domain location.
            coordlist = merge_dict[cluster]
            coord_set = [int(i[1]) for i in coordlist]
            coord_set.extend([int(i[2]) for i in coordlist])
            coord_set = sorted(coord_set)
            start = coord_set[0]
            stop = coord_set[-1]
            strand = coordlist[0][5]
            ## To determain sub_class, select the case with highest score
            sub_class = sorted(coordlist, key=lambda x: float(x[4]))[-1][3]
            ## To merge the ORF location
            orf_list = sorted([int(b) for i in coordlist for b in i[-1].split('-')])
            orf_coord = '-'.join([str(orf_list[0]), str(orf_list[-1])])
            chrm_id = coordlist[0][0]
            merge_list.append([chrm_id, str(start), str(stop), sub_class, orf_coord, strand])
        # merge_list = sorted(merge_list, key=lambda x: [x[0], int(x[1])])
        if merge_list:
            merge_bed = BT.BedTool([BT.create_interval_from_list(line) for line in merge_list]).sort()
            return merge_bed
        else:
            return 0

    def parser_hmmsearch(self, Rep_bed, Hel_bed, subgenome):
        # To find Rep-Hel structure which might imply a possible Helitron-like transposases.
        if not Rep_bed or not Hel_bed:
            return []
        
        ## To merge helicase or rep domain splicing sites (helitron-like transposase contain introns)
        merge_hel = self.merge_bedfile(Hel_bed, window=1500)
        merge_rep = self.merge_bedfile(Rep_bed, window=1500)
        if not merge_hel or not merge_rep:  ## Either hel or rep data is null
            return []
        
        ## To find rep and helicase gene pairs that rep is less than self.distance_domain bp upstream of hel
        joint_rephel = merge_rep.window(merge_hel, l=0, r=int(self.distance_domain), sm=True, sw=True)
        bedlist = []
        for line in joint_rephel:
            splitlines = list(line)
            strand = splitlines[5]
            chrm = splitlines[0]
            #print(chrm)
            rep_start, rep_end = splitlines[1:3]
            hel_start, hel_end = splitlines[7:9]

            # If the helicase and rep domain have a intersection, skip
            if self.intersect([int(rep_start), int(rep_end)], [int(hel_start), int(hel_end)], lportion=0.2, rportion=0.2):
                continue
            rep_orf = splitlines[4].split('-')
            hel_orf = splitlines[10].split('-')
            loc = sorted([rep_start, rep_end, hel_start, hel_end], key=lambda x: int(x))
            start, end = loc[0], loc[-1]  ## They are REP and Helicase domain region
            bedlist.append((chrm, int(start), int(end), '-'.join([rep_start, rep_end]), '-'.join([hel_start, hel_end]),
                            strand, splitlines[3], splitlines[9], 'NA'))
        bedlist = list(set(bedlist))  ## To avoid duplicates
        bedlist = sorted(bedlist, key=lambda x: [x[0], x[1]])
        return bedlist
    
    
    def intergrated_program(self, subgenome):
        # This function is used to recover terminal signals of Helitron-like elements (TIRs for HLE2; TC... motif and ...CTRR motif for Helitron)
        rep_hmmsearch_opt, hel_hmmsearch_opt = self.hmmsearch(subgenome)
        #if rep_hmmsearch_opt and hel_hmmsearch_opt:
        #   print("not NOne")
        #else:
        #   print("NOne")
        ORF_list = self.parser_hmmsearch(rep_hmmsearch_opt, hel_hmmsearch_opt, subgenome)
        sys.stdout.write('Find %s rep-hel blocks in %s.\n' % (str(len(ORF_list)), os.path.basename(subgenome).replace('.fa', '')))
        RC_total_candidate = []
        for Helitron_candidate in ORF_list:
            ORF_chrmid = Helitron_candidate[0]
            #print(ORF_chrmid)
            ORF_start = int(Helitron_candidate[1])
            ORF_stop = int(Helitron_candidate[2])
            rep_loc = Helitron_candidate[3]
            hel_loc = Helitron_candidate[4]
            strand = Helitron_candidate[5]
            original_chrid = ORF_chrmid.replace('[', ':').replace(']', '')  # CP128295.1:2911903-2925157
            left_seq = self.genome_dict[original_chrid][:50]
            right_seq = self.genome_dict[original_chrid][-50:]
            
            chrm_limit = len(self.genome_dict[original_chrid])
            rep_name, hel_name = Helitron_candidate[6:8]

            ## To decide class name
            if hel_name == 'HLE1' and rep_name == 'HLE1':
                classname = 'HLE1'
            elif hel_name == 'HLE2' and rep_name == 'HLE2':
                classname = 'HLE2'
            else:
                classname = '_or_'.join([rep_name, hel_name])
                continue
            ## To produce orf id which will be used as sole identifier of terminal signals.
            ORFID = '-'.join([ORF_chrmid, str(ORF_start), str(ORF_stop)])
            RC_total_candidate.append((original_chrid,str(ORF_start),str(ORF_stop),strand,classname,left_seq,right_seq))
         
        return RC_total_candidate

    def split_genome(self, chunk_size=200000000, flanking_size=50000, num_groups=2):
        # To split big genomes into small chunks
        if not os.path.exists(f'{self.input_dir}/genomes'):
            os.mkdir(f'{self.input_dir}/genomes')
        subgenome_list = []
        for chrm in self.genome_dict:
            seq_len = len(self.genome_dict[chrm])
            if seq_len < 1000:  ##Skip chrms whose length is shorter than 1000 bp
               # sys.stdout.write(
               #     "Chrm %s will not be used to detect autonomous HLEs as its length is shorter than 1000 bp\n" % chrm)
                continue
            subgenome_list.append((chrm, seq_len))

        num_groups = num_groups if num_groups <= len(subgenome_list) else len(subgenome_list)
        ###  To split the genomes into several files.
        # Calculate target sum for each group
        total_sum = sum([i[1] for i in subgenome_list])
        target_sum = total_sum / num_groups
        # Sort numbers in descending order
        numbers = sorted(subgenome_list, key=lambda x: -x[1])
        # Split numbers into groups with similar sums
        groups = [[] for i in range(num_groups)]
        group_sums = [0] * num_groups
        for number in numbers:
            # Find the group with the smallest current sum and add the number to it
            min_sum_index = group_sums.index(min(group_sums))
            groups[min_sum_index].append(number)
            group_sums[min_sum_index] += number[1]

        ## To split big chrms into smaller chunks.
        subgenome_list = []
        init_num = 1
        for subgroup in groups:
            subgenome = ''.join([f'{self.input_dir}/genomes/subgenome', str(init_num), '.fa'])
            init_num += 1
            with open(subgenome, 'w') as F:
                for chrminfo in subgroup:
                    chrid = chrminfo[0]
                    #print(chrid)
                    seq = self.genome_dict[chrid]
                    seq_len = len(seq)
                    num = seq_len // chunk_size
                    for i in range(num + 1):
                        start, stop = i * chunk_size, (i + 1) * chunk_size + flanking_size
                        if start >= seq_len:
                            continue
                        if stop > seq_len:
                            stop = seq_len
                        chrid_mod = chrid.split(":")
                        chrid_mod = chrid_mod[0] + '[' +chrid_mod[1] + ']'
                        subchrm = 'startat'.join([chrid_mod, str(start)])
                        #print(subchrm)
                        chunk_seq = str(self.genome_dict[chrid][start:stop])
                        F.write(''.join(['>', subchrm, '\n']))
                        F.write(chunk_seq)
                        F.write('\n')
            subgenome_list.append(subgenome)
        return subgenome_list

    def autonomous_detect(self,index):
        # main program to search for transposae and terminal signals.
        subgenome_list = self.split_genome(chunk_size=200000000, flanking_size=20000, num_groups=200)
        if len(subgenome_list) < self.process_num:
            processnum = len(subgenome_list)
        else:
            processnum = self.process_num
        sys.stdout.write('Start to search for HLE rep-hel blocks...\n')
        # Use python multiple threading
        planpool = ThreadPool(processnum)
        #processnum = processnum if self.cpu_count > processnum else self.cpu_count
        #planpool = Pool(processnum)
        run_result = []
        for subgenome in subgenome_list:
            #print(subgenome)
            run_result.append(planpool.apply_async(self.intergrated_program, args=(subgenome,)))
        planpool.close()
        planpool.join()
        Helitron_list = []
        seqId_list = []
        for result in run_result:
            result_get = result.get()
            if result_get:
                Helitron_list.extend(result_get)
        left_seqfile = f"{self.input_dir}/left_ORF_{index}.fa"
        right_seqfile = f"{self.input_dir}/right_ORF_{index}.fa"
        leftORF = open(left_seqfile,'a')
        rightORF = open(right_seqfile,'a')
        count = 0
        HLE1_count = 0
        HLE2_count = 0
        for ORF_result in Helitron_list:
            count += 1
            seqId = ORF_result[0]
            strand = ORF_result[3]
            seqId_strand = seqId + '(' + strand + ')'
            if seqId_strand not in seqId_list:
               seqId_list.append(seqId_strand)
            else:
               continue
            left_seq = ORF_result[-2].upper()
            class_name = ORF_result[-3]
            #if class_name != 'HLE2':
               #print(class_name)
            if class_name == 'HLE1':
               HLE1_count += 1
               continue
            else:
               HLE2_count += 1
            right_seq = ORF_result[-1].upper()
            left_final_seq = left_seq
            right_final_seq = right_seq
            if ORF_result[3] == '-':
               left_final_seq = reverse_complement(right_seq)
               right_final_seq = reverse_complement(left_seq)
            leftORF.write(f">{seqId_strand}\n{left_final_seq}\n")
            rightORF.write(f">{seqId_strand}\n{right_final_seq}\n")
        #print(count)
        #print(HLE1_count)
        print(HLE2_count)    
        return Helitron_list
        
def parse_args():
    parser = argparse.ArgumentParser(description='Find candidate helitrons')
    parser.add_argument('--input_dir', help='Temporary output directory')
    parser.add_argument('--RepHel')
    parser.add_argument('--genome', default="sample.fa", help='Reference genome file')
    parser.add_argument('--threads', type=int, default=40, help='Number of threads')
    parser.add_argument('--output')
    parser.add_argument('--primary_dir')
    #parser.add_argument('--debug', type=int, default=0, help='Debug mode')
    return parser.parse_args()
    
def main():
    args = parse_args()
    genome = args.genome
    input_dir = args.input_dir
    threads = args.threads
    out_file = args.output
    RepHel = args.RepHel
    primary_dir = args.primary_dir
    index = 0
    left_file_list = []
    right_file_list = []
    right_file_list = []
    index = 0
    left_file_string = None
    right_file_string = None
    genome_list = []
    for file_name in os.listdir(f"{input_dir}"):
        #print(file_name)
        detect_file = f'longest_repeats_{str(index)}.flanked.fa'
        if "flanked.fa" not in file_name:
           continue
        genome_list.append(f"{input_dir}/" + file_name)

        HomoSearcher = Homologous_search(RepHel, f"{input_dir}", f"{input_dir}/" + file_name,10000,2500,0,1e-5,40,0)
        success = HomoSearcher.autonomous_detect(index)
        if os.path.exists(f"{input_dir}/left_ORF_{index}.fa") and os.path.exists(f"{input_dir}/right_ORF_{index}.fa"):
           left_file_list.append(f"{input_dir}/left_ORF_{index}.fa")
           right_file_list.append(f"{input_dir}/right_ORF_{index}.fa")
        else:
           print("no HLE2")
        index += 1
    left_file_string = " ".join(left_file_list)
    right_file_string = " ".join(right_file_list)
    genome_string = ",".join(genome_list)
    if left_file_string and right_file_string:
       os.system(f"cat {left_file_string} > {input_dir}/left_ORF.fa")
       os.system(f"cat {right_file_string} > {input_dir}/right_ORF.fa") 
       os.system(f"python module/pair.py --input_dir {input_dir} && python module/blastn_multi_genome.py --input_dir {input_dir} --genome {genome_string} --out {out_file} --threads {threads}")
    else:
       print("no pair")
    #os.chdir(primary_dir)
    #print(f"{os.getcwd()}")
    

if __name__ == "__main__":
    main()




