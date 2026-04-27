import argparse
import re
import subprocess
import datetime
import json
import os
import sys
import time
from multiprocessing import cpu_count


def parse_args():
    parser = argparse.ArgumentParser(description='Find candidate helitrons')
    parser.add_argument('--input_dir')
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = args.input_dir
    

    genome_fa = os.path.join(input_dir, "genome.fa")
    fa_files = "confident_final_helitrons.fa"
    

    input_fa = os.path.join(input_dir, fa_files)
    cl_output = os.path.join(input_dir, "cl.fa")
    

    file_size_kb = os.path.getsize(genome_fa) / 1024
    c_value = "0.9"
    cmd = [
        "cd-hit-est",
        "-i", input_fa,
        "-o", cl_output,
        "-c", c_value,
        "-aL", "0.95",
        "-aS", "0.95",
        "-d", "0",
        #"-T", str(cpu_count())
    ]
    
    subprocess.run(cmd, check=True)
    
    print("cluster finish")

if __name__ == "__main__":
    main()


