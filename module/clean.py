#!/usr/bin/env python3
from Bio import SeqIO
import sys
import re

def main():
    if len(sys.argv) != 3:
        sys.exit(1)

    input_file, output_file = sys.argv[1], sys.argv[2]

    with open(input_file) as input_handle:
        records = list(SeqIO.parse(input_handle, "fasta"))

        for record in records:
            record.description = ""
            if '_members.fa.rdmSubset.fa.aln.fa' in record.id:
                 record.id = record.id.split('_members.fa.rdmSubset.fa.aln.fa')[0]
            else:
                 record.id = record.id.split('_members.fa.aln.fa')[0]  # 简化ID

        with open(output_file, "w") as output_handle:
            SeqIO.write(records, output_handle, "fasta")

    print(f"clean {len(records)} sequences")

if __name__ == "__main__":
    main()

