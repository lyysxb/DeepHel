#!/usr/bin/env python3
import subprocess
import os
from collections import defaultdict

def load_blast_tsv(blast_output):
    hits = defaultdict(dict)
    with open(blast_output) as f:
        for line in f:
            c = line.strip().split()
            if len(c) < 14:
                continue
            qseqid, sseqid = c[0], c[1]
            if qseqid == sseqid:
                continue
            hsp = {
                "qstart":   int(c[6]),
                "qend":     int(c[7]),
                "sstart":   int(c[8]),
                "send":     int(c[9]),
                "length":   int(c[3]),
                "bitscore": float(c[11]),
                "qlen":     int(c[12]),
                "slen":     int(c[13]),
            }
            hits[qseqid].setdefault(sseqid, []).append(hsp)
    return hits

def _build_block(members):
    members = sorted(members, key=lambda h: h["qstart"])
    qlo = min(h["qstart"] for h in members)
    qhi = max(h["qend"] for h in members)
    scov = sum(abs(h["send"] - h["sstart"]) + 1 for h in members)
    return {
        "qstart": qlo,
        "qend": qhi,
        "qcov": qhi - qlo + 1,
        "scov": scov,
        "n_hsp": len(members),
        "members": members,
        "path": [(h["qstart"], h["qend"], h["sstart"], h["send"]) for h in members],
    }

def merge_hsps(hsps):
    hsps = list(hsps)
    if not hsps:
        return []
    seed = hsps[0]
    qlo, qhi = seed["qstart"], seed["qend"]
    accepted = [seed]
    for h in hsps[1:]:
        lo, hi = h["qstart"], h["qend"]
        if lo > qhi or hi < qlo:
            continue
        overlap = min(qhi, hi) - max(qlo, lo) + 1
        len_new = hi - lo + 1
        len_cur = qhi - qlo + 1
        if (overlap / len_new > 0.8
                or overlap / len_cur > 0.8):
            continue
        qlo, qhi = min(qlo, lo), max(qhi, hi)
        accepted.append(h)
    return [_build_block(accepted)]

def block_is_contained(b, qlen, slen):
    target_coverage = b["scov"] / slen if slen > 0 else 0
    boundary = any((m["sstart"] == 1 or (qlen - m["send"]) == 0) for m in b["members"])
    return (
        (b["qcov"] / qlen > 0.9)
        and boundary
        and (target_coverage < 0.25)
        and qlen > 1000
    )

def read_fasta(fasta_file):
    sequences = {}
    seq_id, seq_lines = "", []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                if seq_id:
                    sequences[seq_id] = "".join(seq_lines)
                seq_id = line[1:].strip().split()[0]
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if seq_id:
            sequences[seq_id] = "".join(seq_lines)
    return sequences

def self_blast_and_filter(fasta_file, output_file="filtered.fasta",
                          db_name="self_blast_db",
                          blast_output="blast_results.tsv",
                          run_blast=True, num_threads=4):
    if run_blast:
        subprocess.run(
            f"makeblastdb -in {fasta_file} -dbtype nucl -out {db_name} -parse_seqids",
            shell=True, check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        subprocess.run(
            f"blastn -query {fasta_file} -db {db_name} -out {blast_output} "
            f"-outfmt '6 qseqid sseqid pident length mismatch gapopen "
            f"qstart qend sstart send evalue bitscore qlen slen' "
            f"-evalue 1e-5 -num_threads {num_threads}",
            shell=True, check=True
        )
    sequences = read_fasta(fasta_file)
    to_remove = set()
    hits = load_blast_tsv(blast_output)
    for qseqid, subj_hits in hits.items():
        for sseqid, hsps in subj_hits.items():
            qlen, slen = hsps[0]["qlen"], hsps[0]["slen"]
            for b in merge_hsps(hsps):
                if block_is_contained(b, qlen, slen):
                    to_remove.add(qseqid)
                    break
    with open(output_file, "w") as out:
        for seq_id, seq in sequences.items():
            if seq_id not in to_remove:
                out.write(f">{seq_id}\n{seq}\n")

    blast_db_exts = ['.nhr', '.nin', '.nsq','.ndb','.njs','.nog','.nos','.not','.ntf','.nto']
    for ext in blast_db_exts:
        fpath = f"{db_name}{ext}"
        if os.path.exists(fpath):
            os.remove(fpath)
    if os.path.exists(blast_output):
        os.remove(blast_output)
