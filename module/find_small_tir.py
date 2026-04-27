import os
def find_common_substrings_dp(seq1, seq2, min_length):

    len1, len2 = len(seq1), len(seq2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    common_substrings = []
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] >= min_length:
                    start1 = i - dp[i][j]
                    end1 = i - 1
                    start2 = j - dp[i][j]
                    end2 = j - 1
                    common_substrings.append({
                        'substring': seq1[start1:i],
                        'seq1_start': start1,
                        'seq1_end': end1,
                        'seq2_start': start2,
                        'seq2_end': end2
                    })
            else:
                dp[i][j] = 0

    filtered_substrings = []
    substring_map = {}

    for cs in common_substrings:
        substring = cs['substring']
        start1 = cs['seq1_start']
        end2 = cs['seq2_end']
        start2 = cs['seq2_start']
        if start1 <=1 and start2 <=1:
            if substring not in substring_map:
                substring_map[substring] = cs
            else:

                existing = substring_map[substring]
                if start1 < existing['seq1_start']:
                    existing['seq1_start'] = start1
                    existing['seq1_end'] = cs['seq1_end']
                if end2 > existing['seq2_end']:
                    existing['seq2_start'] = cs['seq2_start']
                    existing['seq2_end'] = end2

    filtered_substrings = list(substring_map.values())
    return filtered_substrings
#print(find_common_substrings_dp("ATCGTAGTAAAA","AAAAATCGTA",3))
