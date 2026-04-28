# DeepHel

DeepHel: a multi-stage deep learning framework for accurate Heltiron annotation

## Installation

```bash
git clone https://github.com/lyysxb/DeepHel.git
cd DeepHel
conda env create -f environment.yml
```

## Usage

Example: Running a test on Arabidopsis thaliana

```bash
conda activate DeepHel
python main.py --out_dir ./athaliana --genome ./data/athaliana.fa --threads 48
```

**Parameter**

`--out_dir`: The directory where output files will be saved.

`--genome`: Path of the input genome file in fasta format

`--threads`: Number of CPU threads.

## Output Files

| File Name                      | Description                                         |
| :----------------------------- | :-------------------------------------------------- |
| `HLE_candidate.fa`             | `The output of candidate identification Module`     |
| `Intact.fa`                    | `The output of Intact TE identification Module`     |
| `all_consensus.fa`             | `The output of Boundary Identification Module`      |
| `confident_final_helitrons.fa` | `The output of Helitron Classification Module`      |
| `cl.fa`                        | `clustered sequences`                               |
| `nest/Final.fa`                | `The final result after removing nested insertions` |

## Notes

The file `nest/Final.fa` is the final output of the DeepHel pipeline,

We provide the output of DeepHel on Arabidopsis thaliana,  and all output files for Arabidopsis thaliana can be found under the athaliana folder
