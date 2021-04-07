#imports
import os
import numpy as np
import csv
from tqdm import tqdm
import pandas as pd
#___________________________________________________________________________________________________
#rename files
#commented out after success
#___________________________________________________________________________________________________
# counter = 0
# pattern = 'part-00000'
# while counter<127:
#     command = f'! for f in ./{pattern}.lzo[!.]*; do mv "$f" ./{pattern}.lzo; done' #looks for files like part-000000.lzo?Google...
#     os.system(command)
#     command = f'! for f in ./{pattern}.lzo[!\?]*; do mv "$f" ./{pattern+"-index"}.lzo; done' #looks for files like part-000000.lzo.index...
#     os.system(command)
#     counter+=1
#     pattern = pattern[:-len(str(counter))]+str(counter)


#___________________________________________________________________________________________________
# draw subsamples
#___________________________________________________________________________________________________
counter = 0
pattern = 'part-00000'
while counter<127:
    print(f"{'=' * 120}\nExtracting {pattern}")
    command = f"lzop -x ./{pattern}.lzo"
    os.system(command)
    print(f"Extraction done\n\nOpening file")
    with open(f"{pattern}", encoding="utf-8") as file:
        lines = file.readlines()
        print("File is opened\n")
        random_indicies = np.random.default_rng().choice(len(lines), size=int(len(lines)*0.01), replace=False)#int(len(lines)*0.01)
        with open("subsample.csv", mode='a') as csv_file:
            for j, id in enumerate(tqdm(random_indicies)):
                line = lines[id]
                csv_file.write(line)
        print(f"\nWriting to csv done")
    os.remove(f"{pattern}")

    with open("history.txt","a") as history:
        history.write(pattern)
        history.write('\n')
    counter += 1
    pattern = pattern[:-len(str(counter))] + str(counter)
