#!/usr/bin/env python
# 
# Usage:
# python prepare.py --input some.txt
# chmod +x prepare.py && ./prepare.py --input some.txt

import os
import requests
import tiktoken
import numpy as np
import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("path")

args = parser.parse_args()
file = Path(args.path)

directory = os.path.dirname(os.path.normpath(file))
print(f'output to: {directory}')  # Output: /path/to/

with open(file, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(directory, 'train.bin'))
val_ids.tofile(os.path.join(directory, 'val.bin'))
