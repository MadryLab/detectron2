import argparse
from pathlib import Path
import os
import torch as ch
from collections import OrderedDict
import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--in-path')
parser.add_argument('--base-config')
parser.add_argument('--out-path')
args = parser.parse_args()

blob = ch.load(args.in_path)
model = blob['model']

od = OrderedDict()

for k, v in model.items():
    if not ('batches_tracked' in k) and 'module.model.' in k:
        new_key = k.replace('module.model.', '')

        od[new_key] = v

out_fn = '_'.join(args.in_path.split('/')[-2:])
out_path = Path(args.out_path) / out_fn

ch.save(od, out_path)

print(f'saved in {out_path}')

with open(args.base_config, 'r') as f:
    config = yaml.load(f.read())

import pdb; pdb.set_trace()
