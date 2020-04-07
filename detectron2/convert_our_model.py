import argparse
import os
import torch as ch
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--in-path')
parser.add_argument('--out-path')
args = parser.parse_args()

blob = ch.load(args.in_path)
model = blob['model']

od = OrderedDict()

for k, v in model.items():
    if not ('batches_tracked' in k) and 'module.model.' in k:
        new_key = k.replace('module.model.', '')

        od[new_key] = v
