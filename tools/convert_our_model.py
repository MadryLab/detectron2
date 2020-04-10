import argparse
import subprocess
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
out_config = Path(args.out_path) / (out_fn + '_config.yaml')

ch.save(od, out_path)

final_out_path = f'{out_path}.pkl'

subprocess.run(f'./convert-torchvision-to-d2.py {out_path} {final_out_path}', shell=True)

print(f'saved in {out_config}')

with open(args.base_config, 'r') as f:
    config = yaml.load(f.read())

PIXEL_MEAN = [123.675, 116.280, 103.530]
PIXEL_STD = [58.395, 57.120, 57.375]
RESNETS = {
    'DEPTH':50,
    'STRIDE_IN_1X1':False
}

INPUT = {'FORMAT':'RGB'}

keys = ['WEIGHTS', 'PIXEL_MEAN', 'PIXEL_STD', 'RESNETS']
values = [str(final_out_path), PIXEL_MEAN, PIXEL_STD, RESNETS]

for k,v in zip(keys, values):
    config['MODEL'][k] = v

config['INPUT'] = (INPUT)

with open(out_config, 'w') as f:
    f.write(yaml.dump(config))
