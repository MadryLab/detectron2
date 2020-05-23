import argparse
import shutil
import subprocess
from pathlib import Path
import os
import torch as ch
from collections import OrderedDict
import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

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
out_dir = Path(args.out_path) / out_fn

try:
    out_dir.mkdir()
except:
    shutil.rmtree(out_dir)
    out_dir.mkdir()

out_path = out_dir / 'torchvision_fmt_weights.pt'

out_config = out_dir / 'config.yaml'

ch.save(od, out_path)

final_out_path = out_dir / 'detectron2_weights.pkl'

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

log_dir = out_dir / 'logs'
print('log dir', log_dir)
log_dir.mkdir()
config['OUTPUT_DIR'] = str(log_dir)
config['_BASE_'] = os.path.join('/'.join(args.base_config.split('/')[:-1]), config['_BASE_'])

with open(out_config, 'w') as f:
    f.write(yaml.dump(config))

print(f'config path: {out_config}')
cmd1 = 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; ./train_net.py --config-file'

cmd = f'{cmd1} {out_config} --num-gpus 8'
print('run command',  cmd)
with open('/tmp/cmds.txt', 'a') as f:
    f.write(f'{cmd}\n')


