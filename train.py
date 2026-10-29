"""Train the released FPP method from a fresh policy initialization."""
import runpy
import sys

DEFAULTS = {
    '--steps': '10000', '--schedule-steps': '50000', '--warmup': '2000',
    '--batch': '64', '--lr': '0.0002', '--save-every': '500', '--log-every': '20',
    '--demo-input': 'bpp', '--query-input': 'bpp', '--query-wrist-pool': 'learned',
    '--demo-pooling': 'learned', '--ema-decay': '0.999', '--ema-start': '2000',
    '--diagnostics-every': '200',
}
FLAGS = ['--direct-demo', '--shared-demo-kv', '--demo-full-patches',
         '--action-qk-norm', '--action-fp32-residual', '--stability-guard']

def arguments(argv):
    result = list(argv)
    supplied = {x.split('=',1)[0] for x in result if x.startswith('--')}
    for key,value in DEFAULTS.items():
        if key not in supplied: result.extend([key,value])
    result.extend(key for key in FLAGS if key not in supplied)
    return result

if __name__ == '__main__':
    sys.argv = ['train_video.py', *arguments(sys.argv[1:])]
    runpy.run_module('train_video', run_name='__main__')
