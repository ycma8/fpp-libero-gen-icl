"""Official LIBERO-Gen evaluation of the released FPP EMA policy."""
import runpy
import sys
if __name__ == '__main__':
    if not any(x.split('=',1)[0]=='--weights' for x in sys.argv[1:]):
        sys.argv.extend(['--weights','ema'])
    runpy.run_module('evaluate_official_video', run_name='__main__')
