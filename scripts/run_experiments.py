import sys, argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from airtrust.experiment import run_suite
p=argparse.ArgumentParser()
p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
p.add_argument('--rounds',type=int,default=20)
p.add_argument('--steps',type=int,default=12)
p.add_argument('--output',default='results')
p.add_argument('--no-robustness',action='store_true')
a=p.parse_args()
run_suite(a.output,a.seeds,a.rounds,a.steps,not a.no_robustness)
