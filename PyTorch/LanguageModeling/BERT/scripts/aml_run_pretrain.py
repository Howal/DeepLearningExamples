import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cfg",
                    default=None,
                    type=str,
                    required=True)

args = parser.parse_args()

print(os.environ)
os.system('bash ' + args.cfg)
