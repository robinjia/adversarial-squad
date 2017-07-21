"""Run a SQuAD model (for testing)."""
import argparse
import json
import os
import shutil
import sys

import bidaf
import matchlstm

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Insert a description of this script.')
  parser.add_argument('model', help='Name of model')
  parser.add_argument('filename', help='SQuAD JSON data.')
  parser.add_argument('out_dir', help='Temporary output directory.')
  parser.add_argument('--train_dir', '-t', 
                      help='Path to trained parameters (default is official pretrained BiDAF model)')
  parser.add_argument('--pred-file', '-p', help='Write preds to this file')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def main():
  if os.path.exists(OPTS.out_dir):
    shutil.rmtree(OPTS.out_dir)
  os.makedirs(OPTS.out_dir)
  if OPTS.model == 'bidaf-single':
    scores, preds = bidaf.run_single(OPTS.filename, OPTS.out_dir, verbose=True)
  elif OPTS.model == 'bidaf-ensemble':
    scores, preds = bidaf.run_ensemble(OPTS.filename, OPTS.out_dir, verbose=True)
  elif OPTS.model == 'bidaf-debug-server':
    scores, preds = bidaf.debug_server(OPTS.filename, OPTS.out_dir, verbose=True,
                                       train_dir=OPTS.train_dir)
  elif OPTS.model == 'matchlstm-single':
    scores, preds = matchlstm.run_single(OPTS.filename, OPTS.out_dir, verbose=True)
  elif OPTS.model == 'matchlstm-ensemble':
    scores, preds = matchlstm.run_ensemble(OPTS.filename, OPTS.out_dir, verbose=True)
  else:
    raise ValueError('Unrecognized model "%s"' % OPTS.model)
  print json.dumps(scores, indent=2)
  print json.dumps(preds, indent=2)
  if OPTS.pred_file:
    with open(OPTS.pred_file, 'w') as f:
      json.dump(preds, f)

if __name__ == '__main__':
  OPTS = parse_args()
  main()
