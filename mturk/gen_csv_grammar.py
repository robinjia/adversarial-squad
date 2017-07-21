"""Create HITs for grammar-fixing task."""
import argparse
import cgi
import json
import random
import sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Create HITs for grammar-fixing task.')
  parser.add_argument('filename', help='-mturk.tsv file')
  parser.add_argument('num_per_hit', type=int, default=5,
                      help='Number of sentences per HIT')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def gen_examples(filename):
  examples = []
  with open(filename) as f:
    for line in f:
      qid, sent = line.strip().split('\t')
      s_escaped = sent.replace(',', '&#44;').replace('"', '&quot;')
      examples.append((qid, s_escaped))
  random.shuffle(examples)
  return examples

def make_hits(examples):
  hits = []
  num_hits = (len(examples) / OPTS.num_per_hit) + 1
  for i in range(num_hits):
    start = i * len(examples) / num_hits
    end = (i+1) * len(examples) / num_hits
    cur_examples = examples[start:end]
    id_str = '\t'.join(ex[0] for ex in cur_examples)
    s_str = '\t'.join(ex[1] for ex in cur_examples)
    hits.append(('%d' % i, id_str, s_str))
  return hits

def dump_hits(hits):
  print 'id,qids,sents'
  for h in hits:
    print ('%s,%s,%s' % (h)).encode('ascii', errors='ignore')

def main():
  random.seed(0)
  examples = gen_examples(OPTS.filename)
  hits = make_hits(examples)
  dump_hits(hits)


if __name__ == '__main__':
  OPTS = parse_args()
  main()

