"""Get common English words from Brown Corpus."""
import argparse
from nltk import FreqDist
from nltk.corpus import brown
import string
import sys

OPTS = None
PUNCTUATION = set(string.punctuation) | set(['``', "''", "--"])


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('size', type=int)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def main():
  freq_dist = FreqDist(w.lower() for w in brown.words() if w not in PUNCTUATION)
  vocab = [x[0] for x in freq_dist.most_common()[:OPTS.size]]
  for w in vocab:
    print w


if __name__ == '__main__':
  OPTS = parse_args()
  main()
