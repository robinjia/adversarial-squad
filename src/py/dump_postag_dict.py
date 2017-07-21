"""Compute most common POS tags for words in the Penn Treebank.

This is used by convert_questions.py.

Note: this cannot be run without downloading the Penn Treebank data.
The resulting map is tracked in the Git repo, so you should not need to run this.
"""
import collections
import json

PTB_FILE = 'data/ptb/sentences/train.gold.tagged'
OUT_FILE = 'data/postag_dict.json'

def main():
  postag_counts = collections.defaultdict(collections.Counter)
  with open(PTB_FILE) as f:
    for line in f:
      for tok in line.strip().split(' '):
        word, tag = tok.split('_')
        postag_counts[word.lower()][tag] += 1
  postag_dict = {w: c.most_common(1)[0][0] for w, c in postag_counts.iteritems()}
  with open(OUT_FILE, 'w') as f:
    json.dump(postag_dict, f)

if __name__ == '__main__':
  main()
