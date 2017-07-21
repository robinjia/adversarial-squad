"""Create HITs for human evaluation task."""
import argparse
import cgi
import json
import random
import sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Create HITs for human evaluation task.')
  parser.add_argument('filename', help='SQuAD dataset filename')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def escape(s):
  return cgi.escape(s).replace(',', '&#44;').replace('"', '&quot;').replace('\n', ' ')

def gen_examples(dataset):
  examples = []
  for article in dataset['data']:
    for paragraph in article['paragraphs']:
      p_escaped = escape(paragraph['context'])
      for qa in paragraph['qas']:
        q_escaped = escape(qa['question'])
        examples.append((qa['id'], p_escaped, q_escaped))
  random.shuffle(examples)
  return examples

def dump_hits(hits):
  print 'id,qid,paragraph,question'
  for i, (qid, paragraph, question) in enumerate(hits):
    print ('%d,%s,%s,%s' % (i, qid, paragraph, question)).encode('utf-8')

def main():
  random.seed(0)
  with open(OPTS.filename) as f:
    dataset = json.load(f)
  hits = gen_examples(dataset)
  dump_hits(hits)


if __name__ == '__main__':
  OPTS = parse_args()
  main()

