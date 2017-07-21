"""Create HITs for sentence-verifying task."""
import argparse
import cgi
import collections
import json
import sys
import unicodecsv as csv

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Create HITs for sentence-verifying task.')
  parser.add_argument('squad_file', help='SQuAD file')
  parser.add_argument('batch_file', help='Batch CSV file')
  parser.add_argument('num_per_hit', type=int, default=5,
                      help='Number of sentences per HIT')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def sent_format(s):
  s = s.strip()
  if not s.endswith('.'):
    s = s + '.'
  return s[0].upper() + s[1:]


def escape(s):
  return cgi.escape(s).replace(',', '&#44;').replace('"', '&quot;')

def read_questions():
  id_to_question = {}
  with open(OPTS.squad_file) as f:
    obj = json.load(f)
  for article in obj['data']:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        id_to_question[qa['id']] = qa['question']
  return id_to_question

def read_sentences():
  id_to_sents = collections.defaultdict(list)
  with open(OPTS.batch_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
      input_qids = row['Input.qids'].split('\t')
      input_sents = row['Input.sents'].split('\t')
      ans_is_good = row['Answer.is-good'].split('\t')
      ans_responses = row['Answer.responses'].split('\t')
      for qid, s, is_good, response in zip(input_qids, input_sents, ans_is_good, ans_responses):
        if is_good == 'yes':
          response = s
        if response not in id_to_sents[qid]:
          id_to_sents[qid].append(response)
  return id_to_sents

def make_hits(id_to_question, id_to_sents):
  examples = []
  for qid in id_to_sents:
    sents = id_to_sents[qid]
    q = id_to_question[qid]
    examples.append((qid, q, '|'.join(sents)))
  hits = []
  num_hits = (len(examples) / OPTS.num_per_hit) + 1
  for i in range(num_hits):
    start = i * len(examples) / num_hits
    end = (i+1) * len(examples) / num_hits
    cur_examples = examples[start:end]
    id_str = '\t'.join(ex[0] for ex in cur_examples)
    q_str = '\t'.join(escape(ex[1]).encode('ascii', 'ignore') for ex in cur_examples)
    s_str = '\t'.join(sent_format(escape(ex[2])) for ex in cur_examples)
    hits.append(('%d' % i, id_str, q_str, s_str))
  return hits

def dump_hits(hits):
  print 'id,qids,questions,sents'
  for h in hits:
    print ('%s,%s,%s,%s' % (h)).encode('ascii')

def main():
  id_to_question = read_questions()
  id_to_sents = read_sentences()
  hits = make_hits(id_to_question, id_to_sents)
  dump_hits(hits)
  

if __name__ == '__main__':
  OPTS = parse_args()
  main()

