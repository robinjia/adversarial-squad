"""Update datsaet with human-approved sentences."""
import argparse
import collections
import csv
from HTMLParser import HTMLParser
import json
import random
import sys

html = HTMLParser()
OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Update dataset with human-approved sentences.')
  parser.add_argument('data_file', help='SQuAD JSON file.')
  parser.add_argument('batch_file', help='Batch CSV file from verify task.')
  parser.add_argument('out_prefix', help='Where to write new dataset')
  parser.add_argument('keep_mode', choices=['all', 'sample', 'longest'],
                      help='Which Turk response to keep')
  parser.add_argument('--unanimous', '-u', action='store_true',
                      help='Require unanimous vote')
  parser.add_argument('--prepend', '-p', action='store_true',
                      help='Prepend sentence')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def sent_format(s):
  s = s.replace("<span class='answer'>", '').replace("</span>", '')
  s = html.unescape(s)
  s = s.strip()
  if not s.endswith('.'):
    s = s + '.'
  return s[0].upper() + s[1:]

def read_batch():
  threshold = 3 if OPTS.unanimous else 2
  all_sents = collections.defaultdict(list)
  votes = collections.Counter()
  with open(OPTS.batch_file) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      if row['AssignmentStatus'] == 'Rejected': continue
      qids = row['Input.qids'].split('\t')
      sents = row['Answer.sents'].split('\t')
      responses = row['Answer.responses'].split('\t')
      for qid, s_str, a_str in zip(qids, sents, responses):
        s_list = s_str.split('|')
        a_list = a_str.split('|')
        for s_raw, a in zip(s_list, a_list):
          s = sent_format(s_raw)
          if s not in all_sents[qid]:
            all_sents[qid].append(s)
          if a == 'yes':
            votes[s] += 1
  filtered_sents = collections.defaultdict(list)
  for qid in all_sents:
    for s in all_sents[qid]:
      if votes[s] >= threshold: filtered_sents[qid].append(s)
  return filtered_sents

def is_mut(qid):
  return '-' in qid

def get_lengths(dataset):
  id_to_len = {}
  for article in dataset['data']:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        if is_mut(qa['id']): continue
        id_to_len[qa['id']] = len(paragraph['context'])
  return id_to_len

def update_dataset(batch_data):
  with open(OPTS.data_file) as f:
    orig_data = json.load(f)
  id_to_len = get_lengths(orig_data)

  out_data = []
  out_obj = {'version': orig_data['version'], 'data': out_data}
  for article in orig_data['data']:
    out_paragraphs = []
    out_article = {'title': article['title'], 'paragraphs': out_paragraphs}
    out_data.append(out_article)
    for paragraph in article['paragraphs']:
      if any(not is_mut(qa['id']) for qa in paragraph['qas']):
        # This is an original paragraph
        out_paragraphs.append(paragraph)
        continue
      if len(paragraph['qas']) != 1: raise ValueError()
      qa = paragraph['qas'][0]
      orig_id = qa['id'].split('-')[0]
      # NOTE: this means we can only have one auto-generated sentence
      orig_context = paragraph['context'][:id_to_len[orig_id]]
      new_sents = list(enumerate(batch_data[orig_id]))
      if OPTS.keep_mode == 'sample' and len(new_sents) > 1:
        new_sents = random.sample(new_sents, 1)
      elif OPTS.keep_mode == 'longest':
        new_sents = sorted(new_sents, key=lambda x: len(x[1]), reverse=True)[:1]
      for i, new_sent in new_sents:
        cur_qa = {
            'question': qa['question'],
            'id': '%s-turk%d' % (qa['id'], i),
            'answers': qa['answers']
        }
        if OPTS.prepend:
          new_answers = []
          for a in qa['answers']:
            new_answers.append({
                'text': a['text'],
                'answer_start': a['answer_start'] + len(new_sent) + 1
            })
          cur_qa['answers'] = new_answers
          cur_text = '%s %s' % (new_sent, orig_context)
        else:
          cur_text = '%s %s' % (orig_context, new_sent)
        cur_paragraph = {'context': cur_text, 'qas': [cur_qa]}
        out_paragraphs.append(cur_paragraph)
  return out_obj

def write_data(data):
  with open(OPTS.out_prefix + '.json', 'w') as f:
    json.dump(data, f)
  with open(OPTS.out_prefix + '-indented.json', 'w') as f:
    json.dump(data, f, indent=2)

def main():
  random.seed(0)
  batch_data = read_batch()
  new_data = update_dataset(batch_data)
  write_data(new_data)

if __name__ == '__main__':
  OPTS = parse_args()
  main()

