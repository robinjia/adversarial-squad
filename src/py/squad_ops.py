#!/usr/bin/env python
"""Script to do common operations on SQuAD data files."""
import argparse
import collections
import json
import random
import sys

OPTS = None
DESCRIPTIONS = collections.OrderedDict([
    ('split', 'Split a file into two parts.'),
    ('shard', 'Split a file into |n| equally sized parts.'),
    ('cat', 'Concatenate multiple files.  Does not deduplicate questions.'),
    ('count', 'Count number of articles, paragraphs, and questions.'),
    ('list', 'Print a list of all qids in a file.'),
    ('extract', 'Extract a subset based on a list of qids.'),
    ('paragraphs', 'Strip to paragraph + placeholders.'),
])
HELP_MSG = '''
squad_ops.py: Common operations on SQuAD data files.

Usage: python %(arg0)s [action] [params]

Available actions:
%(actions)s

Run python %(arg0)s [action] -h for additional information.
''' % {
    'arg0': sys.argv[0], 
    'actions': '\n'.join('  %s: %s' % (a, desc) for a, desc in DESCRIPTIONS.iteritems())
}


def parse_args():
  if len(sys.argv) == 1:
    print >> sys.stderr, HELP_MSG
    sys.exit(1)
  action = sys.argv[1]
  if action not in DESCRIPTIONS:
    print >> sys.stderr, '\nUnrecognized action "%s".' % action
    print >> sys.stderr, HELP_MSG
    sys.exit(1)
  name = '%s %s' % ('squad_ops.py', action)
  description = DESCRIPTIONS[action]
  parser = argparse.ArgumentParser(name, description=description)
  if action in ('split', 'shard', 'count', 'list', 'extract', 'paragraphs'):
    parser.add_argument('filename', metavar='data.json', help='Input file')
  if action in ('split', 'shard', 'cat', 'extract', 'paragraphs'):
    parser.add_argument('--indent', '-i', type=int, default=None,
                        help='Indent json output for readability')
  if action in ('split', 'shard'):
    parser.add_argument('out_prefix', metavar='out', help='Output prefix')
    parser.add_argument('--rng-seed', '-s', default=0, type=int, help='RNG seed')
    parser.add_argument('--paragraph', '-p', action='store_true',
                        help='Split based on paragraphs, not questions')
  if action == 'shard':
    parser.add_argument('--num-shards', '-n', default=10, type=int,
                        help='Number of shards')
  elif action == 'split':
    parser.add_argument('--split-frac', '-f', default=0.5, type=float,
                        help='Fraction of data to put in first split')
  if action == 'count':
    parser.add_argument('--article', '-a', action='store_true', help='Only count articles')
    parser.add_argument('--paragraph', '-p', action='store_true', help='Only count paragraphs')
    parser.add_argument('--question', '-q', action='store_true', help='Only count questions')
  if action == 'cat':
    parser.add_argument('files', nargs='+', help='Files to conatenate')
  if action == 'extract':
    parser.add_argument('id_filename', help='File with list of qids')
  if action == 'paragraphs':
    parser.add_argument('--copies', '-c', type=int, default=1,
                        help='Number of copies of each paragraph')
  argv = sys.argv[2:]
  opts = parser.parse_args(argv)
  opts.action = action
  return opts

def get_ids(dataset):
  id_list = []
  for article in dataset['data']:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        id_list.append(qa['id'])
  return id_list

def get_ids_by_paragraph(dataset):
  ids_by_paragraph = []
  for article in dataset['data']:
    for paragraph in article['paragraphs']:
      ids_by_paragraph.append([qa['id'] for qa in paragraph['qas']])
  return ids_by_paragraph


def select_data(dataset, id_list):
  id_set = set(id_list)
  out_data = []
  out_obj = {'version': dataset['version'], 'data': out_data}
  for article in dataset['data']:
    out_paragraphs = []
    out_article = {'title': article['title'], 'paragraphs': out_paragraphs}
    for paragraph in article['paragraphs']:
      kept_qas = []
      out_paragraph = {'context': paragraph['context'], 'qas': kept_qas}
      for qa in paragraph['qas']:
        if qa['id'] not in id_set: continue
        kept_qas.append(qa)
      if kept_qas:
        out_paragraphs.append(out_paragraph)
    if out_paragraphs:
      out_data.append(out_article)
  return out_obj

def shard():
  random.seed(OPTS.rng_seed)
  with open(OPTS.filename) as f:
    dataset = json.load(f)
  if OPTS.paragraph:
    obj_list = get_ids_by_paragraph(dataset)
  else:
    obj_list = get_ids(dataset)
  random.shuffle(obj_list)
  for i in range(OPTS.num_shards):
    start_ind = i * len(obj_list) / OPTS.num_shards
    end_ind = (i + 1) * len(obj_list) / OPTS.num_shards
    if OPTS.paragraph:
      cur_ids = [y for x in obj_list[start_ind:end_ind] for y in x]
    else:
      cur_ids = obj_list[start_ind:end_ind]
    cur_data = select_data(dataset, cur_ids)
    out_filename = OPTS.out_prefix + '-shard%d.json' % i
    with open(out_filename, 'w') as f:
      json.dump(cur_data, f, indent=OPTS.indent)

def split():
  random.seed(OPTS.rng_seed)
  with open(OPTS.filename) as f:
    dataset = json.load(f)
  if OPTS.paragraph:
    obj_list = get_ids_by_paragraph(dataset)
  else:
    obj_list = get_ids(dataset)
  random.shuffle(id_list)
  num_0 = int(round(OPTS.split_frac * len(obj_list)))
  if OPTS.paragraph:
    ids_0 = [y for x in obj_list[:num_0] for y in x]
    ids_1 = [y for x in obj_list[num_0:] for y in x]
  else:
    ids_0 = id_list[:num_0]
    ids_1 = id_list[num_0:]
  out_data = [select_data(dataset, ids_0), select_data(dataset, ids_1)]
  for i, cur_data in enumerate(out_data):
    out_filename = OPTS.out_prefix + '-split%d.json' % i
    with open(out_filename, 'w') as f:
      json.dump(cur_data, f, indent=OPTS.indent)

def count():
  with open(OPTS.filename) as f:
    dataset = json.load(f)
  num_articles = 0
  num_paragraphs = 0
  num_questions = 0
  for a in dataset['data']:
    num_articles += 1
    for p in a['paragraphs']:
      num_paragraphs += 1
      for q in p['qas']:
        num_questions += 1
  print_all = not any((OPTS.article, OPTS.paragraph, OPTS.question))
  if print_all:
    print '%d articles, %d paragraphs, %d questions' % (num_articles, num_paragraphs, num_questions)
  else:
    if OPTS.article:
      print '%d articles' % num_articles
    if OPTS.paragraph:
      print '%d paragraphs' % num_paragraphs
    if OPTS.question:
      print '%d questions' % num_questions


def cat():
  article_map = {}
  version = None
  for fn in OPTS.files:
    with open(fn) as f:
      cur_data = json.load(f)
    version = cur_data['version']
    for article in cur_data['data']:
      title = article['title']
      if title in article_map:
        out_article = article_map[title]
      else:
        out_article = {'title': title, 'paragraphs': []}
        article_map[title] = out_article
      out_article['paragraphs'].extend(article['paragraphs'])
  out_data = {'version': version, 'data': article_map.values()}
  print json.dumps(out_data, indent=OPTS.indent)

def list_qids():
  with open(OPTS.filename) as f:
    dataset = json.load(f)
  qids = get_ids(dataset)
  for qid in qids:
    print qid

def extract():
  with open(OPTS.filename) as f:
    dataset = json.load(f)
  with open(OPTS.id_filename) as f:
    id_list = [line.strip() for line in f]
  out_data = select_data(dataset, id_list)
  print json.dumps(out_data, indent=OPTS.indent)

def strip_to_paragraphs():
  with open(OPTS.filename) as f:
    dataset = json.load(f)
  out_data = []
  out_obj = {'version': dataset['version'], 'data': out_data} 
  for article in dataset['data']:
    out_paragraphs = []
    out_article = {'title': article['title'], 'paragraphs': out_paragraphs}
    out_data.append(out_article)
    for paragraph in article['paragraphs']:
      base_qid = paragraph['qas'][0]['id']
      qas = []
      for i in range(OPTS.copies):
        qas.append({
            'id': '%s-parOnly-%d' % (base_qid, i), 
            'question': '',
            'answers': [{'text': '', 'answer_start': 0}]
        })
      out_paragraph = {'context': paragraph['context'], 'qas': qas}
      out_paragraphs.append(out_paragraph)
  print json.dumps(out_obj, indent=OPTS.indent)

def main():
  global OPTS
  OPTS = parse_args()
  action_funcs = {
      'split': split,
      'shard': shard,
      'count': count,
      'cat': cat,
      'list': list_qids,
      'extract': extract,
      'paragraphs': strip_to_paragraphs,
  }
  f = action_funcs[OPTS.action]
  f()

if __name__ == '__main__':
  main()

