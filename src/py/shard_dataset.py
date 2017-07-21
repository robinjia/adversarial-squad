"""Generate shards of a dataset (used for AddAny)."""
import argparse
import json
import random
import sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Insert a description of this script.')
  parser.add_argument('filename', metavar='data.json', help='Input file')
  parser.add_argument('out_prefix', help='Output prefix')
  parser.add_argument('--num-shards', '-n', default=10, type=int,
                      help='Number of shards')
  parser.add_argument('--rng-seed', '-s', default=0, type=int, help='RNG seed')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def get_ids(dataset):
  id_list = []
  for article in dataset['data']:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        id_list.append(qa['id'])
  return id_list

def select_data(dataset, id_set):
  out_data = []
  out_obj = {'version': dataset['version'], 'data': out_data}
  for article in dataset['data']:
    out_paragraphs = []
    out_article = {'title': article['title'], 'paragraphs': out_paragraphs}
    for paragraph in article['paragraphs']:
      kept_qas = []
      orig_paragraph = {'context': paragraph['context'], 'qas': kept_qas}
      for qa in paragraph['qas']:
        if qa['id'] not in id_set: continue
        kept_qas.append(qa)
      if kept_qas:
        out_paragraphs.append(orig_paragraph)
    if out_paragraphs:
      out_data.append(out_article)
  return out_obj

def main():
  random.seed(OPTS.rng_seed)
  with open(OPTS.filename) as f:
    dataset = json.load(f)
  id_list = get_ids(dataset)
  random.shuffle(id_list)
  for i in range(OPTS.num_shards):
    start_ind = i * len(id_list) / OPTS.num_shards
    end_ind = (i + 1) * len(id_list) / OPTS.num_shards
    cur_ids = set(id_list[start_ind:end_ind])
    cur_data = select_data(dataset, cur_ids)
    out_filename = OPTS.out_prefix + '-shard%d.json' % i
    with open(out_filename, 'w') as f:
      json.dump(cur_data, f)


if __name__ == '__main__':
  OPTS = parse_args()
  main()

