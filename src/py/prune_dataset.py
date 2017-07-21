"""Prune a dataset with one example per original example.

Used to generate the AddOneSent dataset from the AddSent dataset.
"""
import argparse
import collections
import json
import sys
import eval_squad

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Insert a description of this script.')
  parser.add_argument('dataset_file', help='Dataset file')
  parser.add_argument('prediction_file', help='Prediction File')
  parser.add_argument('out_prefix', help='Prefix to write output.')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def get_score(example, predictions):
  article, paragraph, qa = example
  ground_truths = list(map(lambda x: x['text'], qa['answers']))
  pred = predictions[qa['id']]
  return eval_squad.metric_max_over_ground_truths(eval_squad.f1_score, 
                                                  pred, ground_truths)

def choose_qids(dataset, predictions):
  orig_ids = [x for x in predictions if eval_squad.strip_id(x) == x]
  id_to_ex = {} 
  id_to_mut_id = collections.defaultdict(list)
  for article in dataset['data']:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        id_to_ex[qa['id']] = (article, paragraph, qa)
        orig_id = eval_squad.strip_id(qa['id'])
        if orig_id != qa['id']:
          id_to_mut_id[orig_id].append(qa['id'])
  kept_qids = []
  for orig_id in orig_ids:
    if orig_id not in id_to_mut_id:
      kept_qids.append(orig_id)
      continue
    mut_exs = [id_to_ex[mut_id] for mut_id in id_to_mut_id[orig_id]]
    mut_scores = [get_score(x, predictions) for x in mut_exs]
    min_ind, adv_score = min(enumerate(mut_scores), key=lambda x: x[1])
    kept_qids.append(id_to_mut_id[orig_id][min_ind])
  return kept_qids

def prune_dataset(dataset, kept_qids):
  out_data = []
  out_obj = {'version': dataset['version'], 'data': out_data}
  for article in dataset['data']:
    out_paragraphs = []
    out_article = {'title': article['title'], 'paragraphs': out_paragraphs}
    out_data.append(out_article)
    for paragraph in article['paragraphs']:
      cur_qas = []
      for qa in paragraph['qas']:
        if qa['id'] in kept_qids:
          cur_qas.append(qa)
      if cur_qas:
        new_paragraph = {'context': paragraph['context'], 'qas': cur_qas}
        out_paragraphs.append(new_paragraph)
  return out_obj

def write_data(data):
  with open(OPTS.out_prefix + '.json', 'w') as f:
    json.dump(data, f)
  with open(OPTS.out_prefix + '-indented.json', 'w') as f:
    json.dump(data, f, indent=2)

def main():
  with open(OPTS.dataset_file) as dataset_file:
    dataset = json.load(dataset_file)
  with open(OPTS.prediction_file) as prediction_file:
    predictions = json.load(prediction_file)
  kept_qids = choose_qids(dataset, predictions)
  new_data = prune_dataset(dataset, kept_qids)
  write_data(new_data)


if __name__ == '__main__':
  OPTS = parse_args()
  main()

