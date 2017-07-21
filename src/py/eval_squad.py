"""Do an adversarial evaluation of SQuAD."""
import argparse
from collections import Counter, OrderedDict, defaultdict
import json
import re
import string
import sys

OPTS = None

### BEGIN: official SQuAD code version 1.1
### See https://rajpurkar.github.io/SQuAD-explorer/
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
### END: official SQuAD code

def strip_id(id_str):
  return id_str.split('-')[0]

def highlight_after(s, n):
  return s[:n] + colored(s[n:], 'cyan')

def get_answer_color(pred, answers):
  ans_texts = [a['text'] for a in answers]
  exact = metric_max_over_ground_truths(exact_match_score, pred, ans_texts)
  if exact: return 'green'
  f1 = metric_max_over_ground_truths(f1_score, pred, ans_texts)
  if f1: return 'yellow'
  return 'red'


def print_details(dataset, predictions, adv_ids):
  id_to_paragraph = {}
  for article in dataset:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        id_to_paragraph[qa['id']] = paragraph['context']
  for article in dataset:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        orig_id = strip_id(qa['id'])
        if orig_id != qa['id']: continue  # Skip the mutated ones 
        adv_id = adv_ids[orig_id]
        print 'Title: %s' % article['title'].encode('utf-8')
        print 'Paragraph: %s' % paragraph['context'].encode('utf-8')
        print 'Question: %s' % qa['question'].encode('utf-8')
        print 'Answers: [%s]' % ', '.join(a['text'].encode('utf-8')
                                          for a in qa['answers'])
        orig_color = get_answer_color(predictions[orig_id], qa['answers'])
        print 'Predicted: %s' % colored(
            predictions[orig_id], orig_color).encode('utf-8')
        print 'Adversary succeeded?: %s' % (adv_id != orig_id)
        if adv_id != orig_id:
          print 'Adversarial Paragraph: %s' % highlight_after(
              id_to_paragraph[adv_id], len(paragraph['context'])).encode('utf-8')
          # highlight_after is a hack that only works when mutations append stuff.
          adv_color = get_answer_color(predictions[adv_id], qa['answers'])
          print 'Prediction under Adversary: %s' % colored(
              predictions[adv_id], adv_color).encode('utf-8')
        print



def evaluate_adversarial(dataset, predictions, verbose=False, id_set=None):
  orig_f1_score = 0.0
  orig_exact_match_score = 0.0
  adv_f1_scores = {}  # Map from original ID to F1 score
  adv_exact_match_scores = {}  # Map from original ID to exact match score
  adv_ids = {}
  all_ids = set()  # Set of all original IDs
  f1 = exact_match = 0
  for article in dataset:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        orig_id = qa['id'].split('-')[0]
        if id_set and orig_id not in id_set: continue
        all_ids.add(orig_id)
        if qa['id'] not in predictions:
          message = 'Unanswered question ' + qa['id'] + ' will receive score 0.'
          print >> sys.stderr, message
          continue
        ground_truths = list(map(lambda x: x['text'], qa['answers']))
        prediction = predictions[qa['id']]
        cur_exact_match = metric_max_over_ground_truths(exact_match_score,
                                                        prediction, ground_truths)
        cur_f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        if orig_id == qa['id']:
          # This is an original example
          orig_f1_score += cur_f1
          orig_exact_match_score += cur_exact_match
          if orig_id not in adv_f1_scores:
            # Haven't seen adversarial example yet, so use original for adversary
            adv_ids[orig_id] = orig_id
            adv_f1_scores[orig_id] = cur_f1
            adv_exact_match_scores[orig_id] = cur_exact_match
        else:
          # This is an adversarial example
          if (orig_id not in adv_f1_scores or adv_ids[orig_id] == orig_id 
              or adv_f1_scores[orig_id] > cur_f1):
            # Always override if currently adversary currently using orig_id
            adv_ids[orig_id] = qa['id']
            adv_f1_scores[orig_id] = cur_f1
            adv_exact_match_scores[orig_id] = cur_exact_match
  if verbose:
    print_details(dataset, predictions, adv_ids)
  orig_f1 = 100.0 * orig_f1_score / len(all_ids)
  orig_exact_match = 100.0 * orig_exact_match_score / len(all_ids)
  adv_exact_match = 100.0 * sum(adv_exact_match_scores.values()) / len(all_ids)
  adv_f1 = 100.0 * sum(adv_f1_scores.values()) / len(all_ids)
  return OrderedDict([
      ('orig_exact_match', orig_exact_match),
      ('orig_f1', orig_f1),
      ('adv_exact_match', adv_exact_match),
      ('adv_f1', adv_f1),
  ])

def split_by_attempted(dataset):
  all_ids = set()
  attempted_ids = set()
  for article in dataset:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        orig_id = qa['id'].split('-')[0]
        all_ids.add(orig_id)
        if orig_id != qa['id']:
          attempted_ids.add(orig_id)
  not_attempted_ids = all_ids - attempted_ids
  return attempted_ids, not_attempted_ids

def evaluate_by_attempted(dataset, predictions):
  attempted, not_attempted = split_by_attempted(dataset)
  total_num = len(attempted) + len(not_attempted)
  results_attempted = evaluate_adversarial(dataset, predictions,
                                           id_set=attempted)
  print 'Attempted %d/%d = %.2f%%' % (
      len(attempted), total_num, 100.0 * len(attempted) / total_num)
  print json.dumps(results_attempted)
  results_not_attempted = evaluate_adversarial(dataset, predictions,
                                               id_set=not_attempted)
  print 'Did not attempt %d/%d = %.2f%%' % (
      len(not_attempted), total_num, 100.0 * len(not_attempted) / total_num)
  print json.dumps(results_not_attempted)


if __name__ == '__main__':
  expected_version = '1.1'
  parser = argparse.ArgumentParser(
      description='Adverarial evaluation for SQuAD ' + expected_version)
  parser.add_argument('dataset_file', help='Dataset file')
  parser.add_argument('prediction_file', help='Prediction File')
  parser.add_argument('--out-file', '-o', default=None,
                      help='Write JSON output to this file (default is stdout).')
  parser.add_argument('--verbose', '-v', default=False, action='store_true',
                      help='Enable verbose logging.')
  parser.add_argument('--split-by-attempted', default=False, action='store_true',
                      help='Split by whether adversary attempted the example.')
  args = parser.parse_args()
  with open(args.dataset_file) as dataset_file:
    dataset_json = json.load(dataset_file)
   # if (dataset_json['version'] != expected_version):
   #   print >> sys.stderr, (
   #       'Evaluation expects v-' + expected_version +
   #       ', but got dataset with v-' + dataset_json['version'])
    dataset = dataset_json['data']
  with open(args.prediction_file) as prediction_file:
      predictions = json.load(prediction_file)
  if args.verbose:
    from termcolor import colored
  results = evaluate_adversarial(dataset, predictions, verbose=args.verbose)
  if args.out_file:
    with open(args.out_file, 'wb') as f:
      json.dump(results, f)
  else:
    print json.dumps(results)
  if args.split_by_attempted:
    evaluate_by_attempted(dataset, predictions)
