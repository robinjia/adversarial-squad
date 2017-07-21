"""Print results from SQuAD, for manual inspection."""
import argparse
from collections import Counter, OrderedDict, defaultdict
import json
import random
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string
import sys
from termcolor import colored

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset_file', help='Dataset file')
  parser.add_argument('prediction_file', help='Prediction File')
  parser.add_argument('--num-examples', '-n', type=int, default=0,
                      help='Number of examples to print per outcome (default = all).')
  parser.add_argument('--rng-seed', '-s', type=int, default=0, help='RNG seed.')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

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

def get_score(example, predictions):
  article, paragraph, qa = example
  ground_truths = list(map(lambda x: x['text'], qa['answers']))
  pred = predictions[qa['id']]
  return metric_max_over_ground_truths(f1_score, pred, ground_truths)

def highlight_after(s, n):
  return s[:n] + colored(s[n:], 'cyan')

def print_exs(examples, predictions):
  num_question_words = Counter()
  num_pred_ans_words = Counter()
  ngram_match = Counter()
  ans_in_adv = 0
  ans_only_in_adv = 0
  ans_overlaps = 0
  for (orig_a, orig_p, orig_q), (adv_a, adv_p, adv_q) in examples:
    print 'Article: %s' % orig_a['title'].encode('utf-8')
    print 'Question: %s' % colored(adv_q['question'], 'cyan').encode('utf-8')
    print 'Paragraph: %s' % highlight_after(
        adv_p['context'], len(orig_p['context'])).encode('utf-8')
    # highlight_after is a hack that only works when mutations append stuff.
    print 'Answers: [%s]' % ', '.join(a['text'].encode('utf-8')
                                      for a in adv_q['answers'])
    print ('Prediction: %s' % predictions[adv_q['id']]).encode('utf-8')
    q_words = word_tokenize(orig_q['question'])
    # Length of question
    num_question_words[len(q_words)] += 1

    # N-gram overlap
    for n in (2, 3, 4, 5, 6, 7):
      for i in range(len(q_words) - n + 1):
        ngram = ' '.join(q_words[i:i+n])
        if ngram in orig_p['context']:
          ngram_match[n] += 1
          break

    # Answer containment
    adv_sent = adv_p['context'][len(orig_p['context']):]
    pred_ans = predictions[adv_q['id']]
    if pred_ans in adv_sent:
      ans_in_adv += 1
    if pred_ans not in orig_p['context']:
      ans_overlaps += 1
      if pred_ans in adv_sent:
        ans_only_in_adv += 1

    # Answer length
    num_pred_ans_words[len(word_tokenize(pred_ans))] += 1

    print
  print 'Total: %d examples' % len(examples)
  print 'Lengths of questions:'
  for k in sorted(num_question_words):
    print '%d: %d (%.1f%%)' % (k, num_question_words[k], 100.0 * num_question_words[k] / len(examples))
  print 'Lengths of predicted answer:'
  for k in sorted(num_pred_ans_words):
    print '%d: %d (%.1f%%)' % (k, num_pred_ans_words[k], 100.0 * num_pred_ans_words[k] / len(examples))
  print 'N-gram matches:'
  for n in sorted(ngram_match):
    print '%d: %d (%.1f%%)' % (n, ngram_match[n], 100.0 * ngram_match[n] / len(examples))
  print 'In adv_sent: %d (%.1f%%)' % (ans_in_adv, 100.0 * ans_in_adv / len(examples))
  print 'Only in adv_sent: %d (%.1f%%)' % (ans_only_in_adv, 100.0 * ans_only_in_adv / len(examples))
  print 'Overlaps adv_sent: %d (%.1f%%)' % (ans_overlaps, 100.0 * ans_overlaps / len(examples))

  print


def print_results(dataset, predictions):
  """Print some cases where model was right (exact match).
  
  We give OPTS.num_examples examples of
    - When the adversary drove F1 to 0,
    - When the adversary drove F1 between (0, 1)
    - When the adversary could not drive F1 below 1.
  """
  orig_ids = [x for x in predictions if strip_id(x) == x]
  random.shuffle(orig_ids)
  id_to_ex = {}
  id_to_mut_id = defaultdict(list)
  adv_successes = []
  #adv_partials = []
  adv_failures = []
  for article in dataset['data']:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        id_to_ex[qa['id']] = (article, paragraph, qa)
        orig_id = strip_id(qa['id'])
        if orig_id != qa['id']:
          id_to_mut_id[orig_id].append(qa['id'])
  for orig_id in orig_ids:
    ex = id_to_ex[orig_id]
    if orig_id not in id_to_mut_id: continue
    mut_exs = [id_to_ex[mut_id] for mut_id in id_to_mut_id[orig_id]]
    orig_score = get_score(ex, predictions)
    if orig_score != 1.0: continue  # Only keep things that were exact match
    mut_scores = [get_score(x, predictions) for x in mut_exs]
    min_ind, adv_score = min(enumerate(mut_scores), key=lambda x: x[1])
    adv_ex = mut_exs[min_ind]
#    if adv_score == 0.0:
#      adv_successes.append((ex, adv_ex))
#    elif adv_score < 1.0:
#      adv_partials.append((ex, adv_ex))
    if adv_score < 1.0:
      adv_successes.append((ex, adv_ex))
    else:
      adv_failures.append((ex, adv_ex))
  #total = len(adv_successes) + len(adv_partials) + len(adv_failures)
  total = len(adv_successes) + len(adv_failures)
  print colored('== Adversary successes ==', 'green')
  print 'Sampled from %d/%d = %.2f%% exact match examples' % (
      len(adv_successes), total, 100.0 * len(adv_successes) / total)
  print
  print_exs(adv_successes[:OPTS.num_examples], predictions)
#  print colored('== Adversary partial successes ==', 'yellow')
#  print 'Sampled from %d/%d = %.2f%% exact match examples' % (
#      len(adv_partials), total, 100.0 * len(adv_partials) / total)
#  print
#  print_exs(adv_partials[:OPTS.num_examples], predictions)
  print colored('== Adversary failures ==', 'red')
  print 'Sampled from %d/%d = %.2f%% exact match examples' % (
      len(adv_failures), total, 100.0 * len(adv_failures) / total)
  print
  print_exs(adv_failures[:OPTS.num_examples], predictions)

def main():
  if OPTS.num_examples == 0:
    OPTS.num_examples = 1000000
  random.seed(OPTS.rng_seed)
  with open(OPTS.dataset_file) as dataset_file:
    dataset = json.load(dataset_file)
  with open(OPTS.prediction_file) as prediction_file:
    predictions = json.load(prediction_file)
  print_results(dataset, predictions)

if __name__ == '__main__':
  OPTS = parse_args()
  main()
