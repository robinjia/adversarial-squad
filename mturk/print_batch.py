"""Various tools for dealing with HIT results."""
import argparse
import collections
import json
import random
import sys
from termcolor import colored
import unicodecsv as csv

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Various tools for dealing with HIT results.')
  parser.add_argument('filename', help='Batch CSV file.')
  parser.add_argument('mode', help='Options: [%s]' % ', '.join(MODES))
  parser.add_argument('--worker', '-w', help='Filter to this worker ID')
  parser.add_argument('--ensemble', '-e', default=False, action='store_true',
                      help='Ensemble humans')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def dump_grammar():
  with open(OPTS.filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      if OPTS.worker and row['WorkerId'] != OPTS.worker: continue
      if row['AssignmentStatus'] == 'Rejected': continue
      print 'HIT %s' % row['HITId']
      print 'WorkerId: %s' % row['WorkerId']
      print 'Time: %s s' % row['WorkTimeInSeconds']
      input_qids = row['Input.qids'].split('\t')
      input_sents = row['Input.sents'].split('\t')
      ans_is_good = row['Answer.is-good'].split('\t')
      ans_responses = row['Answer.responses'].split('\t')
      for qid, s, is_good, response in zip(input_qids, input_sents, 
                                           ans_is_good, ans_responses):
        print ('  Example %s' % qid)
        print ('    Sentence: %s' % s).encode('utf-8')
        print ('    Is good?  %s' % is_good)
        print ('    Response: %s' % colored(response, 'cyan')).encode('utf-8')


def stats_grammar():
  # Read data
  worker_to_is_good = collections.defaultdict(list)
  worker_to_times = collections.defaultdict(list)
  with open(OPTS.filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      if row['AssignmentStatus'] == 'Rejected': continue
      worker_id = row['WorkerId']
      ans_is_good = row['Answer.is-good'].split('\t')
      time = float(row['WorkTimeInSeconds'])
      worker_to_is_good[worker_id].extend(ans_is_good)
      worker_to_times[worker_id].append(time)

  # Aggregate by worker
  print '%d total workers' % len(worker_to_times)
  worker_stats = {}
  for worker_id in worker_to_times:
    times = sorted(worker_to_times[worker_id])
    t_median = times[len(times)/2]
    t_mean = sum(times) / float(len(times))
    is_good_list = worker_to_is_good[worker_id]
    num_qs = len(is_good_list)
    frac_good = sum(1.0 for x in is_good_list if x == 'yes') / num_qs
    worker_stats[worker_id] = (t_median, t_mean, num_qs, frac_good)

  # Print
  sorted_ids = sorted(list(worker_stats), key=lambda x: worker_stats[x][3])
  for worker_id in sorted_ids:
    t_median, t_mean, num_qs, frac_good = worker_stats[worker_id]
    print 'Worker %s: t_median %.1f, t_mean %.1f, %d questions, %.1f%% good' % (
        worker_id, t_median, t_mean, num_qs, 100.0 * frac_good)


def sent_format(s):
  s = s.replace("<span class='answer'>", '').replace("</span>", '')
  s = s.strip()
  if not s.endswith('.'):
    s = s + '.'
  return s[0].upper() + s[1:]

def dump_verify():
  with open(OPTS.filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      if OPTS.worker and row['WorkerId'] != OPTS.worker: continue
      if row['AssignmentStatus'] == 'Rejected': continue
      print 'HIT %s' % row['HITId']
      print 'WorkerId: %s' % row['WorkerId']
      print 'Time: %s s' % row['WorkTimeInSeconds']
      qids = row['Input.qids'].split('\t')
      questions = row['Input.questions'].split('\t')
      sents = row['Answer.sents'].split('\t')
      responses = row['Answer.responses'].split('\t')
      for qid, q, s_str, response_str in zip(
          qids, questions, sents, responses):
        print ('  Example %s' % qid)
        print ('  Question %s' % q)
        s_list = s_str.split('|')
        a_list = response_str.split('|')
        for s, a in zip(s_list, a_list):
          print ('    Sentence: %s' % sent_format(s)).encode('utf-8')
          print ('    Is good?  %s' % colored(a, 'cyan'))


def stats_verify():
  # Read data
  worker_to_is_good = collections.defaultdict(list)
  worker_to_times = collections.defaultdict(list)
  with open(OPTS.filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      if row['AssignmentStatus'] == 'Rejected': continue
      worker_id = row['WorkerId']
      ans_is_good = [x for s in row['Answer.responses'].split('\t')
                     for x in s.split('|')]
      time = float(row['WorkTimeInSeconds'])
      worker_to_is_good[worker_id].extend(ans_is_good)
      worker_to_times[worker_id].append(time)

  # Aggregate by worker
  print '%d total workers' % len(worker_to_times)
  worker_stats = {}
  for worker_id in worker_to_times:
    times = sorted(worker_to_times[worker_id])
    t_median = times[len(times)/2]
    t_mean = sum(times) / float(len(times))
    is_good_list = worker_to_is_good[worker_id]
    num_qs = len(is_good_list)
    frac_good = sum(1.0 for x in is_good_list if x == 'yes') / num_qs
    worker_stats[worker_id] = (t_median, t_mean, num_qs, frac_good)

  # Print
  sorted_ids = sorted(list(worker_stats), key=lambda x: worker_stats[x][3])
  for worker_id in sorted_ids:
    t_median, t_mean, num_qs, frac_good = worker_stats[worker_id]
    print 'Worker %s: t_median %.1f, t_mean %.1f, %d questions, %.1f%% good' % (
        worker_id, t_median, t_mean, num_qs, 100.0 * frac_good)


def dump_human_eval():
  with open(OPTS.filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      if OPTS.worker and row['WorkerId'] != OPTS.worker: continue
      print 'Example %s' % row['Input.qid']
      print '  WorkerId: %s' % row['WorkerId']
      print '  Time: %s s' % row['WorkTimeInSeconds']
      print ('  Paragraph: %s' % row['Input.paragraph']).encode('utf-8')
      print ('  Question: %s' % colored(row['Input.question'], 'cyan')).encode('utf-8')
      print ('  Response: %s' % colored(row['Answer.response'], 'yellow')).encode('utf-8')

def pred_human_eval():
  all_preds = collections.defaultdict(list)
  with open(OPTS.filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      all_preds[row['Input.qid']].append(row['Answer.response'])
  preds = {}
  for qid in all_preds:
    if OPTS.ensemble:
      for a in all_preds[qid]:
        count = sum(1 for pred in all_preds[qid] if a == pred)
        if count > 1:
          preds[qid] = a
          break
      else:
        preds[qid] = random.sample(all_preds[qid], 1)[0]
    else:
      preds[qid] = random.sample(all_preds[qid], 1)[0]
  print json.dumps(preds)

MODES = collections.OrderedDict([
    ('dump-grammar', dump_grammar),
    ('stats-grammar', stats_grammar),
    ('dump-he', dump_human_eval),
    ('pred-he', pred_human_eval),
    ('dump-verify', dump_verify),
    ('stats-verify', stats_verify),
])


def main():
  random.seed(0)
  f = MODES[OPTS.mode]
  f()


if __name__ == '__main__':
  OPTS = parse_args()
  main()

