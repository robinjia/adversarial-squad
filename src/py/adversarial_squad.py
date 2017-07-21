"""Run AddAny and AddCommon adversaries (plus others unpublished variants)."""
import argparse
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
import os
import random
import shutil
import string
import sys
import time

import bidaf
import eval_squad
import matchlstm

OPTS = None

OUT_DIR = 'adversarial_squad_out'
PUNCTUATION = set(string.punctuation) | set(['``', "''"])
MODES = [
    'basic',  # Only add common words
    'add-question-words',  # Add common words + force add question words
    'sample-nearby',  # Sample words near question words only
    'add-nearby',  # Add common words + Force add words near question words
    'mix-nearby',  # Sample mixture of words near question words and common words
]
MODELS = [
    'bidaf-single',
    'bidaf-ensemble',
    'matchlstm-single',
    'matchlstm-ensemble',
]

def parse_args():
  parser = argparse.ArgumentParser('Add random words to break SQuAD models.')
  parser.add_argument('mode', choices=MODES, help='What words to add.')
  parser.add_argument('model', choices=MODELS, help='Which model to run.')
  parser.add_argument('in_data', help='JSON data.')
  parser.add_argument('vocab_file', help='List of common English words.')
  parser.add_argument('--num-additions', '-n', default=10, type=int,
                      help='Number of random words to add.')
  parser.add_argument('--sample-num', '-k', default=10, type=int,
                      help='Number of random words to sample at each step.')
  parser.add_argument('--num-particles', '-p', default=[1],
                      type=lambda x: [int(t) for t in x.split(',')],
                      help='Comma separated list of particles per mega-epoch.')
  parser.add_argument('--num-nearby', '-q', default=1, type=int,
                      help='Number of nearby words per question word.')
  parser.add_argument('--num-epochs', '-T', default=1, type=int,
                      help='Number of epochs (change every word once).')
  parser.add_argument('--nearby-file', help='File with nearby words.')
  parser.add_argument('--out-json', '-o', 
                      help='Where to write output JSON file.')
  parser.add_argument('--plot-file', help='Where to write plot.')
  parser.add_argument('--rng_seed', '-s', default=0, type=int, help='RNG seed.')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  args = parser.parse_args()
  return args

def read_data(filename):
  with open(filename) as f:
    obj = json.load(f)
  return obj

def extract_question_info(dataset):
  info = {}
  for a in dataset['data']:
    for p in a['paragraphs']:
      for q in p['qas']:
        text = p['context']
        question = q['question']
        answers = [x['text'] for x in q['answers']]
        info[q['id']] = {
            'context': text,
            'question': question,
            'answers': answers
        }
  return info

def get_vocabularies(dataset, vocab_file, nearby_file):
  """Create map from example ID to (basic_words, nearby_words."""
  with open(vocab_file) as f:
    basic_vocab = [line.strip() for line in f]
  with open(nearby_file) as f:
    nearby_words = json.load(f)
  stemmer = LancasterStemmer()
  vocabs = {}
  for a in dataset['data']:
    for p in a['paragraphs']:
      for q in p['qas']:
        q_words = [w.lower() for w in word_tokenize(q['question'])]
        if OPTS.mode == 'basic':
          vocabs[q['id']] = (basic_vocab, [])
        elif OPTS.mode == 'add-question-words':
          vocabs[q['id']] = (basic_vocab, q_words)
        elif OPTS.mode.endswith('-nearby'):
          q_stems = [stemmer.stem(qw) for qw in q_words]
          cur_vocab = [w for w in basic_vocab if w not in q_stems]
          cur_nearby = []
          for q_word, q_stem in zip(q_words, q_stems):
            if q_word in nearby_words:
              qw_nearby = []
              for nearby_word in nearby_words[q_word]:
                if len(qw_nearby) == OPTS.num_nearby: break
                if nearby_word['word'] in PUNCTUATION: continue
                nearby_stem = stemmer.stem(nearby_word['word'])
                if nearby_stem != q_stem:
                  qw_nearby.append(nearby_word['word'])
              cur_nearby.extend(qw_nearby)
          vocabs[q['id']] = (cur_vocab, cur_nearby)
  return vocabs

def get_f1(prediction, answers):
  return eval_squad.metric_max_over_ground_truths(
      eval_squad.f1_score, prediction, answers)

def run_model(json_data, out_dir):
  """Run a model.

  Args:
    json_data: The JSON object containing SQuAD data.
    out_dir: A directory to store scratch work.
  Returns: tuple (scores, pred_obj)
    scores: map from example ID to current adversary score
    pred_obj: map from example ID to current prediction.
  """
  json_filename = os.path.join(out_dir, 'candidates.json')
  with open(json_filename, 'wb') as f:
    json.dump(json_data, f)
  if OPTS.model == 'bidaf-single':
    return bidaf.run_single(json_filename, out_dir)
  elif OPTS.model == 'bidaf-ensemble':
    return bidaf.run_ensemble(json_filename, out_dir)
  elif OPTS.model == 'matchlstm-single':
    return matchlstm.run_single(json_filename, out_dir)
  elif OPTS.model == 'matchlstm-ensemble':
    return matchlstm.run_ensemble(json_filename, out_dir)
  else:
    raise ValueError('Unrecognized model "%s"' % OPTS.model)


def test_original_data(dataset, question_info, out_dir):
  print 'Original data:'
  t0 = time.time()
  cur_out_dir = os.path.join(out_dir, 'orig')
  os.mkdir(cur_out_dir)
  scores, preds = run_model(dataset, cur_out_dir)
  results = eval_squad.evaluate_adversarial(dataset['data'], preds)
  t1 = time.time()
  print '  Time elapsed: %.2f seconds (for %d questions)' % (
      (t1 - t0), len(scores))
  total_score = sum(scores[k] for k in scores)
  print '  Total score: %0.4f/%d' % (total_score , len(scores))
  for k in results:
    print '  %s: %02f' % (k, results[k])
  for k in sorted(scores):
    info = question_info[k]
    print '    Example %s' % k
    print ('      Question: %s' % info['question']).encode('utf-8')
    print '      Score: %.4f' % scores[k]
    print ('      y_pred: "%s"' % preds[k]).encode('utf-8')
    print ('      y_gold: [%s]' % ', '.join(set(info['answers']))).encode('utf-8')
  return preds
  

def score_candidates(dataset, candidates, out_dir, mock=False):
  """Score a bunch of candidate additions for each example.
  
  Args:
    dataset: SQuAD dataset.
    candidates: A dict from example ID to a list of list of lists of added words.
    out_dir: Empty directory to write scratch output to.
    mock: If True, return scores of all 1's (faster, for testing)
  Returns:
    scores: A dict from example ID to a list of list of scores
    preds: A dict from example ID to a list of list of answer predictions
  """
  if len(candidates) == 0:
    return ({}, {})
  if mock:
    return {k: [1.0] * len(candidates[k]) for k in candidates}
  cur_data = []
  full_data = {'version': dataset['version'], 'data': cur_data}
  for a in dataset['data']:
    cur_paragraphs = []
    cur_article = {'title': a['title'], 'paragraphs': cur_paragraphs}
    for p in a['paragraphs']:
      for q in p['qas']:
        if q['id'] in candidates:
          for i, particle in enumerate(candidates[q['id']]):
            for j, candidate in enumerate(particle):
              cur_qa = {
                  'question': q['question'],
                  'id': '%s-part%04d-cand%04d' % (q['id'], i, j),
                  'answers': q['answers']
              }
              added_sent = ' '.join(candidate)
              if not(added_sent.endswith('.')):
                added_sent = added_sent + '.'
              cur_text = '%s %s' % (p['context'], added_sent)
              cur_paragraph = {'context': cur_text, 'qas': [cur_qa]}
              cur_paragraphs.append(cur_paragraph)
    if cur_paragraphs:
      cur_data.append(cur_article)
  raw_scores, raw_preds = run_model(full_data, out_dir)
  scores = {}
  preds = {}
  for ex_id in candidates:
    scores[ex_id] = []
    preds[ex_id] = []
    for i, particle in enumerate(candidates[ex_id]):
      particle_scores = []
      particle_preds = []
      for j in range(len(candidates[ex_id][i])):
        key = '%s-part%04d-cand%04d' % (ex_id, i, j)
        particle_scores.append(raw_scores[key])
        particle_preds.append(raw_preds[key])
      scores[ex_id].append(particle_scores)
      preds[ex_id].append(particle_preds)
  shutil.rmtree(out_dir)
  return scores, preds

def print_state(dataset, state, question_info, original_preds):
  best_particles = {k: min(state[k], key=lambda x: x[1]) for k in state}
  total_score = sum(best_particles[k][1] for k in state)
  adversarial_data = make_adversarial_json(dataset, state)
  preds = {'%s-adversarial' % k: best_particles[k][2] for k in state}
  preds.update(original_preds)
  results = eval_squad.evaluate_adversarial(adversarial_data['data'], preds)
  print '  Total score: %0.4f/%d' % (total_score , len(state))
  for k in results:
    print '  %s: %02f' % (k, results[k])
  for k in sorted(best_particles):
    info = question_info[k]
    words, score, pred = best_particles[k]
    print '    Example %s' % k
    print ('      Question: %s' % info['question']).encode('utf-8')
    print '      Score: %.4f' % score
    print ('      Add: "%s"' % ' '.join(words)).encode('utf-8')
    print ('      y_pred: "%s"' % pred).encode('utf-8')
    print ('      y_gold: [%s]' % ', '.join(set(info['answers']))).encode('utf-8')
  sys.stdout.flush()
  return results

def init_state(dataset, question_info, vocabs, out_dir, original_preds,
               cur_num_particles):
  """Create the initial state in the search process.

  The state is a dict from example ID to a tuple (words, score).
  |words| is a list of |num_additions| words in |vocab|.

  Args:
    dataset: SQuAD dataset.
    question_info: A dict from example ID to info.
    vocabs: dict from example ID to (basic_words, nearby_words)
  """
  print 'Initial state:'
  t0 = time.time()
  candidates = {}
  for a in dataset['data']:
    for p in a['paragraphs']:
      for q in p['qas']:
        cur_vocab, _ = vocabs[q['id']]
        candidates[q['id']] = [[random.sample(cur_vocab, OPTS.num_additions)]
                               for i in range(cur_num_particles)]
  cur_out_dir = os.path.join(out_dir, 'init')
  os.mkdir(cur_out_dir)
  scores, preds = score_candidates(dataset, candidates, cur_out_dir)
  state = {k: [(c[0], s[0], p[0]) 
               for c, s, p in zip(candidates[k], scores[k], preds[k])]
           for k in candidates}
  t1 = time.time()
  num_candidates = sum(len(x) for k in candidates for x in candidates[k])
  print '  Time elapsed: %.2f seconds (for %d candidates)' % (
      (t1 - t0), num_candidates)
  results = print_state(dataset, state, question_info, original_preds)
  return state, results

def reinit_state(dataset, state, question_info, vocabs, out_dir, original_preds,
                 cur_num_particles):
  """Re-initialize the state with more particles for unsolved examples."""
  print 'Reinitialize to %d particles' % cur_num_particles
  t0 = time.time()
  candidates = {}
  for k in state:
    min_words, min_score, min_pred = min(state[k], key=lambda x: x[1])
    min_f1 = get_f1(min_pred, question_info[k]['answers'])
    if min_f1 == 0: continue
    num_new = cur_num_particles - len(state[k])
    cur_vocab, _ = vocabs[k]
    candidates[k] = [[random.sample(cur_vocab, OPTS.num_additions)]
                     for i in range(num_new)]
  scores, preds = score_candidates(dataset, candidates, out_dir)
  for k in candidates:
    for c, s, p in zip(candidates[k], scores[k], preds[k]):
      state[k].append((c[0], s[0], p[0]))
  t1 = time.time()
  num_candidates = sum(len(x) for k in candidates for x in candidates[k])
  print '  Time elapsed: %.2f seconds (for %d candidates)' % (
      (t1 - t0), num_candidates)
  results = print_state(dataset, state, question_info, original_preds)


def sample_at_most(x, n):
  """Sample up to n elements of x, or all of x."""
  if len(x) <= n:
    return list(x)
  else:
    return random.sample(x, n)

def gen_swap_words(basic_words, nearby_words):
  """Randomly generate list of words to swap in."""
  if OPTS.mode.startswith('mix-'):
    num_basic = OPTS.sample_num / 2
    num_nearby = OPTS.sample_num - num_basic
    swap_words = (sample_at_most(basic_words, num_basic) + 
                  sample_at_most(nearby_words, num_nearby))
  elif OPTS.mode == 'sample-nearby':
    swap_words = sample_at_most(nearby_words, OPTS.sample_num)
  else:  # 'basic', 'add-question-words', 'add-nearby'
    swap_words = sample_at_most(basic_words, OPTS.sample_num)
  if OPTS.mode.startswith('add-'):
    for w in nearby_words:
      if w not in swap_words:
        swap_words.append(w)
  return swap_words

def gen_candidates(state, question_info, vocabs, i):
  """Generate candidate steps away the current state.

  Args:
    state: dict from example ID to (words, score, pred)
    question_info: dict from example ID to info
    vocabs: dict from example ID to (basic_words, nearby_words)
    i: index within |words| to mutate
  """
  candidates = {}
  for k in state:
    min_words, min_score, min_pred = min(state[k], key=lambda x: x[1])
    min_f1 = get_f1(min_pred, question_info[k]['answers'])
    if min_f1 == 0: continue
    cur_basic_words, cur_nearby_words = vocabs[k]
    candidates[k] = []
    for words, score, pred in state[k]:
      swap_words = gen_swap_words(cur_basic_words, cur_nearby_words)
      cur_cands = []
      for new_word in swap_words:
        new_words = list(words)
        new_words[i] = new_word
        cur_cands.append(new_words)
      candidates[k].append(cur_cands)
  return candidates

def plot_results(all_results, plot_file):
  f1s = [r['adv_f1'] for r in all_results]
  ems = [r['adv_exact_match'] for r in all_results]
  f1_h, = plt.plot(f1s, 'ro-', label='F1')
  em_h, = plt.plot(ems, 'bo-', label='Exact Match')
  plt.legend(handles=[f1_h, em_h], loc='best')
  axes = plt.gca()
  axes.set_ylim([0.0, 100.0])
  plt.title('Adversarial Evaluation')
  plt.xlabel('Number of Iterations')
  plt.ylabel('Percent Score')
  plt.xticks(range(0, len(all_results), OPTS.num_additions))  # Each epoch
  plt.savefig(plot_file)

def do_search(dataset, state, question_info, vocabs, out_dir, original_preds,
              all_results):
  for mega_epoch, cur_num_particles in enumerate(OPTS.num_particles):
    if mega_epoch > 0:
      cur_out_dir = os.path.join(out_dir, 'reinit_%d' % mega_epoch)
      os.mkdir(cur_out_dir)
      reinit_state(dataset, state, question_info, vocabs, cur_out_dir,
                   original_preds, cur_num_particles)
    for epoch in range(OPTS.num_epochs):
      inds = range(OPTS.num_additions)
      random.shuffle(inds)
      for t, i in enumerate(inds):
        print 'Mega-Epoch %02d, Epoch %02d, Round %02d (modify index %d):' % (
            mega_epoch, epoch, t, i)
        t0 = time.time()
        candidates = gen_candidates(state, question_info, vocabs, i)
        cur_out_dir = os.path.join(out_dir, 'mega%02d_epoch%02d_round%02d' % (
            mega_epoch, epoch, t))
        os.mkdir(cur_out_dir)
        scores, preds = score_candidates(dataset, candidates, cur_out_dir)
        new_state = {}
        for k in state:
          if k not in candidates:
            new_state[k] = state[k]
            continue
          new_state[k] = []
          for j, (cur_cands, cur_scores, cur_preds) in enumerate(
              zip(candidates[k], scores[k], preds[k])):
            min_ind, min_score = min(enumerate(cur_scores), key=lambda x: x[1])
            if min_score < state[k][j][1]:
              new_state[k].append((cur_cands[min_ind], min_score,
                                   cur_preds[min_ind]))
            else:
              new_state[k].append(state[k][j])
        state = new_state
        t1 = time.time()
        num_candidates = sum(len(x) for k in candidates for x in candidates[k])
        print '  Time elapsed: %.2f seconds (for %d candidates)' % (
            (t1 - t0), num_candidates)
        cur_results = print_state(dataset, state, question_info, original_preds)
        all_results.append(cur_results)
        if OPTS.plot_file:
          plot_results(all_results, OPTS.plot_file)
  return state

def make_adversarial_json(dataset, state, filename=None):
  """Write output JSON data for adversarial evaluation script."""
  cur_data = []
  full_data = {'version': dataset['version'], 'data': cur_data}
  for a in dataset['data']:
    cur_paragraphs = []
    cur_article = {'title': a['title'], 'paragraphs': cur_paragraphs}
    cur_data.append(cur_article)
    for p in a['paragraphs']:
      cur_paragraphs.append(p)  # Keep the original data
      for q in p['qas']:
        words, score, pred = min(state[q['id']], key=lambda x: x[1])
        cur_qa = {
            'question': q['question'],
            'id': '%s-adversarial' % q['id'],
            'answers': q['answers']
        }
        added_sent = ' '.join(words)
        if not(added_sent.endswith('.')):
          added_sent = added_sent + '.'
        cur_text = '%s %s' % (p['context'], added_sent)
        cur_paragraph = {'context': cur_text, 'qas': [cur_qa]}
        cur_paragraphs.append(cur_paragraph)
  if filename:
    with open(filename, 'wb') as f:
      json.dump(full_data, f)
  else:
    return full_data

def run():
  out_dir = os.path.join(OUT_DIR, 'out_%s_%s_n%d_k%d_p%s_T%d_s%d' % (
      OPTS.mode, OPTS.model, OPTS.num_additions, OPTS.sample_num, 
      ','.join(str(x) for x in OPTS.num_particles),
      OPTS.num_epochs, OPTS.rng_seed))
  if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
  os.makedirs(out_dir)
  dataset = read_data(OPTS.in_data)
  question_info = extract_question_info(dataset)
  vocabs = get_vocabularies(dataset, OPTS.vocab_file, OPTS.nearby_file)
  original_preds = test_original_data(dataset, question_info, out_dir)
  start_state, start_results = init_state(
      dataset, question_info, vocabs, out_dir, original_preds,
      OPTS.num_particles[0])
  all_results = [start_results]
  final_state = do_search(dataset, start_state, question_info, vocabs, out_dir,
                          original_preds, all_results)
  if OPTS.out_json:
    make_adversarial_json(dataset, final_state, filename=OPTS.out_json)

def main():
  random.seed(OPTS.rng_seed)
  run()

if __name__ == '__main__':
  OPTS = parse_args()
  main()
