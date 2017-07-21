"""Running the Match-LSTM models."""
import glob
import json
import numpy as np
import os
import sys
import subprocess

import eval_squad

ROOT_DIR = 'matchlstm'
GLOVE_DIR = 'glove_shortlists'
MAX_SPAN_LEN = 15  # from matchlstm/bpointBEMlstm.lua:predict
BEAM_SIZE = 10  # Beam size when computing approximate expected F1 score
MODEL_INDICES = (1, 2, 3, 4, 5)  # Models used by ensemble
DEVNULL = open(os.devnull, 'w')

def symlink_files(out_dir):
  """Symlink files from ROOT_DIR to the current |out_dir|."""
  for fname in glob.glob(os.path.join(ROOT_DIR, '*')):
    in_path = os.path.join(os.getcwd(), fname)
    out_path = os.path.join(out_dir, os.path.basename(fname))
    os.symlink(in_path, out_path)

def read_model_scores(model_index=None):
  """Read model_scores.txt."""
  model_scores = []
  if model_index is None:
    filename = 'model_scores.txt'
  else:
    filename = 'model_%d_scores.txt' % model_index
  with open(filename) as f:
    for line in f:
      s_start, s_end = [[float(x) for x in t.split()]
                        for t in line.rstrip().split('\t')]
      model_scores.append((s_start, s_end))
  return model_scores

def read_data_tokens():
  """Read data_token.txt."""
  data_token_list = []
  with open('data_token.txt') as f:
    for line in f:
      p_toks, q_toks = [t.split() for t in line.rstrip().split('\t')]
      data_token_list.append((p_toks, q_toks))
  return data_token_list

def read_data(data):
  id_list = []
  answers = {}
  for a in data['data']:
    for p in a['paragraphs']:
      for q in p['qas']:
        id_list.append(q['id'])
        cur_answers = [x['text'] for x in q['answers']]
        answers[q['id']] = cur_answers
  return id_list, answers

def get_y_pred_beam(start_scores, end_scores, beam_size=BEAM_SIZE):
  beam = []
  for i, s_start in enumerate(start_scores):
    for j in range(MAX_SPAN_LEN + 1):
      if i + j >= len(end_scores): continue
      cur_score = start_scores[i] + end_scores[i + j]
      beam.append((i, i + j, cur_score))
  beam.sort(key=lambda x: x[2], reverse=True)
  return beam[:beam_size]

def get_phrase(data_tokens, start, end):
  return ' '.join(data_tokens[0][start:end+1])

def get_f1(prediction, answers):
  return eval_squad.metric_max_over_ground_truths(
      eval_squad.f1_score, prediction, answers)

def get_expected_f1(start_scores, end_scores, data_tokens, answer,
                    temperature=1.0):
  beam = get_y_pred_beam(start_scores, end_scores)
  max_score = beam[0][2]
  partition = sum(np.exp((x[2] - max_score) / temperature) for x in beam)
  adv_score = 0.0
  for start, end, score in beam:
    phrase = get_phrase(data_tokens, start, end)
    cur_f1 = get_f1(phrase, answer)
    prob = np.exp((score - max_score) / temperature) / partition
    adv_score += prob * cur_f1
  return adv_score

def run_single(json_filename, out_dir, verbose=False):
  """Run Match LSTM (non-ensemble, bi-Ans-Ptr).
  
  Args:
    json_filename: Name of JSON file with SQuAD data.
    out_dir: Directory for scratch work.
    verbose: If True, print subprocess output.
  Returns: tuple(scores, pred_obj)
    scores: map from example ID to current adversary score
    pred_obj: map from example ID to current prediction.
  """
  if verbose:
    pipeout = None
  else:
    pipeout = DEVNULL
  with open(json_filename) as f:
    data = json.load(f)
  initial_dir = os.getcwd()
  # Set absolute paths
  json_filename = os.path.join(initial_dir, json_filename)
  out_dir = os.path.join(initial_dir, out_dir)
  symlink_files(out_dir)
  os.chdir(out_dir)
  env = os.environ.copy()
  env['Glove_DATA'] = os.path.join(initial_dir, GLOVE_DIR)
  subprocess.check_call(['th', 'main4.lua', '-input', json_filename],
                        env=env, stdout=pipeout, stderr=pipeout)
  model_scores = read_model_scores()
  data_token_list = read_data_tokens()
  id_list, answers = read_data(data)
  adv_scores = {}
  for qid, data_tokens, (s_start, s_end) in zip(
      id_list, data_token_list, model_scores):
    answer = answers[qid]
    adv_score = get_expected_f1(s_start, s_end, data_tokens, answer,
                                temperature=2.0)
        # Raw scores are sum of forward/backward scores.
        # To have similar sharpness, should average instead of summing,
        # hence temperature of 2.0
    adv_scores[qid] = adv_score
  with open('prediction.json') as f:
    pred_obj = json.load(f)
  os.chdir(initial_dir)
  return adv_scores, pred_obj

def run_ensemble(json_filename, out_dir, verbose=False):
  """Run Match LSTM ensemble, Ans-Ptr.
  
  Args:
    json_filename: Name of JSON file with SQuAD data.
    out_dir: Directory for scratch work.
    verbose: If True, print subprocess output.
  Returns: tuple(scores, pred_obj)
    scores: map from example ID to current adversary score
    pred_obj: map from example ID to current prediction.
  """
  if verbose:
    pipeout = None
  else:
    pipeout = DEVNULL
  with open(json_filename) as f:
    data = json.load(f)
  initial_dir = os.getcwd()
  # Set absolute paths
  json_filename = os.path.join(initial_dir, json_filename)
  out_dir = os.path.join(initial_dir, out_dir)
  symlink_files(out_dir)
  os.chdir(out_dir)
  env = os.environ.copy()
  env['Glove_DATA'] = os.path.join(initial_dir, GLOVE_DIR)

  # Run preprocessing
  subprocess.check_call(['python', 'js2tokens.py', json_filename],
                        env=env, stdout=pipeout, stderr=pipeout)

  # Launch all models in parallel
  procs = []
  for i in MODEL_INDICES:
    run_args = [
        'th', 'main4.lua', '-model', 'pointBEMlstm',
        '-modelSaved', 'model_BE_%d' % i, '-input', json_filename]
    proc = subprocess.Popen(run_args, env=env, stdout=pipeout, stderr=pipeout)
    procs.append(proc)
  for p in procs: p.wait()

  # Ensemble to get predictions
  subprocess.check_call(['th', 'ensemble.lua'], 
                        env=env, stdout=pipeout, stderr=pipeout)
  subprocess.check_call(
      ['python', 'txt2js.py', json_filename, 'test_output.txt'],
      env=env, stdout=pipeout, stderr=pipeout)
  with open('prediction.json') as f:
    pred_obj = json.load(f)

  # Combine all model scores by averaging 
  id_list, answers = read_data(data)
  model_scores = [None] * len(id_list)
  for i in MODEL_INDICES:
    cur_model_scores = read_model_scores(i)
    for j in range(len(model_scores)):
      s_start, s_end = [np.array(x) for x in cur_model_scores[j]]
      if model_scores[j]:
        model_scores[j][0] += s_start
        model_scores[j][1] += s_end
      else:
        model_scores[j] = [s_start, s_end]
  for j in range(len(model_scores)):
    model_scores[j][0] /= len(MODEL_INDICES)
    model_scores[j][1] /= len(MODEL_INDICES)

  # Generate scores
  data_token_list = read_data_tokens()
  adv_scores = {}
  for qid, data_tokens, (s_start, s_end) in zip(
      id_list, data_token_list, model_scores):
    answer = answers[qid]
    adv_score = get_expected_f1(s_start, s_end, data_tokens, answer)
    adv_scores[qid] = adv_score

  os.chdir(initial_dir)
  return adv_scores, pred_obj
