"""Running the BiDAF models."""
import atexit
import errno
import gzip
import json
import numpy as np
import os
import pickle
import requests
import socket
import sys
import subprocess
import time

import eval_squad

ROOT_DIR = 'bi-att-flow-dev'
GLOVE_DIR = 'glove_shortlists'
SAVE_DIR = 'save'
MODEL_NUM = 37  # Use this when doing single-model
ALL_MODEL_NUMS = [31, 33, 34, 35, 36, 37, 40, 41, 43, 44, 45, 46]
    # All models used by the ensemble
DEVNULL = open(os.devnull, 'w')
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 6050
SERVER_URL = 'http://%s:%d/query' % (SERVER_HOST, SERVER_PORT)
BEAM_SIZE = 10  # Beam size when computing approximate expected F1 score

def get_load_path(model_num):
  return os.path.join(SAVE_DIR, '%d' % model_num, 'save')

def get_shared_path(model_num):
  return os.path.join(SAVE_DIR, '%d' % model_num, 'shared.json')

def get_f1(prediction, answers):
  return eval_squad.metric_max_over_ground_truths(
      eval_squad.f1_score, prediction, answers)

def get_y_pred_beam(start_probs, end_probs, beam_size=BEAM_SIZE):
  beam = []
  for i, p_start in enumerate(start_probs):
    for j, p_end in enumerate(end_probs):
      if i <= j:
        beam.append((i, j + 1, p_start * p_end))
  beam.sort(key=lambda x: x[2], reverse=True)
  return beam[:beam_size]

def get_phrase(context, words, span):
  """Reimplementation of bi-att-flow.squad.utils.get_phrase."""
  start, stop = span
  char_idx = 0
  char_start, char_stop = None, None
  for word_idx, word in enumerate(words):
    char_idx = context.find(word, char_idx)
    if word_idx == start:
      char_start = char_idx
    char_idx += len(word)
    if word_idx == stop - 1:
      char_stop = char_idx
  return context[char_start:char_stop]


def get_expected_f1(start_probs, end_probs, context, words, answers):
  beam = get_y_pred_beam(start_probs, end_probs)
  total_prob = sum(x[2] for x in beam)
  score = 0.0
  for (start, end, prob) in beam:
    phrase = get_phrase(context, words, (start, end))
    cur_f1 = get_f1(phrase, answers)
    score += prob / total_prob * cur_f1
  return score


def run_single(json_filename, out_dir, verbose=False):
  """Run BiDAF (non-ensemble).
  
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
  inter_dir = os.path.join(out_dir, 'inter_single')
  os.mkdir(inter_dir)
  env = os.environ.copy()
  env['PYTHONPATH'] = ROOT_DIR
  prepro_args = [
      'python3', '-m', 'squad.prepro', '--mode', 'single', 
      '--single_path', json_filename, '-pm', '--target_dir', inter_dir, 
      '--glove_dir', GLOVE_DIR]
  subprocess.check_call(prepro_args, env=env, stdout=pipeout, stderr=pipeout)
  eval_pklgz_path = os.path.join(inter_dir, 'eval.pklgz')
  eval_json_path = os.path.join(inter_dir, 'eval.json')
  data_path = os.path.join(inter_dir, 'data_single.json')
  shared_path = os.path.join(inter_dir, 'shared_single.json')
  target_path = os.path.join(out_dir, 'preds.json')
  run_args = [
      'python3', '-m', 'basic.cli', '--data_dir', inter_dir, 
      '--eval_path', eval_pklgz_path, '--nodump_answer', 
      '--load_path', get_load_path(MODEL_NUM),
      '--shared_path', get_shared_path(MODEL_NUM),
      '--eval_num_batches', '0', '--mode', 'forward', '--batch_size', '1',
      '--len_opt', '--cluster', '--cpu_opt', '--load_ema']
  subprocess.check_call(run_args, env=env, stdout=pipeout, stderr=pipeout)
  ensemble_args = [
      'python3', '-m', 'basic.ensemble', 
      '--data_path', data_path, '--shared_path', shared_path,
      '-o', target_path, eval_pklgz_path]
  subprocess.check_call(ensemble_args, env=env, stdout=pipeout, stderr=pipeout)
  # Due to python2/3 incompatibilities, use python3 to convert pklgz to JSON
  subprocess.check_call([
      'python3', '-c', 
      'import gzip, json, pickle; json.dump(pickle.load(gzip.open("%s")), open("%s", "w", encoding="utf-8"))' % (eval_pklgz_path, eval_json_path)])
  with open(data_path) as f:
    data_single_obj = json.load(f)
  with open(eval_json_path) as f:
    eval_obj = json.load(f)
  with open(shared_path) as f:
    shared_single_obj = json.load(f)
  with open(target_path) as f:
    pred_obj = json.load(f)
  # Extract the scores
  id_list = data_single_obj['ids']
  scores = {}
  for i, cur_id in enumerate(id_list):
    # NOTE: we assume that everything is flattened
    # This corresponds to the -m flag in squad.prepro being set.
    start_probs = eval_obj['yp'][i][0]
    end_probs = eval_obj['yp2'][i][0]
    a_idx, p_idx = data_single_obj['*x'][i]
    context = shared_single_obj['p'][a_idx][p_idx]
    words = shared_single_obj['x'][a_idx][p_idx][0]
    answers = data_single_obj['answerss'][i]
    score = get_expected_f1(start_probs, end_probs, context, words, answers)
    scores[cur_id] = score
  return scores, pred_obj

def run_ensemble(json_filename, out_dir, verbose=False):
  """Run BiDAF ensemble.
  
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

  # Run preprocessing
  inter_dir = os.path.join(out_dir, 'inter_single')
  os.mkdir(inter_dir)
  env = os.environ.copy()
  env['PYTHONPATH'] = ROOT_DIR
  prepro_args = [
      'python3', '-m', 'squad.prepro', '--mode', 'single', 
      '--single_path', json_filename, '-pm', '--target_dir', inter_dir, 
      '--glove_dir', GLOVE_DIR]
  subprocess.check_call(prepro_args, env=env, stdout=pipeout, stderr=pipeout)

  # To avoid race conditions, create directories now
  out_dir = 'out/basic/00'
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  for subdir in ('answer', 'eval', 'log', 'save'):
    cur_dir = os.path.join(out_dir, subdir)
    if not os.path.exists(cur_dir):
      os.mkdir(cur_dir)

  # Launch all models in parallel
  eval_pklgz_paths = []
  procs = []
  for n in ALL_MODEL_NUMS:
    eval_pklgz_path = os.path.join(inter_dir, 'eval-%d.pklgz' % n)
    eval_pklgz_paths.append(eval_pklgz_path)
    cur_load_path = get_load_path(n)
    cur_shared_path = get_shared_path(n)
    run_args = [
        'python3', '-m', 'basic.cli', '--data_dir', inter_dir, 
        '--eval_path', eval_pklgz_path, '--nodump_answer', 
        '--load_path', cur_load_path, '--shared_path', cur_shared_path,
        '--eval_num_batches', '0', '--mode', 'forward', '--batch_size', '1',
        '--len_opt', '--cluster', '--cpu_opt', '--load_ema']
    proc = subprocess.Popen(run_args, env=env, stdout=pipeout, stderr=pipeout)
    procs.append(proc)

  # Wait for all models to finish
  for p in procs: p.wait()

  # Generate ensemble predictions
  shared_path = os.path.join(inter_dir, 'shared_single.json')
  data_path = os.path.join(inter_dir, 'data_single.json')
  target_path = os.path.join(out_dir, 'preds.json')
  ensemble_args = [
      'python3', '-m', 'basic.ensemble', 
      '--data_path', data_path, '--shared_path', shared_path,
      '-o', target_path] + eval_pklgz_paths
  subprocess.check_call(ensemble_args, env=env, stdout=pipeout, stderr=pipeout)

  # Due to python2/3 incompatibilities, use python3 to convert pklgz to JSON
  eval_objs = []
  for n in ALL_MODEL_NUMS:
    eval_pklgz_path = os.path.join(inter_dir, 'eval-%d.pklgz' % n)
    eval_json_path = os.path.join(inter_dir, 'eval-%d.json' % n)
    subprocess.check_call([
        'python3', '-c', 
        'import gzip, json, pickle; json.dump(pickle.load(gzip.open("%s")), open("%s", "w", encoding="utf-8"))' % (eval_pklgz_path, eval_json_path)])
    with open(eval_json_path) as f:
      eval_objs.append(json.load(f))

  # Compute expected F1
  with open(data_path) as f:
    data_single_obj = json.load(f)
  with open(shared_path) as f:
    shared_single_obj = json.load(f)
  with open(target_path) as f:
    pred_obj = json.load(f)
  # Extract the scores
  id_list = data_single_obj['ids']
  scores = {}
  for i, cur_id in enumerate(id_list):
    # NOTE: we assume that everything is flattened
    # This corresponds to the -m flag in squad.prepro being set.
    # 
    # Average all the start/end probs.
    start_probs = sum(np.array(eval_obj['yp'][i][0]) for eval_obj in eval_objs
                      )/ float(len(eval_objs))
    end_probs = sum(np.array(eval_obj['yp2'][i][0]) for eval_obj in eval_objs
                    )/ float(len(eval_objs))
    a_idx, p_idx = data_single_obj['*x'][i]
    context = shared_single_obj['p'][a_idx][p_idx]
    words = shared_single_obj['x'][a_idx][p_idx][0]
    answers = data_single_obj['answerss'][i]
    score = get_expected_f1(start_probs, end_probs, context, words, answers)
    scores[cur_id] = score
  return scores, pred_obj

def start_server(out_dir, verbose=False, nlp_cluster=False, train_dir=None):
  """Start BiDAF server, return the process once it's up.
  
  Args:
    out_dir: Directory for scratch work.
    verbose: If True, print subprocess output.
    nlp_cluster: If True, configure python3 to work on NLP cluster
    train_dir: If provided, use params from this training run
  """
  if verbose:
    pipeout = None
  else:
    pipeout = DEVNULL
  inter_dir = os.path.join(out_dir, 'inter_single')
  os.mkdir(inter_dir)
  env = os.environ.copy()
  env['PYTHONPATH'] = ROOT_DIR
  eval_pklgz_path = os.path.join(inter_dir, 'eval.pklgz')
  eval_json_path = os.path.join(inter_dir, 'eval.json')
  data_path = os.path.join(inter_dir, 'data_single.json')
  shared_path = os.path.join(inter_dir, 'shared_single.json')
  target_path = os.path.join(out_dir, 'preds.json')
  if nlp_cluster:
    env['LD_LIBRARY_PATH'] = 'libc/lib/x86_64-linux-gnu/:libc/usr/lib64/'
    python3_args = [
        'libc/lib/x86_64-linux-gnu/ld-2.17.so', 
        '/u/nlp/packages/anaconda2/envs/robinjia-py3/bin/python',
    ]
  else:
    python3_args = ['python3']
  if train_dir:
    save_args = [
        '--out_base_dir', train_dir,
        '--shared_path', os.path.join(train_dir, 'basic/00/shared.json'),
    ]
  else:
    save_args = [
        '--load_path', get_load_path(MODEL_NUM),
        '--shared_path', get_shared_path(MODEL_NUM),
    ]
  run_args = python3_args + [
      '-O', '-m', 'basic.cli', '--data_dir', inter_dir, 
      '--eval_path', eval_pklgz_path, '--nodump_answer', 
      '--eval_num_batches', '0', '--mode', 'server', '--batch_size', '1',
      '--len_opt', '--cluster', '--cpu_opt', '--load_ema'] + save_args
      # python3 -O disables assertions
  if verbose:
    print >> sys.stderr, run_args
  p = subprocess.Popen(run_args, env=env, stdout=pipeout, stderr=pipeout)
  atexit.register(p.terminate)

  # Keep trying to connect until the server is up
  s = socket.socket()
  while True:
    time.sleep(1)
    try:
      s.connect((SERVER_HOST, SERVER_PORT))
    except socket.error as e:
      if e.errno != errno.ECONNREFUSED:
        # Something other than Connection refused means server is running
        break
  s.close()
  return p


def query_server(dataset, verbose=False):
  """Query BiDAF server, which is assumed to be running."""
  response = requests.post(SERVER_URL, json=dataset)
  response_json = json.loads(response.text)
  data_single_obj = response_json['data_single']
  eval_obj = response_json['eval']
  shared_single_obj = response_json['shared_single']
  pred_obj = response_json['predictions']

  # Extract the scores
  id_list = data_single_obj['ids']
  scores = {}
  beams = {}
  for i, cur_id in enumerate(id_list):
    # NOTE: we assume that everything is flattened
    # This corresponds to the -m flag in squad.prepro being set.
    start_probs = eval_obj['yp'][i][0]
    end_probs = eval_obj['yp2'][i][0]
    a_idx, p_idx = data_single_obj['*x'][i]
    context = shared_single_obj['p'][a_idx][p_idx]
    words = shared_single_obj['x'][a_idx][p_idx][0]
    answers = data_single_obj['answerss'][i]
    raw_beam = get_y_pred_beam(start_probs, end_probs)
    phrase_beam = []
    for start, end, prob in raw_beam:
      phrase = get_phrase(context, words, (start, end))
      cur_f1 = get_f1(phrase, answers)
      phrase_beam.append((phrase, prob, cur_f1))
    beams[cur_id] = phrase_beam
    score = get_expected_f1(start_probs, end_probs, context, words, answers)
    scores[cur_id] = score
  return scores, pred_obj, beams

def debug_server(json_filename, out_dir, verbose=False, **kwargs):
  """Run BiDAF (non-ensemble), server mode for debugging.
  
  Args:
    json_filename: Name of JSON file with SQuAD data.
    out_dir: Directory for scratch work.
    verbose: If True, print subprocess output.
  Returns: tuple(scores, pred_obj)
    scores: map from example ID to current adversary score
    pred_obj: map from example ID to current prediction.
  """
  t0 = time.time()
  process = start_server(out_dir, verbose=verbose, **kwargs)
  t1 = time.time()
  if verbose:
    print >> sys.stderr, 'Server startup took %.2f seconds' % (t1 - t0)
  with open(json_filename) as f:
    dataset = json.load(f)
  scores, pred_obj, beams = query_server(dataset, verbose=verbose)
  t2 = time.time()
  if verbose:
    print >> sys.stderr, 'Query took %.2f seconds' % (t2 - t1)
  return scores, pred_obj
