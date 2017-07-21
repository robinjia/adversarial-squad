"""Server to test mutated versions on SQuAD models."""
import argparse
import bottle
import cgi
import difflib
import json
import os
import random
import shutil
import sys

import bidaf

OPTS = None

SQUAD_DEV_FILE = 'data/squad/dev-v1.1.json'

def parse_args():
  parser = argparse.ArgumentParser('Insert a description of this script.')
  parser.add_argument('out_dir', help='Temporary output directory.')
  parser.add_argument('--hostname', '-n', default='0.0.0.0', help='hostname.')
  parser.add_argument('--port', '-p', default=9000, type=int, help='port.')
  parser.add_argument('--debug', '-d', default=False, action='store_true', help='Debug mode')
  parser.add_argument('--filename', '-f',
                      help=('Process each sentence in the given file. '
                            '(defaults to dev file).'),
                      default=SQUAD_DEV_FILE)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def render_diff(old_text, new_text):
  print (old_text, old_text.__class__)
  print (new_text, new_text.__class__)
  sm = difflib.SequenceMatcher(a=old_text, b=new_text)
  out_toks = []
  for opcode, s1, e1, s2, e2 in sm.get_opcodes():
    if opcode == 'equal':
      out_toks.append(old_text[s1:e1])
    elif opcode == 'insert':
      out_toks.append('<span class="insert">' + new_text[s2:e2] + '</span>')
    elif opcode == 'delete':
      out_toks.append('<span class="delete">' + old_text[s1:e1] + '</span>')
    elif opcode == 'replace':
      out_toks.append('<span class="delete">' + old_text[s1:e1] + '</span>')
      out_toks.append('<span class="insert">' + new_text[s2:e2] + '</span>')
  return ''.join(out_toks)

def run_model(dataset, article, paragraph, qa, new_paragraph=None,
                      new_question=None):
  """Query the model on a single example."""
  if not new_paragraph:
    new_paragraph = paragraph['context']
  if not new_question:
    new_question = qa['question']
  query_data = {
      'version': dataset['version'],
      'data': [{
          'title': article['title'],
          'paragraphs': [{
              'context': new_paragraph,
              'qas': [{
                  'question': new_question,
                  'id': qa['id'],
                  'answers': qa['answers']
              }]
          }]
      }]
  }
  scores, preds, beams = bidaf.query_server(query_data)
  score = scores[qa['id']]
  beam = beams[qa['id']]
  return score, beam[:5]

def main():
  if os.path.exists(OPTS.out_dir):
    shutil.rmtree(OPTS.out_dir)
  os.makedirs(OPTS.out_dir)
  with open(OPTS.filename) as f:
    dataset = json.load(f)
  num_articles = len(dataset['data'])
  print >> sys.stderr, 'Starting BiDAF server...'
  bidaf.start_server(OPTS.out_dir, verbose=True)
  app = bottle.Bottle()

  @app.route('/')
  def index():
    return bottle.template('index', dataset=dataset)

  @app.route('/inspect/<a_ind:int>/<p_ind:int>/<q_ind:int>')
  def inspect(a_ind, p_ind, q_ind):
    # 1-index as much as possible, because it looks nicer
    article = dataset['data'][a_ind - 1]
    num_paragraphs = len(article['paragraphs'])
    paragraph = article['paragraphs'][p_ind - 1]
    num_qas = len(paragraph['qas'])
    qa = paragraph['qas'][q_ind - 1]  
    qid = qa['id']
    answers = ', '.join(set(a['text'] for a in qa['answers']))
    score, beam = run_model(dataset, article, paragraph, qa)
    return bottle.template(
        'inspect', a_ind=a_ind, p_ind=p_ind, q_ind=q_ind, qid=qid,
        num_articles=num_articles, num_paragraphs=num_paragraphs,
        num_qas=num_qas,
        article=article['title'].replace('_', ' '),
        paragraph_diff=cgi.escape(paragraph['context']),
        paragraph_text=paragraph['context'],
        question_diff = cgi.escape(qa['question']),
        question_text=qa['question'],
        answers=answers, score=score, beam=beam)

  @app.route('/random')
  def get_random():
    a_ind = random.sample(range(num_articles), 1)[0]
    article = dataset['data'][a_ind]
    p_ind = random.sample(range(len(article['paragraphs'])), 1)[0]
    paragraph = article['paragraphs'][p_ind]
    q_ind = random.sample(range(len(paragraph['qas'])), 1)[0]
    return bottle.redirect('/inspect/%d/%d/%d' % (a_ind + 1, p_ind + 1, q_ind + 1))

  @app.route('/post_query', method='post')
  def post_query():
    a_ind = bottle.request.forms.get('a_ind', type=int)
    p_ind = bottle.request.forms.get('p_ind', type=int)
    q_ind = bottle.request.forms.get('q_ind', type=int)
    new_paragraph = bottle.request.forms.getunicode('new_paragraph').strip()
    new_question = bottle.request.forms.getunicode('new_question').strip()
    # 1-index as much as possible, because it looks nice
    article = dataset['data'][a_ind - 1]
    num_paragraphs = len(article['paragraphs'])
    orig_paragraph = article['paragraphs'][p_ind - 1]
    num_qas = len(orig_paragraph['qas'])
    orig_qa = orig_paragraph['qas'][q_ind - 1]
    qid = orig_qa['id']
    answers = ', '.join(set(a['text'] for a in orig_qa['answers']))
    new_paragraph_diff = render_diff(
        cgi.escape(orig_paragraph['context']),
        cgi.escape(new_paragraph))
    new_question_diff = render_diff(
        cgi.escape(orig_qa['question']),
        cgi.escape(new_question))
    score, beam = run_model(dataset, article, orig_paragraph, orig_qa,
                            new_paragraph=new_paragraph,
                            new_question=new_question)
    return bottle.template(
        'inspect', a_ind=a_ind, p_ind=p_ind, q_ind=q_ind, qid=qid,
        num_articles=num_articles, num_paragraphs=num_paragraphs,
        num_qas=num_qas,
        article=article['title'].replace('_', ' '),
        paragraph_diff=new_paragraph_diff,
        paragraph_text=new_paragraph,
        question_diff=new_question_diff,
        question_text=new_question,
        answers=answers, score=score, beam=beam)

  bottle.run(app, host=OPTS.hostname, port=OPTS.port, debug=OPTS.debug)


if __name__ == '__main__':
  OPTS = parse_args()
  main()

