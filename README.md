# Adversarial Examples for Evaluating Reading Comprehension Systems (EMNLP 2017)
This repository contains code for the paper:

> Adversarial Examples for Evaluating Reading Comprehension Systems.  
> Robin Jia and Percy Liang  
> Empirical Methods in Natural Language Processing (EMNLP), 2017.  

*Note:* if you want to run adversarial evaluation on your own SQuAD model, please see 
[our Codalab worksheet](https://worksheets.codalab.org/worksheets/0xc86d3ebe69a3427d91f9aaa63f7d1e7d/)
for instructions.
This git repository exposes the code that was used to generate some of the files 
on that Codalab worksheet.

## Dependencies
Run `pull-dependencies.sh` to pull SQuAD data, GloVe vectors, Stanford CoreNLP,
and some custom python utilities.
Other python requirements are in `requirements.txt`.

## Examples
The following sequence of commmands generates the raw AddSent training data described in Section 4.6 of our paper.

    mkdir out
    # Precompute nearby words in word vector space; takes roughly 1 hour
    python src/py/find_squad_nearby_words.py glove/glove.6B.100d.txt -n 100 -f data/squad/train-v1.1.json > out/nearby_n100_glove_6B_100d.json
    # Run CoreNLP on the SQuAD training data; takes roughly 1 hour, uses ~18GB memory
    python src/py/convert_questions.py corenlp -d train
    # Actually generate the raw AddSent examples; takes roughly 7 minutes, uses ~15GB memory
    python src/py/convert_questions.py dump-highConf -d train -q

The final script will generate three files with prefix `train-convHighConf` in the `out` directory, 
including `train-convHighConf.json`.
`train-convHighConf-mturk.tsv` is in a format that can be processed by scripts in the `mturk` directory.

Other one-off scripts are described in their docstrings.
