import sys 
reload(sys) 
sys.setdefaultencoding('utf-8')
import json
import string
from nltk.tokenize import word_tokenize

fp = open(sys.argv[1],'r')
line = fp.readline()
js = json.loads(line)
output = ''
fpw = open('data_token.txt','w')
for c in js["data"]:
	for p in c["paragraphs"]:
		context = word_tokenize(p["context"])
		
		for qa in p["qas"]:
			
			question = word_tokenize(qa["question"])
			fpw.write(' '.join(context)+'\t'+' '.join(question)+'\n')
			#print (' '.join(context)+'\t'+' '.join(question))
fp.close()
fpw.close()
