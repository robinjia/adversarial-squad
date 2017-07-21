
import sys 
reload(sys) 
sys.setdefaultencoding('utf-8')
import json
import string
from nltk.tokenize import word_tokenize

fp = open(sys.argv[1],'r')
line = fp.readline()
js = json.loads(line)
fpr = open(sys.argv[2],'r')

predictions = {}
for c in js["data"]:
	for p in c["paragraphs"]:
		context = p["context"]
		context_word = word_tokenize(context)
		for qa in p["qas"]:
			question_id = qa["id"]
			
			pred = fpr.readline().strip()
			newpred = []
			pre_word = ''
			words = pred.split()
			for word in words:
				word = word.replace("``", "\"")
				word = word.replace("''", "\"")
				if pre_word != '' :
					if newpred[-1]+word in context:
						newpred[-1] = newpred[-1]+word
					else:
						newpred.append(word)
				else:
					newpred.append(word)
				pre_word = word
			predictions[question_id] = ' '.join(newpred)
			
predictions_js = json.dumps(predictions)
fpw = open('prediction.json', 'w')
fpw.write(predictions_js)
fpw.close()
fp.close()
fpr.close()
