import json
import pdb
from deeppavlov import build_model, configs
model = build_model(configs.squad.squad, download=True)

file = json.load(open("dev.json","r"))
dev_eval = []

print("total number of sequence: ",len(file))
for i in range(len(file)):
	if i%100 == 0 and i>0:
		print("---- iteration", i,"--------")
	query = file[i]
	tweet = query['Tweet']
	question = query['Question']
	qid = query['qid']
	ans = model([tweet], [question])
	dev_eval.append({'Answer':ans[0],'qid':qid})


json.dump(dev_eval, open('dev_eval.json', 'w'))