import torch
import json
import pdb
from torch.utils.data import Dataset, DataLoader
from utils.data import TweetData
from utils.models import BiDAF
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import numpy as np


def mse(a,b):
    a = a.data.numpy()
    b = np.array(b)
    mse = mean_squared_error(a,b)
    return mse

model = BiDAF(50, 50, 50)
model.load_state_dict(torch.load('./param'))

dev = json.load(open('/home/Yvette/norm_emb_data/emb_clean_dev.json'))
embs = json.load(open('/home/Yvette/norm_emb_data/emb_dic.json'))
q = dev[12]
print(q['qid'])
dev_twe = torch.tensor(q['Tweet'],dtype=torch.float).resize(1,59,50)
dev_que = torch.tensor(q['Question'],dtype=torch.float).resize(1,23,50)
dev_ans = torch.tensor(q['Answer'][0],dtype=torch.float).resize(1,12,50)
_,output = model(dev_twe, dev_que, dev_ans) 
output = output.resize(12,50)
ans = []
for i in range(12):
    loss = 1000000
    rword = ""
    ot = output[i,:]
    for word in embs:
        msel = mse(ot,embs[word])
        if msel <loss:
            loss = msel
            rword = word
    ans.append(rword)
print(ans)
# pdb.set_trace()