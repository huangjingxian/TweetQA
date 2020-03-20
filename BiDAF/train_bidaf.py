import torch
import json
import pdb
from torch.utils.data import Dataset, DataLoader
from utils.data import TweetData
from utils.models import BiDAF
import torch.optim as optim
from torch.autograd import Variable

def train(dataloader):
    model = BiDAF(50, 50, 50)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    twe = None
    for i_batch, sample_batched in enumerate(dataloader):
        twe,que,ans = Variable(sample_batched[0],requires_grad=False),Variable(sample_batched[1],requires_grad=False),Variable(sample_batched[2],requires_grad=False)

        model.zero_grad() 
        optimizer.zero_grad() 
        loss,p1,p2 = model(twe, que, ans)
        loss.backward()
        optimizer.step()
        if i_batch%100==0:
            print(i_batch,'---',loss)
            print(torch.argmax(p1),torch.argmax(p2), ans)
        # if i_batch%200==0:
        #     q = dev[6]
        #     dev_twe = torch.tensor(q['Tweet'],dtype=torch.float).resize(1,59,50)
        #     dev_que = torch.tensor(q['Question'],dtype=torch.float).resize(1,23,50)
        #     dev_ans = torch.tensor(q['Answer'][0],dtype=torch.float).resize(1,12,50)
        #     devloss,_ = model(dev_twe, dev_que, dev_ans)
        #     print('dev loss:', devloss)
    # torch.save(model.state_dict(), './param')
if __name__=='__main__':
    tweet_dataset = TweetData('../se_data/emb_se_train.json')
    dataloader = DataLoader(tweet_dataset, batch_size=1,
                        shuffle=False, num_workers=8)
    # dev = json.load(open('/home/Yvette/norm_emb_data/emb_clean_dev.json'))
    train(dataloader)