import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
torch.manual_seed(0)
class Attention(nn.Module):
    def __init__(self, in_twe_dim, in_que_dim, hidden_dim):
        super(Attention,self).__init__()
        self.hidden_dim = hidden_dim
        self.Ws1 = self.generate_Ws1(65,hidden_dim)
        self.Ws2 = self.generate_Ws2(20,hidden_dim)
        self.Ws3 = self.generate_Ws3(65,hidden_dim)
        self.softmax1 = torch.nn.Softmax(dim=2)
        self.softmax2 = torch.nn.Softmax(dim=1)

    def generate_Ws1(self, len_twe, hidden_dim):
        ws1 = Variable(nn.init.normal_(torch.empty(1,2*hidden_dim)).repeat(len_twe,1))
        return ws1

    def generate_Ws2(self, len_que, hidden_dim):
        ws2 = Variable(nn.init.normal_(torch.empty(1,2*hidden_dim)).repeat(len_que,1))
        return ws2

    def generate_Ws3(self, len_twe, hidden_dim):
        ws3 = Variable(nn.init.normal_(torch.empty(1,2*hidden_dim)).repeat(len_twe,1))
        return ws3

    def forward(self, H, U):
        H1 = torch.sum(H*self.Ws1,dim=2).repeat(20,1,1).permute(1,2,0)
        U1 = torch.sum(U*self.Ws2,dim=2).repeat(65,1,1).permute(1,0,2)
        HU = torch.bmm(H*self.Ws3, U.permute(0,2,1))
        S = H1+U1+HU
        at = self.softmax1(S)
        Util = torch.bmm(at, U)
        beta,_ = torch.max(S,2)
        b = self.softmax2(beta).repeat(100,1,1).permute(1,2,0)
        Htil = torch.sum(b*H,dim=1).repeat(65,1,1).permute(1,0,2)
        G = torch.cat((H,Util,H*Util,H*Htil),dim=2)
        return G


class BiDAF(nn.Module):
    def __init__(self, in_twe_dim, in_que_dim, hidden_dim):
        super(BiDAF,self).__init__()
        self.drop = nn.Dropout(0)
        self.drop2 = nn.Dropout(0)
        # The first bi-lstm layer -- in_twe_dim = in_que_dim = 50
        self.lstm1twe = nn.LSTM(in_twe_dim, hidden_dim, bidirectional=True)
        self.lstm1que = nn.LSTM(in_que_dim, hidden_dim, bidirectional=True) 
        # attention layer
        self.attention = Attention(in_twe_dim, in_que_dim, hidden_dim)
        # the last bi-lstm layer
        self.lstm = nn.LSTM(hidden_dim*8,hidden_dim, bidirectional=True, num_layers = 2)
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, bidirectional=True)
        # two linear layer
        self.linear1 = nn.Linear(10*hidden_dim,1,bias=True)
        self.linear2 = nn.Linear(10*hidden_dim,1,bias=True)
        
        self.softmaxp1 = nn.Softmax(dim=1)
        self.softmaxp2 = nn.Softmax(dim=1)
        # loss function
        self.ce = nn.CrossEntropyLoss()


    def forward(self, twe, que, ans):

        H,_ = self.lstm1twe(twe)
        U,_ = self.lstm1que(que)
        
        # # Attention layer
        G = self.attention(H,U)
        M,_= self.lstm(G)
        M2,_ = self.lstm2(M)
        GcomM = torch.cat((G,M),dim=2)
        GcomM2 = torch.cat((G,M2),dim=2)
        p1 = self.softmaxp1(self.linear1(GcomM).reshape(GcomM.shape[0],65))
        p2 = self.softmaxp2(self.linear2(GcomM2).reshape(GcomM2.shape[0],65))
        a1= ans[:,0]
        a2 = ans[:,1]
        
        loss = self.ce(p1,a1)+self.ce(p2,a2)
        # pdb.set_trace()
        return loss,p1,p2

