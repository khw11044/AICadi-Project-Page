import torch
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange
import sys
import os  

sys.path.append(os.getcwd())

class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # 추가 
        self.upscale = nn.Linear(34, 1024)   # 1024 -> 1280
        self.res_common = res_block()
        self.res_pose1 = res_block()
        
        self.rnn = nn.LSTM(int(1024*width_mult if width_mult > 1.0 else 1024),  # 1280 -> 34
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        
        if self.bidirectional:
            self.lin = nn.Linear(2*self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))
        else:
            return (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))

    def forward(self, x, lengths=None):
        B, T, J = x.size() 
        self.hidden = self.init_hidden(B)      # LSTM 네트워크에 쓰임, 설정
        
        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))
        x = nn.LeakyReLU()(self.res_common(x))
        
        if self.dropout:
            x = self.drop(x)
        
        r_in = x.view(B, T, -1)    # [768, 1280] -> [12, 64, 1280]
        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(B*T,9)

        return out


class res_block(nn.Module):
    def __init__(self):
        super(res_block, self).__init__()
        self.l1 = nn.Linear(1024, 1024) # 1024 -> 1280
        self.l2 = nn.Linear(1024, 1024)
        #self.bn1 = nn.BatchNorm1d(1024)
        #self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        inp = x
        x = nn.LeakyReLU()(self.l1(x))
        x = nn.LeakyReLU()(self.l2(x))
        x += inp

        return x