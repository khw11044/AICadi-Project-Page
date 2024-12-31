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
        
        self.rnn = nn.LSTM(int(51*width_mult if width_mult > 1.0 else 51),  # 1280 -> 34
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
        # batch_size, timesteps, C, H, W = x.size()       # 12, 64, 3,160,160
        B, T, J, C = x.size() 
        x = rearrange(x, 'b t j c -> b t (j c)',j=J).contiguous() # 12*64, 17*2

        self.hidden = self.init_hidden(B)      # LSTM 네트워크에 쓰임, 설정

        r_out, states = self.rnn(x, self.hidden)
        out = self.lin(r_out)
        out = out.view(B*T,9)

        return out


