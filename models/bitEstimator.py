from .basics import *
import torch.nn as nn
import torch

# import pickle
# import os
# import codecs

class Bitparm(nn.Module):
    '''
    save params
    '''

    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)


class BitEstimator(nn.Module):
    '''
    Estimate bit
    '''

    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)
        # for i in range(3):
        #     # print(x.size(), F.softplus(self.h[i]).size())
        #     x = x * F.softplus(self.h[i]) + self.b[i]
        #     x = x + F.tanh(x) * F.tanh(self.a[i])
        # return F.sigmoid(x * F.softplus(self.h[3]) + self.b[3])
