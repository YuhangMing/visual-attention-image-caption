
# coding: utf-8

import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import pickle

class mRNN(nn.Module):
    '''
    Args:
        nvocab (int): size of the vocabulary
        nemb (int): dimension of word embedding
        nfeat ([int, int]): dimension of the features
        nhid (int): dimension of the hidden layers
        att (string): type of attention, soft or hard
    Example Usage:
        model = mRNN()
        model.init_hidden(feature)
        output = model.forward(feature, caption)
        word = model.predict(feature, pre_word, idx2word)
    '''
    def __init__(self, nvocab=10000, nemb=512, nfeat=[512, 196], nhid=1024, att='soft'):
#     def __init__(self, nvocab=10000, nemb=512, nfeat=[512, 196], nhid=1024, nlayers=1, batch=1):
        super().__init__()
        self.nvocab=nvocab
        self.D = nfeat[0]
        self.L = nfeat[1]
        self.M = nemb
        self.N = nhid
        self.att = att
        self.encoder = nn.Embedding(nvocab, nemb)
        # lstmcell takes in input of shape (batch, input_size), 
        # and (h_0, c_0) of shape (batch, hidden_size)
        # outputs h_1, c_1 of shape (batch, hidden_size)
        self.lstm = nn.LSTMCell(nemb+self.D, nhid)
        # The linear layer that maps from hidden state space to word space
        self.Lo = nn.Linear(nemb, nvocab)
        self.Lh = nn.Linear(nhid, nemb)
        self.Lz = nn.Linear(self.D, nemb)
#         self.nlayers = nlayers
#         self.bsz = batch
        # initialize hidden states
        self.init_c = nn.Linear(self.D, nhid)
        self.init_h = nn.Linear(self.D, nhid)
        # attention parameters/weights
        self.Vatt = nn.Linear(self.D, 1)
        self.Watt1 = nn.Linear(self.D, self.D)
        self.Watt2 = nn.Linear(self.N, self.D)

    def init_hidden(self, features, bsz=1):
        # Initial memory state and hidden state are obtained by an average of the 
        # annotation vectors fed through 2 separate MLPs
        avg_a = torch.mean(torch.t(features), 0)
        # CHECK HERE WHEN BSZ IS LARGER THAN 1 !!!!!!!!!!!!!!!!!!!!
        c_init = self.init_c(avg_a)
        h_init = self.init_h(avg_a)
        self.hidden = (h_init.view(bsz, -1), c_init.view(bsz, -1))
        
    def attention(self, features):
        # 512x196
        lol_tanh = torch.nn.Tanh()
        input1 = self.Watt1(torch.t(features))
#         print(input1.size())
        input2 = self.Watt2(self.hidden[0])
#         print(input2.size())
        alpha_interim = lol_tanh(input1 + input2.expand(self.L, self.D))
#         print(alpha_interim.size())
        # 196x512 * 512*512 + 196x1 -> 196*512
        lol_softmax = torch.nn.Softmax()
        alpha = lol_softmax(torch.t(self.Vatt(alpha_interim)))
        # (196x512 * 512x1)^T -> 1x196
        # pick one feature out of the 196 based on distribution alpha
        # alpha should be a list of probabilities
        prob = alpha.data.cpu().numpy()[0]
        if self.att == 'soft':
            ## Soft attention
            weighted = features * alpha.expand(self.D, self.L)
            self.feat_idx = prob
            context = torch.sum(weighted, 1)
        elif self.att == 'hard':
            ## Hard attention
            self.feat_idx = np.random.choice(np.arange(self.L), p=prob)
            context = features[:, self.feat_idx]
        else:
            raise('Unsupport attention model')
        # 512x196 -> 512
        # return 1x512
        return context.unsqueeze(0)
    
    def forward(self, features, captions):
        # Input: LongTensor (1, batch_size)
        # Output: (1, batch_size, nemb)
        # We start with sending in a word at a time, context is calculated from attention
        context = self.attention(features)
        # 1x512
        embeds = self.encoder(captions)
        # 1x512
        embeds_ = torch.cat((embeds,context), 1)
        # 1x1024
#         packed = pack_padded_sequence(embeddings, lengths, batch_first=True)  # For batch learning
        h_t, c_t = self.lstm(embeds_, self.hidden)
        self.hidden = (h_t, c_t)
        # 1x1024
        output = self.Lo(embeds + self.Lh(h_t) + self.Lz(context))
        # 1x2978
        return output

    def predict(self, features, pre_word, idx2word, T=1):
        context = self.attention(features)
        embeds = self.encoder(pre_word)
        embeds_ = torch.cat((embeds,context), 1)
        h_t, c_t = self.lstm(embeds_, self.hidden)
        self.hidden = (h_t, c_t)
        output = self.Lo(embeds + self.Lh(h_t) + self.Lz(context))
        # convert to word
        pred = output.data.cpu().numpy()[-1, :]
        pred = pred - np.max(pred)   # prevent explosion
        smT = np.exp(pred/T) / np.sum(np.exp(pred/T))
        # random sample from the predicted distribution
        ind = np.random.choice(len(smT), p=smT)
        char = idx2word[ind]
        return char

if __name__ == '__main__':
    # load features and vocabulary
    with open('features.pickle', 'rb') as fp:
        features = pickle.load(fp)
    print(features['train'].shape, features['hold'].shape, features['test'].shape)
    with open('captions.pickle', 'rb') as fp:
        captions = pickle.load(fp)
        ##### CHECK LATER, SOME CAPTIONS ARE MISSING
    print(len(captions['train']), len(captions['hold']), len(captions['test']))
    word2idx = np.load('vocab.npy').item()
    print(len(word2idx))
    word2idx['<start>'] = 0
    word2idx['.'] = len(word2idx)
    word2idx['<NULL>'] = len(word2idx)
    idx2word = {y:x for x,y in word2idx.items()}
    print(len(word2idx), len(idx2word))

    # test Train
    model = mRNN(nvocab=len(word2idx)).cuda()
    print(model)
    visual_info = Variable(torch.FloatTensor(features['train'][7,:,:]).cuda())
    cap = random.choice(captions['train'][idx])
    model.init_hidden(visual_info)
    words = cap.split()

    last_word = Variable(torch.LongTensor([word2idx['<start>']]).cuda())
    output = model.forward(visual_info, last_word)
    if words[0].lower() in word2idx.keys():
        targ = Variable(torch.LongTensor([word2idx[words[0].lower()]]).cuda())
    else:
        targ = Variable(torch.LongTensor([word2idx['<NULL>']]).cuda())
    all_out = output
    all_targ = targ

    for iw in range(1, len(words)):
        output = model.forward(visual_info, targ)
        if words[iw].lower() in word2idx.keys():
            targ = Variable(torch.LongTensor([word2idx[words[iw].lower()]]).cuda())
        else:
            targ = Variable(torch.LongTensor([word2idx['<NULL>']]).cuda())
        all_out = torch.cat((all_out, output), 0)
        all_targ = torch.cat((all_targ, targ), 0)
    print(all_out.data.cpu().numpy()[0])

