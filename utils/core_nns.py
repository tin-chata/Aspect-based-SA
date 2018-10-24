#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:41:43 2018

@author: dtvo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embs(nn.Module):
    """
    This module take embedding inputs (characters or words) feeding to an RNN layer to extract:
        - all hidden features
        - last hidden features
        - all attentional hidden features
        - last attentional hidden features
    """
    def __init__(self, HPs, attention=False):
        super(Embs, self).__init__()
        [nnmode, size, dim, pre_embs, hidden_dim, dropout, layers, bidirect, zero_padding, attention] = HPs
        self.zero_padding = zero_padding
        rnn_dim = hidden_dim // 2 if bidirect else hidden_dim
            
        self.embeddings = nn.Embedding(size, dim, padding_idx=0)
        if pre_embs is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pre_embs))
        else:
            self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(size, dim)))

        self.drop = nn.Dropout(dropout)

        if nnmode == "rnn":
            self.hidden_layer = nn.RNN(dim, rnn_dim, num_layers=layers, batch_first=True, bidirectional=bidirect)
        elif nnmode == "gru":
            self.hidden_layer = nn.GRU(dim, rnn_dim, num_layers=layers, batch_first=True, bidirectional=bidirect)
        else: 
            self.hidden_layer = nn.LSTM(dim, rnn_dim, num_layers=layers, batch_first=True, bidirectional=bidirect)
        
        self.attention = attention
        if attention:
            self.att_hidden = nn.Linear(hidden_dim, hidden_dim)
            self.att_alpha = nn.Linear(hidden_dim,1, bias=False)
            self.att_norm = nn.Softmax(-1)
            
    def forward(self, inputs, input_lengths):
        return self.get_last_hiddens(inputs, input_lengths)

    def get_last_hiddens(self, inputs, input_lengths):
        """
            input:  
                inputs: tensor(batch_size, seq_length)
                input_lengths: tensor(batch_size,  1)
            output: 
                tensor(batch_size, hidden_dim)
        """
        if self.zero_padding:
            # set zero vector for padding, unk, eot, sot
            self.set_zeros([0,1,2,3])
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        # embs = tensor(batch_size, seq_length,input_dim)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.data.cpu().numpy(), True)
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * 2)
        # hc_n = (h_n,c_n); h_n = tensor(2, batch_size, rnn_dim)
        rnn_out, hc_n = self.hidden_layer(pack_input)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        if self.attention:
            a_hidden = F.relu(self.att_hidden(rnn_out))
            # a_alpha = tensor(batch_size, seq_length, 1)
            a_alpha = F.relu(self.att_alpha(a_hidden))
            # a_alpha = tensor(batch_size, seq_length)
            a_alpha.squeeze_()
            # alpha = tensor(batch_size, seq_length)
            alpha = self.att_norm(a_alpha)
            # att_out = tensor(batch_size, seq_length, input_dim)
            att_out = rnn_out*alpha.view(batch_size,seq_length,1)
            # att_h = tensor(batch_size, input_dim)
            att_h = att_out.sum(1)    
            return att_h
        else:
            # concatenate forward and backward h_n; h_n = tensor(batch_size, rnn_dim*2)
            if type(hc_n) == tuple:
                h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)
            else:
                h_n = torch.cat([hc_n[0,:,:], hc_n[1,:,:]],-1)
            return h_n

    def get_adapt_hiddens(self, inputs, input_lengths=None, asp_locs=None, idx_range=None):
        """
            input:
                inputs: tensor(batch_size, seq_length)
                input_lengths: tensor(batch_size,  1)
            output:
                tensor(batch_size, hidden_dim)
        """
        if self.zero_padding:
            # set zero vector for padding, unk, eot, sot
            self.set_zeros([0,1,2,3])
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        # embs = tensor(batch_size, seq_length,input_dim)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.data.cpu().numpy(), True)
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * 2)
        # hc_n = (h_n,c_n); h_n = tensor(2, batch_size, rnn_dim)
        rnn_out, hc_n = self.hidden_layer(pack_input)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        h_adapt = rnn_out[idx_range, asp_locs, :]
        h_adapt.squeeze_()
        # concatenate forward and backward h_n; h_n = tensor(batch_size, rnn_dim*2)
        # if type(hc_n) == tuple:
        #     h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)
        # else:
        #     h_n = torch.cat([hc_n[0,:,:], hc_n[1,:,:]],-1)
        return h_adapt

    def get_all_hiddens(self, inputs, input_lengths=None):
        """
            input:  
                inputs: tensor(batch_size, seq_length)
                input_lengths: tensor(batch_size,  1)
            output: 
                tensor(batch_size, seq_length, hidden_dim)
        """
        if self.zero_padding:
            # set zero vector for padding, unk, eot, sot
            self.set_zeros([0,1,2,3])
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        # embs = tensor(batch_size, seq_length,input_dim)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * 2)
        # hc_n = (h_n,c_n); h_n = tensor(2, batch_size, rnn_dim)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.hidden_layer(pack_input)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        if self.attention:
            # concatenate forward and backward h_n; h_n = tensor(batch_size, rnn_dim*2)
            if type(hc_n) == tuple:
                h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)
            else:
                h_n = torch.cat([hc_n[0,:,:], hc_n[1,:,:]],-1)
            # a_hidden = tensor(batch_size, seq_length, rnn_dim * 2)
            a_hidden = F.relu(self.att_hidden(rnn_out))
            # a_alpha = tensor(batch_size, seq_length, 1)
            a_alpha = F.relu(self.att_alpha(a_hidden))
            # a_alpha = tensor(batch_size, seq_length)
            a_alpha.squeeze_()
            # alpha = tensor(batch_size, seq_length)
            alpha = self.att_norm(a_alpha)
            # att_out = tensor(batch_size, seq_length, input_dim)
            att_out = h_n.view(batch_size,1,-1)*alpha.view(batch_size,seq_length,1)   
            return att_out
        else:
            return rnn_out
    
    def random_embedding(self, size, dim):
        pre_embs = np.empty([size, dim])
        scale = np.sqrt(3.0 / dim)
        for index in range(size):
            pre_embs[index,:] = np.random.uniform(-scale, scale, [1, dim])
        return pre_embs
        
    def set_zeros(self,idx):
        for i in idx:
            self.embeddings.weight.data[i].fill_(0)

class BiRNN(nn.Module):
    """
    This module take embedding inputs (characters or words) feeding to an RNN layer before adding a softmax function for classification
    """
    def __init__(self, word_HPs=None, num_labels=None):
        super(BiRNN, self).__init__()
        [nnmode,word_size, word_dim, wd_embeddings, word_hidden_dim, word_dropout, word_layers, word_bidirect, zero_padding, word_att] = word_HPs
        self.zero_padding = zero_padding
        self.num_labels = num_labels
        self.rnn = Embs(word_HPs)
        self.dropfinal = nn.Dropout(word_dropout)
        hidden_dim = word_hidden_dim

        if num_labels > 2:
            self.hidden2tag = nn.Linear(hidden_dim, num_labels)
            self.lossF = nn.CrossEntropyLoss()
        else:
            self.hidden2tag = nn.Linear(hidden_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss()            
    
    def forward(self, word_tensor, word_lengths, aspect_tensor, arange_tensor):
        word_h_n = self.rnn.get_adapt_hiddens(word_tensor, word_lengths, aspect_tensor, arange_tensor)
        label_score = self.hidden2tag(word_h_n)
        label_score = self.dropfinal(label_score)
        return label_score

    def NLL_loss(self, label_score, label_tensor):
        if self.num_labels > 2:
            batch_loss = self.lossF(label_score, label_tensor)
        else:
            batch_loss = self.lossF(label_score, label_tensor.float().view(-1,1))
        return batch_loss  

    def inference(self, label_score, k=1):
        if self.num_labels > 2:
            label_prob = F.softmax(label_score, dim=-1)
            label_prob, label_pred = label_prob.data.topk(k)
        else:
            label_prob = torch.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred


if __name__ == "__main__":
    from data_utils import Data2tensor, Vocab, seqPAD, Csvfile

    filename = "/media/data/aspectSA/train_v2.csv"
    vocab = Vocab(wl_th=None, cutoff=1)
    vocab.build([filename], firstline=False)
    word2idx = vocab.wd2idx(vocab.w2i)
    tag2idx = vocab.tag2idx(vocab.l2i)
    train_data = Csvfile(filename, firstline=False, word2idx=word2idx, tag2idx=tag2idx)

    train_iters = Vocab.minibatches(train_data, batch_size=4)
    data = []
    asp_locs = []
    label_ids = []
    for words, asps, labels in train_iters:
        data.append(words)
        asp_locs.append(asps)
        label_ids.append(labels)
        word_ids, sequence_lengths = seqPAD.pad_sequences(words, pad_tok=0, wthres=1024)

    w_tensor = Data2tensor.idx2tensor(word_ids)
    asp_tensor = Data2tensor.idx2tensor(asps)
    idx_arrange = Data2tensor.idx2tensor(list(range(w_tensor.size(0))))
    # w_tensor[idx_arrange, asp_tensor]
    y_tensor = Data2tensor.idx2tensor(labels)

    data_tensors = Data2tensor.sort_tensors(labels, word_ids, sequence_lengths)
    label_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors