#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 03:38:32 2018

@author: duytinvo
"""
import torch
import random
import argparse
import numpy as np
from model import Classifier
from process_data import process_sent_ap
from utils.other_utils import SaveloadHP

seed_num = 12345
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def predict_null(classifier, sent, asp, i2l):
    from utils.data_utils import Data2tensor, seqPAD
    wl = classifier.args.vocab.wl
    ## set model in eval model
    classifier.model.eval()
    fake_label = [0]
    words, asp_loc = classifier.word2idx(sent, asp)
    word_ids, sequence_lengths = seqPAD.pad_sequences([words], pad_tok=0, wthres=wl)
    data_tensors = Data2tensor.sort_tensors(fake_label, [asp_loc], word_ids, sequence_lengths, classifier.device)
    fake_label_tensor, aspect_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors
    arange_tensor = Data2tensor.idx2tensor(list(range(word_tensor.size(0))), classifier.device)
    word_h_n = classifier.model.rnn.get_all_hiddens(word_tensor, sequence_lengths).mean(1)
    label_score = classifier.model.hidden2tag(word_h_n)
    label_score = classifier.model.dropfinal(label_score)
    label_prob, label_pred = classifier.model.inference(label_score, len(i2l))
    return label_prob, label_pred


def interactive_shell(args):
    """Creates interactive shell to play with model

    Args:
        model: instance of Classification

    """
    margs = SaveloadHP.load(args.model_args)
    margs.use_cuda = args.use_cuda
    
    i2l = {}
    for k, v in margs.vocab.l2i.items():
        i2l[v] = k
        
#    device = torch.device("cuda:0" if args.use_cuda else "cpu")        
    print("Load Model from file: %s" % (margs.model_name))
    classifier = Classifier(margs)
    classifier.model.load_state_dict(torch.load(margs.model_name))
    classifier.model.to(classifier.device)
        
    print("""
To exit, enter 'EXIT'.
Enter a sentence like 
input> wth is it????""")

    while True:
        try:
            # for python 2
            sentence = raw_input("review-sentence> ")
            aspect = raw_input("aspect> ")

        except NameError:
            # for python 3
            sentence = input("review-sentence> ")
            aspect = input("aspect> ")

        if sentence == "EXIT":
            break

        sent_rep, asp_rep = process_sent_ap(sentence, aspect)
        if aspect == "NULL":
            label_prob, label_pred = predict_null(classifier, sent_rep, asp_rep, i2l)
        else:
            label_prob, label_pred = classifier.predict(sent_rep, asp_rep, len(i2l))
        print("\t[SA_PREDICTION] Polarity score of '%s' is %f" % (aspect, label_prob.item()))


if __name__ == '__main__':
    # from utils.data_utils import Vocab, Data2tensor, Csvfile, seqPAD
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")
        
    argparser.add_argument('--model_args', help='Args file', default="./results/booking_lstm_v6_ps.args", type=str)
    
    args = argparser.parse_args()
        
    interactive_shell(args)
