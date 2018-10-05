#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 03:38:32 2018

@author: duytinvo
"""
import re
import torch
import csv
import random
import argparse
import numpy as np
from collections import Counter
from model import Classifier
from utils.other_utils import SaveloadHP

seed_num = 12345
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def process_sent(sent):
    sent = re.sub("[\'\"`]+", '', sent)
    sent = re.sub('[^0-9a-zA-Z ]+', ' ', sent)
    sent = sent.lower()
    return sent.strip().split()


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

        aspect = " ".join(process_sent(aspect))
        asp_rep = aspect.replace(" ", "_")
        sentence = " ".join(process_sent(sentence))
        sent_rep = sentence.replace(aspect, asp_rep)

        label_prob, label_pred = classifier.predict(sent_rep, asp_rep, len(i2l))
        print("\t[SA_PREDICTION] Polarity score of aspect '%s' is %f" % (aspect, 1-label_prob.item()))


if __name__ == '__main__':
    # from utils.data_utils import Vocab, Data2tensor, Csvfile, seqPAD
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")
        
    argparser.add_argument('--model_args', help='Args file', default="./results/sf_distlstm_v3_pp.args", type=str)
    
    args = argparser.parse_args()
        
    interactive_shell(args)
