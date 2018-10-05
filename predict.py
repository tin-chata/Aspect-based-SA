#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 03:38:32 2018

@author: duytinvo
"""
import re
import os
import csv
import torch
import random
import numpy as np
from model import Classifier
from utils.other_utils import SaveloadHP
from process_data import process_sent_ap
from corenlpsf import StanfordNLP
from corenlpsf import extract_nn as extract_np
from collections import defaultdict, Counter

seed_num = 12345
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def read_aspect(readfile="/media/data/hotels/booking_v2/processed/extracted_tag/tag_aspects.csv"):
    aspects = Counter()
    with open(readfile, "r") as f:
        csvreader = csv.reader(f)
        for row in csvreader:
                wd, cnt = row
                if int(cnt) >= 5:
                    aspects[wd] += int(cnt)
    common_aspects = []
    for a in aspects.most_common():
        common_aspects.append(a[0])
    return common_aspects


class ABSA(object):
    def __init__(self, model_args="./results/booking_lstm_v4_ps.args",
                 apfile="/media/data/hotels/booking_v2/processed/extracted_tag/tag_aspects.csv", use_cuda=False):
        margs = SaveloadHP.load(model_args)
        margs.use_cuda = use_cuda

        self.i2l = {}
        for k, v in margs.vocab.l2i.items():
            self.i2l[v] = k

        print("Load Model from file: %s" % margs.model_name)
        self.classifier = Classifier(margs)
        self.classifier.model.load_state_dict(torch.load(margs.model_name))
        self.classifier.model.to(self.classifier.device)
        self.common_aspects = read_aspect(apfile)
        self.sNLP = StanfordNLP(port=8000)

    def predict(self, date, review, rating):
        sa_info = defaultdict(list)
        sentences = self.sNLP.ssplit(review)
        for sentence in sentences:
            sentence = " ".join(sentence)
            print(100 * "=")
            print("[SENTENCE] %s" % sentence)
            print(100 * "=")
            # TODO Improve this NN extraction
            NPs = extract_np(sentence, self.sNLP)
            print("(0)[NP_TRUNCATE] List of noun phrases: \n", NPs)
            for aspect in set(NPs):
                aspect = aspect.lower()
                sentence = sentence.lower()
                if aspect in self.common_aspects:
                    sent_rep, asp_rep = process_sent_ap(sentence, aspect)
                    print(100*"-")
                    label_prob, label_pred = self.classifier.predict(sent_rep, asp_rep, len(self.i2l))
                    print("\t(1)[SA_PREDICTION] Polarity score of aspect '%s' is %f" % (aspect, label_prob.item()))
                    sa_info[aspect].append((date, aspect, label_prob.item(), sentence, rating))

        return sa_info


def write_dict2csv(wfile, sa_info):
    with open(wfile, "w") as f:
        for aps in sa_info:
            csvwriter = csv.writer(f)
            csvwriter.writerows(sa_info[aps])


if __name__ == "__main__":
    review_file = "/Users/duytinvo/Projects/aspectSA/hotel/data/customer_reviews/Hotel_Review-g181808-d7332235-Reviews-Hampton_Inn_Suites_Airdrie-Airdrie_Alberta.html.csv"
    # review_file = "/Users/duytinvo/Projects/aspectSA/hotel/data/customer_reviews/Hotel_Review-g181808-d579389-Reviews-Holiday_Inn_Express_Suites_Airdrie-Airdrie_Alberta.html.csv"
    pdir, bname = os.path.split(review_file)
    bdir = bname.split(".")[0]

    absa = ABSA()
    with open(os.path.join(pdir, "SA_SUMMARY__" + bdir + ".csv"), "w") as g:
        csvwriter = csv.writer(g)
        with open(review_file, "r") as f:
            csvreader = csv.reader(f)
            for line in csvreader:
                dat, rev, rat = line
                print(100 * "*")
                print("[REVIEW] %s"%rev)
                print(100 * "*")
                try:
                    sa_info = absa.predict(dat, rev, rat)
                    for aps in sa_info:
                        csvwriter.writerows(sa_info[aps])
                except:
                    pass