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


class ABSA(object):
    def __init__(self, model_args="./results/booking_lstm_v4_ps.args",
                 ap_file="/media/data/hotels/booking_v2/processed/extracted_tag/tag_aspects.csv",
                 use_cuda=False, port=9000, thres=5):
        margs = SaveloadHP.load(model_args)
        margs.use_cuda = use_cuda

        self.i2l = {}
        for k, v in margs.vocab.l2i.items():
            self.i2l[v] = k

        print("Load Model from file: %s" % margs.model_name)
        self.classifier = Classifier(margs)
        self.classifier.model.load_state_dict(torch.load(margs.model_name))
        self.classifier.model.to(self.classifier.device)
        self.common_aspects = self.read_aspect(ap_file, thres)
        self.sNLP = StanfordNLP(port=port)

    def read_aspect(self, readfile="/media/data/hotels/booking_v2/processed/extracted_tag/tag_aspects.csv", thres=5):
        aspects = Counter()
        with open(readfile, "r") as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                wd, cnt = row
                if int(cnt) >= thres:
                    aspects[wd] += int(cnt)
        common_aspects = []
        for a in aspects.most_common():
            common_aspects.append(a[0])
        return common_aspects

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
            NPs = list(set(NPs))
            if len(NPs) != 0:
                print("(0)[NP_TRUNCATE] List of noun phrases: %s\n" % ", ".join(NPs))
            else:
                print("(0)[NP_TRUNCATE] List of noun phrases: NULL\n")
            if len(NPs) != 0:
                for aspect in NPs:
                    aspect = aspect.lower()
                    sentence = sentence.lower()
                    sent_rep, asp_rep = process_sent_ap(sentence, aspect)
                    print(100*"-")
                    label_prob, label_pred = self.classifier.predict(sent_rep, asp_rep, len(self.i2l))
                    if aspect in self.common_aspects:
                        print("\t(1)[SA_PREDICTION] Polarity score of aspect '%s' is %f" % (aspect, label_prob.item()))
                        sa_info[aspect].append((date, aspect, label_prob.item(), sentence, rating))
                    else:
                        # print("\t(1)[SA_PREDICTION] Polarity score of aspect 'UND_%s' is %f" % (
                        # aspect, label_prob.item()))
                        # sa_info["UND"].append((date, aspect, label_prob.item(), sentence, rating))
                        if len(aspect.split()) >= 2:
                            pos = self.sNLP.pos(aspect)
                            s, t = zip(*pos)
                            if "IN" in t:
                                subap =s[:t.index("IN")]
                                if subap in self.common_aspects:
                                    print("\t(1)[SA_PREDICTION] Polarity score of aspect '%s' is %f" %
                                          (subap, label_prob.item()))
                                    sa_info[subap].append((date, aspect, label_prob.item(), sentence, rating))
                                else:
                                    subap = aspect.split()[-1]
                                    if subap in self.common_aspects:
                                        print("\t(1)[SA_PREDICTION] Polarity score of aspect '%s' is %f" %
                                              (subap, label_prob.item()))
                                        sa_info[subap].append((date, aspect, label_prob.item(), sentence, rating))
                                    else:
                                        print("\t(1)[SA_PREDICTION] Polarity score of aspect 'UND_%s' is %f" % (
                                            aspect, label_prob.item()))
                                        sa_info["UND"].append((date, aspect, label_prob.item(), sentence, rating))
                            else:
                                subap = aspect.split()[-1]
                                if subap in self.common_aspects:
                                    print("\t(1)[SA_PREDICTION] Polarity score of aspect '%s' is %f" %
                                          (subap, label_prob.item()))
                                    sa_info[subap].append((date, aspect, label_prob.item(), sentence, rating))
                                else:
                                    print("\t(1)[SA_PREDICTION] Polarity score of aspect 'UND_%s' is %f" % (
                                        aspect, label_prob.item()))
                                    sa_info["UND"].append((date, aspect, label_prob.item(), sentence, rating))

                        else:
                            print("\t(1)[SA_PREDICTION] Polarity score of aspect 'UND_%s' is %f" % (
                                aspect, label_prob.item()))
                            sa_info["UND"].append((date, aspect, label_prob.item(), sentence, rating))
            else:
                aspect = "NULL"
                sentence = sentence.lower()
                sent_rep, asp_rep = process_sent_ap(sentence, aspect)
                print(100 * "-")
                label_prob, label_pred = self.classifier.predict(sent_rep, sent_rep.split()[-1], len(self.i2l))
                print("\t(1)[SA_PREDICTION] Polarity score of aspect '%s' is %f" % (aspect, label_prob.item()))
                sa_info[aspect].append((date, aspect, label_prob.item(), sentence, rating))
        return sa_info


def write_dict2csv(wfile, sa_info):
    with open(wfile, "w") as f:
        for aps in sa_info:
            csvwriter = csv.writer(f)
            csvwriter.writerows(sa_info[aps])


if __name__ == "__main__":
    """
    python predict.py --review_file /Users/duytinvo/Projects/aspectSA/hotel/data/customer_reviews/Hotel_Review-g181808-d7332235-Reviews-Hampton_Inn_Suites_Airdrie-Airdrie_Alberta.html.csv"
    """
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")

    argparser.add_argument('--model_args', help='Args file', default="./results/booking_lstm_v6_ps.args", type=str)

    argparser.add_argument('--ap_file', help='Args file', type=str,
                           default="/media/data/hotels/booking_v2/processed/extracted_tag/tag_aspects.csv")

    argparser.add_argument('--review_file', help='Review file', type=str,
                           default="/Users/duytinvo/Projects/aspectSA/hotel/data/customer_reviews/Hotel_Review-g181808-d7332235-Reviews-Hampton_Inn_Suites_Airdrie-Airdrie_Alberta.html.csv")

    argparser.add_argument('--port', help='port number', default=9000, type=int)

    argparser.add_argument('--thres', help='Threshold aspects', default=5, type=int)

    args = argparser.parse_args()

    # review_file = "/Users/duytinvo/Projects/aspectSA/hotel/data/customer_reviews/Hotel_Review-g181808-d579389-Reviews-Holiday_Inn_Express_Suites_Airdrie-Airdrie_Alberta.html.csv"
    pdir, bname = os.path.split(args.review_file)
    bdir = bname.split(".")[0]

    absa = ABSA(model_args=args.model_args, ap_file=args.ap_file, use_cuda=args.use_cuda, port=args.port, thres=args.thres)
    with open(os.path.join(pdir, "SA_SUMMARY__" + bdir + ".csv"), "w") as g:
        csvwriter = csv.writer(g)
        with open(args.review_file, "r") as f:
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