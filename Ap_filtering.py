"""
Created on 2018-10-10
@author: duytinvo
"""
import csv
import re
from corenlpsf import StanfordNLP
from collections import Counter


def check_overlap(aspects, aspect, sent):
    """
        Check if olp_aps not in sent
    """
    for ap in aspects:
        if len(ap) > len(aspect) and re.search(r"\b%s\b" % ap, sent, flags=re.I):
            return False
    return True


def topap(tag_file, n=50):
    aspects = Counter()
    with open(tag_file, "r") as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            wd, cnt = row
            aspects[wd] += int(cnt)

    common_aspects = []
    for a in aspects.most_common():
        common_aspects.append(a[0])
        if len(common_aspects) == n:
            break
    return common_aspects


def mapping_ap(ap_file, label, sent_file, tag_file, n, port):
    c = 0
    sNLP = StanfordNLP(port=port)
    common_aspects = topap(tag_file, n)
    with open(ap_file, "w") as f:
        csvwriter = csv.writer(f)
        with open(sent_file, "r") as g:
            for line in g:
                sent = line.strip()
                wds = []
                pos = []
                for ap in common_aspects:
                    if re.search(r"\b%s\b" % ap, sent, flags=re.I) and 7 <= len(sent.split()) <= 50:
                        if check_overlap(common_aspects, ap, sent):
                            if len(wds) == len(pos) == 0:
                                wds, pos = zip(*sNLP.pos(line.strip()))
                                wds = [wd.lower() for wd in wds]
                            try:
                                if pos[wds.index(ap.split()[0])] in ["NN", "NNS", "NNP", "NNPS"] and \
                                        pos[wds.index(ap.split()[-1])] in ["NN", "NNS", "NNP", "NNPS"]:
                                    ap_rep = re.sub(r" ", r"_", ap)
                                    sent_rep = re.sub(r"\b%s\b" % ap, ap_rep, sent, flags=re.I)
                                    if len(pos) == wds.index(ap.split()[-1]) + 1:
                                        csvwriter.writerow([sent_rep, ap_rep, label])
                                        c += 1
                                        if c % 10000 == 0:
                                            print("Processing %d lines" % c)
                                    else:
                                        if pos[wds.index(ap.split()[-1]) + 1] not in ["NN", "NNS", "NNP", "NNPS"]:
                                            csvwriter.writerow([sent_rep, ap_rep, label])
                                            c += 1
                                            if c % 10000 == 0:
                                                print("Processing %d lines" % c)
                            except:
                                pass


if __name__ == "__main__":
    """
    python Ap_filtering.py --label negative --sent_file /media/data/hotels/booking_v2/processed/extracted_sent/negative.set.txt --ap_file /media/data/hotels/booking_v2/processed/extracted_ap/aspect_negative_v2.csv 
    """
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--tag_file', help='aspect file',
                           default="/media/data/hotels/booking_v2/processed/extracted_tag/tag_aspects2.csv",
                           type=str)
    argparser.add_argument('--sent_file', help='read file',
                           default="/media/data/hotels/booking_v2/processed/extracted_sent/positive.set.txt",
                           type=str)
    argparser.add_argument('--ap_file', help='written file',
                           default="/media/data/hotels/booking_v2/processed/extracted_ap/aspect_positive_v2.csv",
                           type=str)

    argparser.add_argument('--label', help='polarity', default="positive", type=str)

    argparser.add_argument('--n', help='top n aspects', default=100, type=int)

    argparser.add_argument('--port', help='port number', default=8000, type=int)

    args = argparser.parse_args()

    mapping_ap(args.ap_file, args.label, args.sent_file, args.tag_file, args.n, args.port)

