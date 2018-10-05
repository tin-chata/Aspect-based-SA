"""
Created on 2018-08-23
@author: duytinvo
"""
import csv
import re
from corenlpsf import StanfordNLP, extract_allnp


def process_sent(sent):
    sent = re.sub("[\'\"`]+", '', sent)
    sent = re.sub('[^0-9a-zA-Z ]+', ' ', sent)
    sent = sent.lower()
    return sent.split()


def check_contrast(sent):
    """
        Check if contrasting words not in sent
    """
    contrasting = ["even though", "though", "although", "but", "despite", "in spite of", "even if", "if", "however", "the only",
                   "apart from", "except", "nevertheless", "unless"]
    for wd in contrasting:
        if re.search(r"\b%s\b" % wd, sent):
            return False, wd
    return True, wd


def overlap_aspect(ap, aspects):
    """
    Return a list of olp_aps from aspects that are partial mapping with ap
    """
    olp_aps = [aspect for aspect in aspects if ap != aspect and re.search(r"\b%s\b" % ap, aspect)]
    return olp_aps


def check_overlap(olp_aps, sent):
    """
        Check if olp_aps not in sent
    """
    for ap in olp_aps:
        if re.search(r"\b%s\b" % ap, sent):
            return False
    return True


def extract_data(aspects, label, read_file, write_file, cwrite_file, sNLP):
    c1 = 0
    c2 = 0
    pos_lex = ["is good", "is nice", "is great", "are good", "are nice", "are great",
               "was good", "was nice", "was great", "were good", "were nice", "were great",
               "is awesome", "is fantastic", "is excellent", "are awesome", "are fantastic", "are excellent",
               "was awesome", "was fantastic", "was excellent", "were awesome", "were fantastic", "were excellent"]
    with open(write_file, "w", newline='') as f:
        with open(cwrite_file, "w", newline='') as g:
            writer = csv.writer(f, delimiter=",")
            cwriter = csv.writer(g, delimiter=",")
            with open(read_file, "r") as h:
                for line in h:
                    raw_sent = line.strip()
                    sent = process_sent(raw_sent)
                    if 3 <= len(sent) <= 50:
                        sent = " ".join(sent)
                        cont, wd = check_contrast(sent)
                        if cont:
                            for aspect in aspects:
                                if re.search(r"\b%s\b" % aspect, sent):
                                    olp_aps = overlap_aspect(aspect, aspects)
                                    if check_overlap(olp_aps, sent):
                                        nps = extract_allnp(raw_sent, sNLP)
                                        nps = [" ".join(process_sent(np)) for np in nps]
                                        if aspect in nps:
                                            asp_rep = re.sub(r" ", r"_", aspect)
                                            sent_rep = re.sub(r"\b%s\b" % aspect, asp_rep, sent)
                                            if label == "positive" and re.search(r"\bis bad\b", sent):
                                                continue
                                            flag = False
                                            for lex in pos_lex:
                                                if label == "negative" and (re.search(r"\b%s\b" % lex, sent)):
                                                    flag = True
                                                    continue
                                            if flag:
                                                continue
                                            writer.writerow([sent_rep, asp_rep, label])
                                            c1 += 1
                                            if c1 % 1000 == 0:
                                                print("Write %d lines" % c1)
                        else:
                            cwriter.writerow([sent, wd, label])
                            c2 += 1
                            if c2 % 1000 == 0:
                                print("Write %d contrasting lines" % c2)


if __name__ == "__main__":
    """
    python booking_AP_mapping.py --label positive --topap 50 --read_file /media/data/hotels/booking_v2/processed/extracted_sent/positive.set.txt --write_file /media/data/hotels/booking_v2/processed/extracted_ap/aspect_positive.csv --cwrite_file /media/data/hotels/booking_v2/processed/extracted_ap/aspect_positive.cont.csv
    """
    import argparse
    from collections import Counter
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--aspect_file', help='aspect file',
                           default="/media/data/hotels/booking_v2/processed/extracted_tag/tag_aspects.csv",
                           type=str)
    argparser.add_argument('--read_file', help='read file',
                           default="/media/data/hotels/booking_v2/processed/extracted_sent/positive.set.txt",
                           type=str)
    argparser.add_argument('--write_file', help='written file',
                           default="/media/data/hotels/booking_v2/processed/extracted_ap/aspect_positive.csv",
                           type=str)
    argparser.add_argument('--cwrite_file', help='written file',
                           default="/media/data/hotels/booking_v2/processed/extracted_ap/aspect_positive.cont.csv",
                           type=str)
    argparser.add_argument('--label', help='polarity', default="positive", type=str)

    argparser.add_argument('--topap', help='top n aspects', default=-1, type=int)

    argparser.add_argument('--port', help='top n aspects', default=9000, type=int)

    args = argparser.parse_args()

    aspects = Counter()
    with open(args.aspect_file, "r") as f:
        csvreader = csv.reader(f)
        for row in csvreader:
                wd, cnt = row
                aspects[wd] += int(cnt)

    common_aspects = []
    for a in aspects.most_common():
        common_aspects.append(a[0])
        if len(common_aspects) >= args.topap > 0:
            break

    sNLP = StanfordNLP(port=args.port)
    extract_data(common_aspects, args.label, args.read_file, args.write_file, args.cwrite_file, sNLP)
