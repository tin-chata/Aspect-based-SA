"""
Created on 2018-09-26
@author: duytinvo
"""
import re
import csv
from sklearn.externals import joblib
from collections import Counter
from nltk import pos_tag


def write_csv(filename, data):
    with open(filename, "w") as f:
        csvwriter = csv.writer(f)
        for line in data:
            csvwriter.writerow(line)


def write_txt(filename, data):
    data.sort(key=lambda x: len(x), reverse=False)
    with open(filename, "w") as f:
        for line in data:
            f.write(line + "\n")


def sort_data(readfile, sortedfile):
    """
    filter all short sentences that have a raw version being less than 1 words
    and a processed version being less than 5 characters (e.g. notv)
    """
    data = []
    c = 0
    with open(readfile, "r") as f:
        for line in f:
            sent = line.strip()
            psent = re.sub("[\'\"`]+", '', sent)
            psent = re.sub(r'[^0-9a-zA-Z ]+', ' ', psent)
            psent = " ".join(psent.split())
            if len(psent.split()) >= 2 and len(psent) >= 6:
                data.append(sent)
                c += 1
                if c % 10000 == 0:
                    print("Processing %d sentences" % c)
    data = list(set(data))
    data.sort(key=lambda x: len(x), reverse=False)
    write_txt(sortedfile, data)


def extract_an(pos):
    bigrams = []
    if len(pos) >= 2:
        biwds = zip(pos[:-1],pos[1:])
        biwds = [list(zip(*biwd)) for biwd in biwds]
        bigrams = [" ".join(biwd[0]) for biwd in biwds if biwd[1][0] in ("JJ", "JJR", "JJS") and biwd[1][1] in ["NN", "NNS"]]
    trigrams = []
    if len(pos) >= 3:
        triwds = zip(pos[:-2], pos[1:-1], pos[2:])
        triwds = [list(zip(*triwd)) for triwd in triwds]
        trigrams = [" ".join(triwd[0]) for triwd in triwds if
                    triwd[1][0] in ("JJ", "JJR", "JJS") and triwd[1][1] in ["NN", "NNS"] and triwd[1][2] in ["NN", "NNS"]]
    ans = bigrams
    for an in ans:
        if an in " ".join(trigrams):
            ans.remove(an)
    ans += trigrams
    frgrams = []
    if len(pos) >= 4:
        frwds = zip(pos[:-3], pos[1:-2], pos[2:-1], pos[3:])
        frwds = [list(zip(*frwd)) for frwd in frwds]
        frgrams = [" ".join(frwd[0]) for frwd in frwds if
                   frwd[1][0] in ("JJ", "JJR", "JJS") and frwd[1][1] in ["NN", "NNS"]
                   and frwd[1][2] in ["NN", "NNS"] and frwd[1][3] in ["NN", "NNS"]]
    for an in ans:
        if an in " ".join(frgrams):
            ans.remove(an)
    ans += frgrams
    return ans


def count_an(filename="/media/data/booking.com/adj_nouns.csv"):
    count_nps = Counter()
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            wd, cnt = row
            count_nps[wd] += int(cnt)

    pop_nps = []
    for wd in count_nps:
        if count_nps[wd] <= 3:
            pop_nps.append(wd)

    for wd in pop_nps:
        count_nps.pop(wd)

    count_nouns = Counter()
    count_adjs = Counter()
    for np, count in count_nps.most_common():
        adj = np.split()[0]
        nns = [w for w in np.split()[1:] if w.isalnum()]
        if len(nns) != 0:
            pos = pos_tag(nns)
            w, t = zip(*pos)
            if "JJ" not in t and "JJR" not in t and "JJS" not in t:
                noun = " ".join(nns)
                count_nouns[noun] += count
        if adj.isalnum():
            count_adjs[adj] += count
    write_csv("/media/data/booking.com/nouns.csv", count_nouns.most_common())
    joblib.dump(count_nouns, "/media/data/booking.com/nouns.pkl")
    write_csv("/media/data/booking.com/adjs.csv", count_adjs.most_common())
    intersection = set.intersection(set(count_nouns), set([wd[0] for wd in count_adjs.most_common(180)]))
    for wd in intersection:
        count_nouns.pop(wd)
    write_csv("/media/data/booking.com/aspects.csv", count_nouns.most_common())
    joblib.dump(count_nouns, "/media/data/booking.com/aspects.pkl")


def read_data(filename):
    c = 0
    sentences = []
    with open(filename, "r") as f:
        for line in f:
            sent = line.strip()
            psent = re.sub("[\'\"`]+", '', sent)
            psent = re.sub(r'[^0-9a-zA-Z ]+', ' ', psent)
            psent = " ".join(psent.split())
            if len(psent.split()) >= 2 and len(psent) >= 6:
                sentences.append(sent)
                c += 1
                if c % 100000 == 0:
                    print("Processing %d sentences" % c)
    return set(sentences)


def clean_booking(pos_file="/media/data/hotels/booking_v2/processed/extracted_sent/booking.positive.sorted.tkn.txt",
                  neg_file="/media/data/hotels/booking_v2/processed/extracted_sent/booking.negative.sorted.tkn.txt"):
    pos_data = read_data(pos_file)
    neg_data = read_data(neg_file)
    inter_sents = pos_data.intersection(neg_data)
    pos_data.difference_update(inter_sents)
    neg_data.difference_update(inter_sents)
    write_txt("/media/data/hotels/booking_v2/processed/extracted_sent/positive.set.txt", list(pos_data))
    write_txt("/media/data/hotels/booking_v2/processed/extracted_sent/negative.set.txt", list(neg_data))


def clean_tripadvisor(pos_file="/media/data/hotels/kdd11/processed/extracted_sent/kdd11_pos_reviews_v2.txt",
                      neg_file="/media/data/hotels/kdd11/processed/extracted_sent/kdd11_neg_reviews_v2.txt"):
    pos_data, pos_words = read_data(pos_file)
    neg_data, neg_words = read_data(neg_file)

    pos_data = set(pos_data)
    neg_data = set(neg_data)
    inter_sents = pos_data.intersection(neg_data)
    pos_data.difference_update(inter_sents)
    neg_data.difference_update(inter_sents)

    write_txt("/media/data/hotel/hongmin_wang/kdd11_positive.set_v2.txt", list(pos_data))
    write_txt("/media/data/hotel/hongmin_wang/kdd11_negative.set_v2.txt", list(neg_data))

    nps = pos_words + neg_words
    count_nps = Counter([np.lower() for np in nps])
    write_csv("/media/data/hotel/hongmin_wang/kdd11_adj_nouns_v2.csv", count_nps.most_common())


if __name__ == "__main__":
    # clean_data()
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--read_file', help='Input file',
                           default="/media/data/booking.com/booking_bc_positive.txt",
                           type=str)
    argparser.add_argument('--write_file', help='Output file',
                           default="/media/data/booking.com/booking_bc_positive.sorted.txt",
                           type=str)

    args = argparser.parse_args()

    # clean_tripadvisor_data()
    count_an(filename="/media/data/booking.com/adj_nouns.csv")


