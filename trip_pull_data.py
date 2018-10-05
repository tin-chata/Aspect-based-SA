"""
Created on 2018-09-07
@author: duytinvo
"""
import os
import json
from process_data import remove_symbols
from corenlpsf import StanfordNLP
sNLP = StanfordNLP()


def tripadvisor_allsents(kdd11_path, wfile="/media/data/hotels/kdd11/extracted_rev/kdd11_reviews.txt"):
    count = 0
    kdd11_basenames = os.listdir(kdd11_path)
    with open(wfile, "w") as g:
        for kdd11_basename in kdd11_basenames:
            filename = os.path.join(kdd11_path, kdd11_basename)
            with open(filename, "r", encoding='utf-8') as f:
                try:
                    d = json.load(f)
                    revs = d["Reviews"]
                    for rev in revs:
                        con = rev.get("Content", "")
                        if con.find("showReview(") != -1:
                            continue
                        if len(con) > 0:
                            sents = [" ".join(c) for c in sNLP.ssplit(con)]
                            for sent in sents:
                                psent = remove_symbols(sent)
                                if len(psent.split()) >= 2 and len(psent) >= 6:
                                    g.write(sent + "\n")
                                    count += 1
                                    if count % 1000 == 0:
                                        print("Write %d lines" % count)
                except:
                    pass


def tripadvisor_lensents(kdd11_path, len_thres, extracted_path,  wfile="/media/data/hotels/kdd11/extracted_rev/kdd11_reviews.txt"):
    count = [0]*(len_thres+1)
    kdd11_basenames = os.listdir(kdd11_path)
    with open(wfile, "w") as h:
        for kdd11_basename in kdd11_basenames:
            filename = os.path.join(kdd11_path, kdd11_basename)
            with open(filename, "r") as f:
                try:
                    d = json.load(f)
                    revs = d["Reviews"]
                    for rev in revs:
                        con = rev.get("Content", "")
                        if con.find("showReview(") != -1:
                            continue
                        if len(con) > 0:
                            sents = [" ".join(c) for c in sNLP.ssplit(con)]
                            for sent in sents:
                                psent = remove_symbols(sent)
                                if len(psent.split()) >= 2 and len(psent) >= 6:
                                    h.write(sent + "\n")
                                    count[-1] += 1
                                    if count[-1] % 1000 == 0:
                                        print("Write %d lines to %s" % (count[-1], wfile))

                            rat = float(rev.get("Ratings", {}).get("Overall", "-1.0"))
                            if rat == 5 or rat == 1:
                                for i in range(len_thres):
                                    if len(sents) == i+1:
                                        if rat == 5:
                                            pos_file = extracted_path + "kdd11_pos_" + str(i+1) + "_sents.txt"
                                            with open(pos_file, "a") as g:
                                                for sent in sents:
                                                    psent = remove_symbols(sent)
                                                    if len(psent.split()) >= 2 and len(psent) >= 6:
                                                        g.write(sent + "\n")
                                                        count[i] += 1
                                                        if count[i] % 1000 == 0:
                                                            print("Write %d lines to %s" % (count[i], pos_file))
                                        else:
                                            neg_file = extracted_path + "kdd11_neg_" + str(i+1) + "_sents.txt"
                                            with open(neg_file, "a") as g:
                                                for sent in sents:
                                                    psent = remove_symbols(sent)
                                                    if len(psent.split()) >= 2 and len(psent) >= 6:
                                                        g.write(sent + "\n")
                                                        count[i] += 1
                                                        if count[i] % 1000 == 0:
                                                            print("Write %d lines to %s" % (count[i], neg_file))
                except:
                    pass


if __name__ == "__main__":
    """
    python trip_extract_dataset.py --sent_thres 6
    """
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--trip_pathdir', help='tripadvisor directory',
                           default="/media/data/hotels/kdd11/raw_data/json/",
                           type=str)

    argparser.add_argument('--extracted_path', help='json tripadvisor dataset',
                           default="/media/data/hotels/kdd11/processed/extracted_sent/",
                           type=str)

    argparser.add_argument('--wfile', help='writen file',
                           default="/media/data/hotels/kdd11/processed/extracted_sent/kdd11_all_sents.txt",
                           type=str)

    argparser.add_argument('--len_thres', help='sentence length threshold', default=6, type=int)

    args = argparser.parse_args()

    tripadvisor_lensents(args.trip_pathdir, len_thres=args.len_thres, extracted_path=args.extracted_path, wfile=args.wfile)

    # tripadvisor_allsents(args.trip_pathdir, wfile="/media/data/hotels/kdd11/processed/extracted_sent/kdd11_all_sents.txt")

