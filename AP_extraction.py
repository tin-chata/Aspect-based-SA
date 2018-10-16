"""
Created on 2018-10-03
@author: duytinvo
"""
import re
from collections import Counter
from sklearn.externals import joblib
from process_data import write_csv_lines

noisy_npos = [("('NNP', 'NN', '.')", (0, 1)),
              ("('DT', 'NN')", 1),
              ("('NN', '.')", 0),
              ("('DT', 'NN', '.')", 1),
              ("('NNP', '.')", 0),
              ("('NNP', 'NN')", (0, 1)),
              ("('NN', 'NN')", (0, 1)),
              ("('DT', 'NN', 'NN', '.')", (0, 1)),
              ("('DT', 'NNS')", 1),
              ("('DT', 'NNS', '.')", 1),
              ("('DT', 'NN', 'IN', 'DT', 'NN', '.')", (1, 4)),
              ("('NN', 'NN', '.')", (0, 1)),
              ("('NNP', 'CC', 'NN')", [0, 2]),
              ("('NN', 'IN', 'NN', '.')", (0, 2)),
              ("('DT', 'NN', 'NN')", (1, 2)),
              ("('NN', 'CC', 'NN')", [0, 2]),
              ("('DT', 'NN', 'IN', 'NN', '.')", (1, 3)),
              ("('NN', 'IN', 'NN')", (0, 2)),
              ("('RB', 'TO', 'NN', '.')", 2),
              ("('VBD', 'DT', 'NN', '.')", 2),
              ("('RB', 'TO', 'NN')", 2),
              ("('DT', 'NN', 'IN', 'DT', 'NN')", (1, 4)),
              ("('PRP', 'VBD', 'DT', 'NN', '.')", 3)]          # 100 -->

noisy_apos = [("('RB', 'JJ', '.')", 1),
              ("('JJ', '.')", 0),
              ("('RB', 'JJ')", 1),
              ("('JJ', 'CC', 'JJ', '.')", [0, 2]),
              ("('RB', 'JJ', 'CC', 'JJ', '.')", [1, 3]),
              ("('PRP', 'VBD', 'JJ', '.')", 2),
              ("('PRP', 'VBD', 'RB', 'JJ', '.')", 3),
              ("('JJ', 'CC', 'JJ')", [0, 2])]          # 100 -->

noisy_anpos = [("('JJ', 'NN', '.')", (1, 0)),
               ("('JJ', 'NN')", (1, 0)),
               ("('JJ', 'NNS', '.')", (1, 0)),
               ("('JJ', 'NNS')", (1, 0)),
               ("('NNP', 'JJ', '.')", (0, 1)),
               ("('NN', 'JJ', '.')", (0, 1))]          # 100 -->

interest_anpos = [("('NN', 'VBD', 'JJ', '.')", (0, 2)),                 # nothing was ...
                  ("('DT', 'NN', 'VBD', 'JJ', '.')", (1, 3)),
                  ("('DT', 'NN', 'VBD', 'RB', 'JJ', '.')", (1, 4)),
                  ("('NN', 'VBD', 'RB', 'JJ', '.')", (0, 3)),
                  ("('RB', 'JJ', 'NN', '.')", (2, 1)),
                  ("('NN', 'VBD', 'JJ')", (0, 2)),                      # everything/nothing was ...
                  ("('NNP', 'VBD', 'JJ', '.')", (0, 2)),
                  ("('RB', 'JJ', 'NN', '.')", (2, 1)),
                  ("('NNP', 'JJ', 'NN', '.')", (2, 1)),
                  ("('DT', 'NN', 'VBD', 'JJ', 'CC', 'JJ', '.')", (1, [3, 5])),
                  ("('NN', 'VBD', 'JJ', 'CC', 'JJ', '.')", (0, [2, 4])),
                  ("('JJ', 'JJ', 'NN', '.')", (2, [0, 1])),
                  ("('DT', 'NN', 'VBD', 'RB', 'JJ', 'CC', 'JJ', '.')", (1, [4, 6])),
                  ("('JJ', 'NN', 'NN', '.')", ((1, 2), 0)),
                  ("('DT', 'NN', 'VBZ', 'JJ', '.')", (1, 3)),
                  ("('NNS', 'VBD', 'JJ', '.')", (0, 2)),
                  ("('DT', 'NN', 'VBD', 'JJ')", (1, 3)),
                  ("('NN', 'VBD', 'RB', 'JJ', 'CC', 'JJ', '.')", (0, [3, 5])),
                  ("('JJ', 'NN', 'IN', 'NN', '.')", ((1, 3), 0)),       # `good value for money`, Cant think of anything
                  ("('DT', 'NNS', 'VBD', 'JJ', '.')", (1, 3)),
                  ("('JJ', 'NN', 'IN', 'DT', 'NN', '.')", ((1, 4), 0)),
                  ("('JJ', 'CC', 'JJ', 'NN', '.')", (3, [0, 2])),
                  ("('NN', 'VBD', 'RB', 'JJ')", (0, 3)),
                  ("('NNP', 'VBZ', 'JJ', '.')", (0, 2)),
                  ("('DT', 'JJ', 'NN', '.')", (2, 1)),
                  ("('DT', 'NNS', 'VBD', 'RB', 'JJ', '.')", (1, 4)),
                  ("('RB', 'JJ', 'NN')", (2, 1)),
                  ("('NN', 'VBZ', 'JJ', '.')", (0, 2)),                           # everything is good
                  ("('RB', 'JJ', 'CC', 'JJ', 'NN', '.')", (4, [1, 3])),
                  ("('JJ', 'NN', 'CC', 'JJ', 'NN', '.')", ([1, 4], [0, 3])),
                  ("('NNS', 'VBD', 'RB', 'JJ', '.')", (0, 3)),
                  ("('DT', 'NN', 'VBZ', 'RB', 'JJ', '.')", (1, 4)),
                  ("('NNP', 'VBD', 'JJ')", (0, 2)),
                  ("('PRP', 'VBD', 'DT', 'JJ', 'NN', '.')", (4, 3)),            # we left a next day
                  ("('DT', 'NN', 'NN', 'VBD', 'JJ', '.')", ((1, 2), 4)),
                  ("('RB', 'JJ', 'NNS', '.')", (2, 1)),
                  ("('JJ', 'NN', 'CC', 'NN', '.')", ([1, 3], 0)),
                  ("('DT', 'NN', 'VBD', 'RB', 'JJ')", (1, 4)),
                  ("('JJ', 'NN', 'NN')", ((1, 2), 0)),
                  ("('NNP', 'RB', 'JJ', '.')", (0, 2)),
                  ("('NN', 'NN', 'VBD', 'JJ', '.')", ((0, 1), 3)),
                  ("('JJ', 'NN', ',', 'JJ', 'NN', '.')", ([1, 4], [0, 3])),
                  ("('JJ', 'JJ', 'NN')", (2, [0, 1])),
                  ("('NN', 'RB', 'JJ', '.')", (0, 2)),
                  ("('DT', 'JJ', 'NN')", (2, 1)),
                  ("('DT', 'NN', 'IN', 'DT', 'NN', 'VBD', 'JJ', '.')", ((1, 4), 6)),
                  ("('NN', 'VBZ', 'RB', 'JJ', '.')", (0, 3)),
                  ("('NNP', 'VBD', 'RB', 'JJ', '.')", (0, 3)),
                  ("('DT', 'JJ', 'NN', 'VBD', 'JJ', '.')", (2, [1, 4])),
                  ("('JJ', 'NN', ',', 'JJ', 'NN')", ([1, 4], [0, 3])),
                  ("('NNP', 'NN', 'VBD', 'JJ', '.')", ((0, 1), 3)),
                  ("('JJ', ',', 'JJ', 'NN', '.')", (3, [0, 2])),
                  ("('NNS', 'VBD', 'JJ')", (0, 2)),
                  ("('DT', 'NN', 'VBD', 'JJ', 'CC', 'RB', 'JJ', '.')", (1, [3, 6])),
                  ("('JJ', 'NN', 'CC', 'JJ', 'NN')", ([1, 4], [0, 3])),
                  ("('DT', 'NN', 'VBD', 'JJ', 'CC', 'DT', 'NN', 'VBD', 'JJ', '.')", ([1, 6], [3, 8])),
                  ("('DT', 'NN', 'NN', 'VBD', 'RB', 'JJ', '.')", ((1, 2), 5)),              # 100 -->
                  ("('NNS', 'VBP', 'JJ', '.')", (0, 2)),
                  ("('NNP', 'RB', 'JJ', 'CC', 'JJ', '.')", (0, [2, 4])),
                  ("('NNS', 'VBD', 'JJ', 'CC', 'JJ', '.')", (0, [2, 4])),
                  ("('NN', 'VBZ', 'JJ')", (0, 2)),
                  ("('JJ', 'NN', 'NNS', '.')", ((1, 2), 0)),
                  ("('JJ', 'NN', 'VBD', 'JJ', '.')", (1, [0, 3])),
                  ("('NN', 'VBD', 'JJ', 'CC', 'JJ')", (0, [2, 4])),
                  ("('DT', 'NN', 'VBD', 'RB', 'RB', 'JJ', '.')", (1, 5)),
                  ("('DT', 'NNS', 'VBD', 'JJ', 'CC', 'JJ', '.')", (1, [3, 5])),
                  ("('RB', 'DT', 'JJ', 'NN', '.')", (3, 2)),
                  ("('JJ', 'JJ', 'NNS', '.')", (2, [0, 1])),
                  ("('NNS', 'RB', 'JJ', '.')", (0, 2)),
                  ("('JJ', 'NN', 'IN', 'NN')", ((1, 3), 0)),
                  ("('JJ', 'NN', 'CC', 'NN')", ([1, 3], 0)),
                  ("('JJ', 'NN', 'IN', 'DT', 'JJ', 'NN', '.')", ((1, 5), 0)),
                  ("('JJ', 'CC', 'JJ', 'NN')", (3, [0, 2])),
                  ("('JJ', 'NN', 'IN', 'JJ', 'NN', '.')", ((1, 4), 0)),
                  ("('DT', 'NN', 'VBD', 'JJ', 'JJ', '.')", (1, [3, 4])),
                  ("('DT', 'NNS', 'VBP', 'JJ', '.')", (1, 3)),
                  ("('DT', 'NN', 'CC', 'NN', 'VBD', 'JJ', '.')", ([1, 3], 5)),
                  ("('DT', 'NN', 'VBZ', 'JJ', 'CC', 'JJ', '.')", (1, [3, 5])),
                  ("('NN', 'VBD', 'JJ', 'CC', 'RB', 'JJ', '.')", (0, [2, 5])),
                  ("('NNP', 'JJ', 'CC', 'JJ', '.')", (0, [1, 3])),
                  ("('NN', 'NN', 'VBD', 'RB', 'JJ', '.')", ((0, 1), 4)),
                  ("('NN', 'RB', 'JJ')", (0, 2)),
                  ("('DT', 'NNS', 'VBP', 'RB', 'JJ', '.')", (1, 4)),
                  ("('RB', 'JJ', 'NN', 'NN', '.')", ((2, 3), 1)),
                  ("('JJ', 'NN', 'JJ', 'NN')", ([1, 3], [0, 2])),
                  ("('NNP', 'VBZ', 'JJ')", (0, 2)),
                  ("('VBD', 'DT', 'JJ', 'NN', '.')", (3, 2)),
                  ("('DT', 'NN', 'VBZ', 'JJ')", (1, 3)),
                  ("('NNS', 'VBP', 'RB', 'JJ', '.')", (0, 3)),
                  ("('NN', 'VBD', 'JJ', 'JJ', '.')", (0, [2, 3])),
                  ("('PRP$', 'NN', 'VBD', 'JJ', '.')", (1, 3)),
                  ("('DT', 'NNS', 'VBD', 'RB', 'JJ', 'CC', 'JJ', '.')", (1, [4, 6])),
                  ("('DT', 'NN', 'VBD', 'DT', 'JJ', 'NN', '.')", ([1, 5], 4)),
                  ("('NNP', 'RB', 'JJ')", (0, 2)),
                  ("('NN', 'VBD', 'RB', 'RB', 'JJ', '.')", (0, 4)),
                  ("('JJ', 'NN', 'IN', 'DT', 'NN')", ((1, 4), 0)),
                  ("('JJ', 'NN', 'CC', 'NNS', '.')", ([1, 3], 0)),
                  ("('DT', 'NN', 'VBZ', 'RB', 'JJ', 'CC', 'JJ', '.')", (1, [4, 6])),
                  ("('DT', 'JJ', 'NN', 'VBD', 'RB', 'JJ', '.')", (2, [1, 5])),
                  ("('JJ', 'NN', 'IN')", ((1, 2), 0)),
                  ("('JJ', 'NN', 'TO', 'VB', '.')", (1, 0)),
                  ("('NN', 'VBD', 'JJ', 'NN')", (0, 2)),
                  ("('DT', 'NN', 'IN', 'DT', 'NN', 'VBD', 'RB', 'JJ', '.')", ((1, 4), 7)),
                  ("('NN', 'NN', 'VBD', 'JJ')", ((0, 1), 3)),
                  ("('NNP', 'CC', 'JJ', 'NN', '.')", (3, 2)),
                  ("('DT', 'NNS', 'VBD', 'JJ')", (1, 3)),
                  ("('DT', 'JJ', 'NNS', '.')", (2, 1)),
                  ("('DT', 'NN', 'VBD', 'JJ', 'CC', 'JJ')", (1, [3, 5])),
                  ("('DT', 'NN', 'JJ', '.')", (1, 2)),
                  ("('JJ', 'JJ', 'NN', 'NN', '.')", ((2, 3), [0, 1]))]              # 200 -->


def location(tokens, loc):
    d = []
    if type(loc) is tuple:
        tok = [wd for wd in tokens[loc[0]:loc[1]+1] if re.sub(r'[^0-9a-zA-Z ]+', '', wd).isalnum()]
        # tok = [wd for wd in tokens[loc[0]:loc[1] + 1]]
        if len(tok) > 0:
            d.append(" ".join(tok))
    elif type(loc) is list:
        tok = [tokens[l] for l in loc if re.sub(r'[^0-9a-zA-Z ]+', '', tokens[l]).isalnum()]
        # tok = [tokens[l] for l in loc]
        if len(tok) > 0:
            d.extend(tok)
    else:
        if re.sub(r'[^0-9a-zA-Z ]+', '', tokens[loc]).isalnum():
            d.append(tokens[loc])
        # d.append(tokens[loc])
    return d


def extract_jj_nn(interest_anpos, tag_dict):
    noun_dict = Counter()
    adj_dict = Counter()
    for anpos in interest_anpos:
        pos, loc = anpos
        nloc, aloc = loc
        for sent in tag_dict[pos]:
            tokens = sent[0].lower().split()
            noun_dict.update(location(tokens, nloc))
            adj_dict.update(location(tokens, aloc))

    for anpos in noisy_anpos:
        pos, loc = anpos
        nloc, aloc = loc
        for sent in tag_dict[pos]:
            tokens = sent[0].lower().split()
            noun_dict.update(location(tokens, nloc))
            adj_dict.update(location(tokens, aloc))

    for apos in noisy_apos:
        pos, aloc = apos
        for sent in tag_dict[pos]:
            tokens = sent[0].lower().split()
            adj_dict.update(location(tokens, aloc))

    for npos in noisy_npos:
        pos, nloc = npos
        for sent in tag_dict[pos]:
            tokens = sent[0].lower().split()
            noun_dict.update(location(tokens, nloc))

    strip_wds = ["everything", "everythings", "nothing", "nothing everything", "thing", "things", "lot", "day", "all",
                 "others", "anything", "evrything", "hour", "part", "fun", "mess", "else", "bit", "night", "b", "way",
                 "super", "none", "wife", "pretty", "dislike", "complaints", "complaint", "everyone", "time", "joke"]
    # strip_wds = []
    for wd in strip_wds:
        noun_dict.pop(wd)
    write_csv_lines("/media/data/hotels/booking_v2/processed/extracted_tag/tag_aspects2.csv", noun_dict.most_common())
    return noun_dict, adj_dict


if __name__ == "__main__":
    # tag_file = "/media/data/hotels/booking_v2/processed/extracted_tag/tag_count_dict.pkl"
    # tag_count, tag_dict = joblib.load(tag_file)
    # joblib.dump(tag_count, "/media/data/hotels/kdd11/processed/extracted_tag/kdd11_tag_count.pkl")
    # joblib.dump(tag_dict, "/media/data/hotels/kdd11/processed/extracted_tag/kdd11_tag_dict.pkl")
    # tag_file = "/media/data/hotels/kdd11/processed/extracted_tag/kdd11_tag_count_dict.pkl"
    tag_count = joblib.load("/media/data/hotels/kdd11/processed/extracted_tag/kdd11_tag_count.pkl")
    tag_dict = joblib.load("/media/data/hotels/kdd11/processed/extracted_tag/kdd11_tag_dict.pkl")
    # noun_dict, adj_dict = extract_jj_nn(interest_anpos, tag_dict)