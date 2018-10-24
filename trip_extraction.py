"""
Created on 2018-10-03
@author: duytinvo
"""
import re
from collections import Counter
from sklearn.externals import joblib
from process_data import write_csv_lines

noisy_npos = [("('NNP', 'NN', '.')", 1),                        # 15
              ("('DT', 'NN')", 1),
              ("('NN', '.')", 0),
              ("('DT', 'NN', '.')", 1),                         # 24
              ("('NNP', 'NN')", 1),
              ("('NN', 'NN')", (0, 1)),
              ("('DT', 'NN', 'NN', '.')", (0, 1)),              # 124
              ("('DT', 'NNS')", 1),
              ("('DT', 'NNS', '.')", 1),                        # 57
              ("('DT', 'NN', 'IN', 'DT', 'NN', '.')", (1, 4)),
              ("('NN', 'NN', '.')", (0, 1)),                    # 72
              ("('PRP', 'VBP', 'DT', 'NN', '.')", 3),           # 73
              ("('NNP', 'CC', 'NN')", 2),
              ("('NN', 'IN', 'NN', '.')", (0, 2)),
              ("('DT', 'NN', 'NN')", (1, 2)),
              ("('NN', 'CC', 'NN')", [0, 2]),
              ("('DT', 'NN', 'IN', 'NN', '.')", (1, 3)),
              ("('NN', 'IN', 'NN')", (0, 2)),
              ("('RB', 'TO', 'NN', '.')", 2),
              ("('VBD', 'DT', 'NN', '.')", 2),
              ("('RB', 'TO', 'NN')", 2),
              ("('DT', 'NN', 'IN', 'DT', 'NN')", (1, 4)),
              ("('PRP', 'MD', 'RB', 'VB', 'DT', 'NN', '.')", 5),        # 33
              ("('WP', 'DT', 'NN', '.')", 2),                           # 42
              ("('PRP', 'RB', 'VB', 'DT', 'NN', '.')", 4),              # 62
              ("('PRP', 'RB', 'VBD', 'DT', 'NN', '.')", 4),             # 65
              ("('PRP', 'VBD', 'DT', 'NN', '.')", 3)]                   # 29       # 100 -->

noisy_apos = [("('RB', 'JJ', '.')", 1),                                 # 5
              ("('JJ', '.')", 0),
              ("('RB', 'JJ')", 1),
              ("('JJ', 'CC', 'JJ', '.')", [0, 2]),                      # 69
              ("('RB', 'JJ', 'CC', 'JJ', '.')", [1, 3]),                # 54
              ("('PRP', 'VBD', 'JJ', '.')", 2),                         # 9
              ("('PRP', 'VBD', 'RB', 'JJ', '.')", 3),                   # 10
              ("('PRP', 'VBZ', 'JJ', '.')", 2),                         # 68
              ("('NNP', 'JJ', '.')", 1),                                # 76
              ("('PRP', 'VBZ', 'RB', 'JJ', '.')", 3),                   # 77
              ("('RB', 'RB', 'JJ', '.')", 2),                           # 84
              ("('DT', 'VBD', 'JJ', '.')", 2),                          # 85
              ("('PRP', 'VBD', 'RB', 'JJ', 'CC', 'JJ', '.')", [3, 5]),  # 87
              ("('PRP', 'VBD', 'JJ', 'CC', 'JJ', '.')", [2, 4]),        # 94
              ("('JJ', 'CC', 'JJ')", [0, 2])]                           # 100 -->

noisy_anpos = [("('JJ', 'NN', '.')", (1, 0)),               # 2
               ("('JJ', 'NN')", (1, 0)),
               ("('JJ', 'NNS', '.')", (1, 0)),  # 21
               ("('JJ', 'NNS')", (1, 0)),
               ("('NN', 'JJ', '.')", (0, 1))]          # 113 # 100 -->

interest_anpos = [("('NN', 'VBD', 'JJ', '.')", (0, 2)),                 # 8                 # nothing was ...
                  ("('DT', 'NN', 'VBD', 'JJ', '.')", (1, 3)),           # 1
                  ("('DT', 'NN', 'VBD', 'RB', 'JJ', '.')", (1, 4)),     # 4
                  ("('NN', 'VBD', 'RB', 'JJ', '.')", (0, 3)),           # 19
                  ("('RB', 'JJ', 'NN', '.')", (2, 1)),
                  ("('NN', 'VBD', 'JJ')", (0, 2)),                      # everything/nothing was ...
                  ("('NNP', 'VBD', 'JJ', '.')", (0, 2)),                # 22
                  ("('RB', 'JJ', 'NN', '.')", (2, 1)),                  # 13
                  ("('NNP', 'JJ', 'NN', '.')", (2, 1)),                 # 195
                  ("('DT', 'JJ', 'NN', 'VBZ', 'JJ', '.')", (2, [1, 4])),                # 196
                  ("('DT', 'NN', 'NNS', 'VBD', 'JJ', '.')", ((1, 2), 4)),               # 198
                  ("('NN', 'VBZ', 'RB', 'JJ', 'CC', 'JJ', '.')", (0, [3, 5])),          # 199
                  ("('DT', 'NN', 'VBD', 'JJ', 'CC', 'JJ', '.')", (1, [3, 5])),          # 7
                  ("('NN', 'VBD', 'JJ', 'CC', 'JJ', '.')", (0, [2, 4])),                # 17
                  ("('JJ', 'JJ', 'NN', '.')", (2, [0, 1])),                             # 41
                  ("('DT', 'NN', 'VBD', 'RB', 'JJ', 'CC', 'JJ', '.')", (1, [4, 6])),    # 8
                  ("('JJ', 'NN', 'NN', '.')", ((1, 2), 0)),                             # 31
                  ("('WP', 'DT', 'JJ', 'NN', '.')", (3, 2)),                            # 32
                  ("('DT', 'NN', 'VBZ', 'JJ', '.')", (1, 3)),                           # 3
                  ("('NNS', 'VBD', 'JJ', '.')", (0, 2)),                                # 56
                  ("('DT', 'NN', 'VBD', 'JJ')", (1, 3)),
                  ("('NN', 'VBD', 'RB', 'JJ', 'CC', 'JJ', '.')", (0, [3, 5])),          # 37
                  ("('JJ', 'NN', 'IN', 'NN', '.')", ((1, 3), 0)),                       # 74
                  # `good value for money`, Cant think of anything
                  ("('DT', 'NNS', 'VBD', 'JJ', '.')", (1, 3)),                          # 18
                  ("('JJ', 'NN', 'IN', 'DT', 'NN', '.')", ((1, 4), 0)),                 # 67
                  ("('JJ', 'CC', 'JJ', 'NN', '.')", (3, [0, 2])),                       # 106
                  ("('NN', 'VBD', 'RB', 'JJ')", (0, 3)),
                  ("('NNP', 'VBZ', 'JJ', '.')", (0, 2)),                                # 38
                  ("('DT', 'JJ', 'NN', '.')", (2, 1)),                                  # 30
                  ("('DT', 'NNS', 'VBD', 'RB', 'JJ', '.')", (1, 4)),                    # 36
                  ("('RB', 'JJ', 'NN')", (2, 1)),
                  ("('NN', 'VBZ', 'JJ', '.')", (0, 2)),                                 # 44
                  #  everything is good
                  ("('RB', 'JJ', 'CC', 'JJ', 'NN', '.')", (4, [1, 3])),                 # 101
                  ("('PRP$', 'NN', 'VBD', 'RB', 'JJ', '.')", (1, 4)),                   # 102
                  ("('JJ', 'NN', 'CC', 'JJ', 'NN', '.')", ([1, 4], [0, 3])),            # 97
                  ("('NNS', 'VBD', 'RB', 'JJ', '.')", (0, 3)),                          # 108
                  ("('DT', 'NN', 'VBZ', 'RB', 'JJ', '.')", (1, 4)),                     # 11
                  ("('NNP', 'VBD', 'JJ')", (0, 2)),
                  ("('PRP', 'VBD', 'DT', 'JJ', 'NN', '.')", (4, 3)),                    # 12       # we left a next day
                  ("('DT', 'NN', 'NN', 'VBD', 'JJ', '.')", ((1, 2), 4)),                # 23
                  ("('RB', 'JJ', 'NNS', '.')", (2, 1)),                                 # 104
                  ("('DT', 'NN', 'NN', 'VBD', 'RB', 'JJ', 'CC', 'JJ', '.')", ((1, 2), [5, 7])),         # 105
                  ("('PRP', 'VBD', 'RB', 'JJ', 'IN', 'DT', 'NN', '.')", (6, 3)),                        # 107
                  ("('JJ', 'NN', 'CC', 'NN', '.')", ([1, 3], 0)),                                       # 112
                  ("('DT', 'NN', 'VBD', 'RB', 'JJ', 'CC', 'RB', 'JJ', '.')", (1, [4, 7])),              # 114
                  ("('DT', 'NN', 'VBZ', 'IN', 'DT', 'JJ', 'NN', '.')", ([1, 6], 5)),                    # 115
                  ("('DT', 'NN', 'VBD', 'RB', 'JJ')", (1, 4)),
                  ("('JJ', 'NN', 'NN')", ((1, 2), 0)),
                  ("('NNP', 'RB', 'JJ', '.')", (0, 2)),                                         # 136
                  ("('DT', 'NN', 'VBD', 'JJ', 'CC', 'RB', 'VBN', '.')", (1, 3)),                # 137
                  ("('DT', 'RB', 'JJ', 'NN', '.')", (3, 2)),                                    # 138
                  ("('NN', 'NN', 'VBD', 'JJ', '.')", ((0, 1), 3)),                              # 63
                  ("('JJ', 'NN', ',', 'JJ', 'NN', '.')", ([1, 4], [0, 3])),                     # 116
                  ("('JJ', 'JJ', 'NN')", (2, [0, 1])),
                  ("('NN', 'RB', 'JJ', '.')", (0, 2)),
                  ("('DT', 'JJ', 'NN')", (2, 1)),
                  ("('DT', 'NN', 'IN', 'DT', 'NN', 'VBD', 'JJ', '.')", ((1, 4), 6)),            # 50
                  ("('NN', 'VBZ', 'RB', 'JJ', '.')", (0, 3)),                                   # 83
                  ("('NNP', 'VBD', 'RB', 'JJ', '.')", (0, 3)),                                  # 100
                  ("('DT', 'JJ', 'NN', 'VBD', 'JJ', '.')", (2, [1, 4])),                        # 60
                  ("('DT', 'VBZ', 'DT', 'JJ', 'NN', '.')", (4, 3)),                             # 61
                  ("('JJ', 'NN', ',', 'JJ', 'NN')", ([1, 4], [0, 3])),
                  ("('NNP', 'NN', 'VBD', 'JJ', '.')", ((0, 1), 3)),                             # 99
                  ("('JJ', ',', 'JJ', 'NN', '.')", (3, [0, 2])),                                # 185
                  ("('DT', 'NN', 'VBD', 'JJ', 'NN')", ([1, 4], 3)),                             # 188
                  ("('NNS', 'VBD', 'JJ')", (0, 2)),
                  ("('DT', 'NN', 'VBD', 'JJ', 'CC', 'RB', 'JJ', '.')", (1, [3, 6])),            # 35
                  ("('JJ', 'NN', 'CC', 'JJ', 'NN')", ([1, 4], [0, 3])),
                  ("('DT', 'NN', 'VBD', 'JJ', 'CC', 'DT', 'NN', 'VBD', 'JJ', '.')", ([1, 6], [3, 8])),      # 64
                  ("('DT', 'NN', 'NN', 'VBD', 'RB', 'JJ', '.')", ((1, 2), 5)),              # 53            # 100 -->
                  ("('DT', 'NN', 'IN', 'DT', 'NN', 'VBZ', 'JJ', '.')", ((1, 4), 6)),                          # 55
                  ("('DT', 'NN', 'NN', 'VBZ', 'JJ', '.')", ((1, 2), 4)),                                    # 59
                  ("('NNS', 'VBP', 'JJ', '.')", (0, 2)),                                                    # 75
                  ("('PRP', 'VBZ', 'DT', 'JJ', 'NN', '.')", (4, 3)),                                        # 78
                  ("('NNP', 'RB', 'JJ', 'CC', 'JJ', '.')", (0, [2, 4])),                                    # 176
                  ("('NNS', 'VBD', 'JJ', 'CC', 'JJ', '.')", (0, [2, 4])),                                   # 117
                  ("('NN', 'VBZ', 'JJ')", (0, 2)),
                  ("('JJ', 'NN', 'NNS', '.')", ((1, 2), 0)),                                        # 183
                  ("('NNP', 'VBZ', 'RB', 'JJ', '.')", (0, 3)),                                      # 184
                  ("('JJ', 'NN', 'VBD', 'JJ', '.')", (1, [0, 3])),
                  ("('NN', 'VBD', 'JJ', 'CC', 'JJ')", (0, [2, 4])),
                  ("('DT', 'NN', 'VBD', 'RB', 'RB', 'JJ', '.')", (1, 5)),                           # 82
                  ("('DT', 'VBD', 'DT', 'JJ', 'NN', '.')", (4, 3)),                                 # 86
                  ("('DT', 'NN', 'NN', 'VBD', 'JJ', 'CC', 'JJ', '.')", ((1, 2), [4, 6])),           # 90
                  ("('DT', 'NN', 'VBZ', 'DT', 'JJ', 'NN', '.')", ([1, 5], 4)),                      # 91
                  ("('PRP', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'DT', 'NN', '.')", ((4, 7), 3)),         # 92
                  ("('DT', 'NN', 'VBD', 'JJ', ',', 'JJ', 'CC', 'JJ', '.')", (1, [3, 5, 7])),        # 93
                  ("('DT', 'NNS', 'VBD', 'JJ', 'CC', 'JJ', '.')", (1, [3, 5])),                     # 39
                  ("('RB', 'DT', 'JJ', 'NN', '.')", (3, 2)),                                        # 40
                  ("('JJ', 'JJ', 'NNS', '.')", (2, [0, 1])),
                  ("('NNS', 'RB', 'JJ', '.')", (0, 2)),
                  ("('JJ', 'NN', 'IN', 'NN')", ((1, 3), 0)),
                  ("('JJ', 'NN', 'CC', 'NN')", ([1, 3], 0)),
                  ("('JJ', 'NN', 'IN', 'DT', 'JJ', 'NN', '.')", ((1, 5), 0)),                       # 147
                  ("('JJ', 'CC', 'JJ', 'NN')", (3, [0, 2])),
                  ("('JJ', 'NN', 'IN', 'JJ', 'NN', '.')", ((1, 4), 0)),
                  ("('DT', 'NN', 'VBD', 'JJ', 'JJ', '.')", (1, [3, 4])),
                  ("('DT', 'NNS', 'VBP', 'JJ', '.')", (1, 3)),                                      # 25
                  ("('DT', 'NN', 'CC', 'NN', 'VBD', 'JJ', '.')", ([1, 3], 5)),                      # 118
                  ("('NNS', 'VBP', 'JJ', 'CC', 'JJ', '.')", (0, [2, 4])),                           # 120
                  ("('DT', 'NN', 'VBD', 'JJ', 'RB', '.')", (1, 3)),                                 # 121
                  ("('NN', 'VBZ', 'JJ', 'CC', 'JJ', '.')", (0, [2, 4])),                            # 123
                  ("('DT', 'NN', 'VBZ', 'JJ', 'CC', 'JJ', '.')", (1, [3, 5])),                      # 34
                  ("('NN', 'VBD', 'JJ', 'CC', 'RB', 'JJ', '.')", (0, [2, 5])),                      # 146
                  ("('NNP', 'JJ', 'CC', 'JJ', '.')", (0, [1, 3])),
                  ("('NN', 'NN', 'VBD', 'RB', 'JJ', '.')", ((0, 1), 4)),
                  ("('NN', 'RB', 'JJ')", (0, 2)),
                  ("('DT', 'NNS', 'VBP', 'RB', 'JJ', '.')", (1, 4)),                                # 45
                  ("('DT', 'NNS', 'VBP', 'JJ', 'CC', 'JJ', '.')", (1, [3, 5])),
                  ("('RB', 'JJ', 'NN', 'NN', '.')", ((2, 3), 1)),                                   # 191
                  ("('DT', 'NN', 'IN', 'DT', 'NN', 'VBZ', 'RB', 'JJ', '.')", ((1, 4), 7)),          # 192
                  ("('JJ', 'NN', 'JJ', 'NN')", ([1, 3], [0, 2])),
                  ("('NNP', 'VBZ', 'JJ')", (0, 2)),
                  ("('VBD', 'DT', 'JJ', 'NN', '.')", (3, 2)),                                       # 150
                  ("('RB', ',', 'DT', 'JJ', 'NN', '.')", (4, 3)),                                   # 151
                  ("('DT', 'NN', 'VBZ', 'RB', 'RB', 'JJ', '.')", (1, 5)),                           # 170
                  ("('DT', 'NNS', 'VBP', 'JJ', 'CC', 'RB', 'JJ', '.')", (1, [3, 6])),               # 172
                  ("('RB', ',', 'DT', 'NN', 'VBD', 'JJ', '.')", (3, 5)),                            # 174
                  ("('DT', 'NN', 'VBD', 'RB', 'JJ', 'CC', 'DT', 'NN', 'VBD', 'JJ', '.')", ([1, 7], [4, 9])),    # 175
                  ("('DT', 'NN', 'VBZ', 'JJ')", (1, 3)),
                  ("('NNS', 'VBP', 'RB', 'JJ', '.')", (0, 3)),                                      # 148
                  ("('NN', 'VBD', 'JJ', 'JJ', '.')", (0, [2, 3])),
                  ("('PRP$', 'NN', 'VBD', 'JJ', '.')", (1, 3)),
                  ("('DT', 'NNS', 'VBD', 'RB', 'JJ', 'CC', 'JJ', '.')", (1, [4, 6])),               # 95
                  ("('PRP', 'VBD', 'IN', 'DT', 'JJ', 'NN', '.')", (5, 4)),                          # 96
                  ("('DT', 'NN', 'VBD', 'DT', 'JJ', 'NN', '.')", ([1, 5], 4)),                      # 110
                  ("('PRP$', 'NN', 'VBD', 'JJ', 'CC', 'JJ', '.')", (1, [3, 5])),                    # 111
                  ("('NNP', 'RB', 'JJ')", (0, 2)),
                  ("('NN', 'VBD', 'RB', 'RB', 'JJ', '.')", (0, 4)),
                  ("('JJ', 'NN', 'IN', 'DT', 'NN')", ((1, 4), 0)),
                  ("('JJ', 'NN', 'CC', 'NNS', '.')", ([1, 3], 0)),
                  ("('DT', 'NN', 'VBZ', 'RB', 'JJ', 'CC', 'JJ', '.')", (1, [4, 6])),                # 43
                  ("('DT', 'JJ', 'NN', 'VBD', 'RB', 'JJ', '.')", (2, [1, 5])),                      # 126
                  ("('DT', 'NNS', 'VBP', 'RB', 'JJ', 'CC', 'JJ', '.')", (1, [4, 6])),               # 127
                  ("('DT', 'NN', 'NN', 'VBZ', 'RB', 'JJ', '.')", ((1, 2), 5)),                      # 132
                  ("('PRP', 'VBD', 'DT', 'RB', 'JJ', 'NN', '.')", (5, 4)),                          # 134
                  ("('DT', 'NN', 'VBZ', 'JJ', 'CC', 'RB', 'JJ', '.')", (1, [3, 6])),                # 135
                  ("('JJ', 'NN', 'IN')", ((1, 2), 0)),                                              # 177
                  ("('RB', ',', 'PRP', 'VBD', 'DT', 'JJ', 'NN', '.')", (6, 5)),                     # 178
                  ("('DT', 'NN', 'VBP', 'RB', 'JJ', 'CC', 'JJ', '.')", (1, [4, 6])),                # 179
                  ("('DT', 'NN', 'VBD', 'JJ', 'CC', 'DT', 'NN', 'VBD', 'RB', 'JJ', '.')", ([1, 6], [3, 9])),       # 180
                  ("('DT', 'NNS', 'VBD', 'JJ', 'CC', 'RB', 'JJ', '.')", (1, [3, 6])),               # 181
                  ("('JJ', 'NN', 'TO', 'VB', '.')", (1, 0)),                                        # 200
                  ("('NN', 'VBD', 'JJ', 'NN')", (0, 2)),
                  ("('DT', 'NN', 'IN', 'DT', 'NN', 'VBD', 'RB', 'JJ', '.')", ((1, 4), 7)),          # 140
                  ("('JJ', 'NN', 'RB', '.')", (1, 0)),                                              # 143
                  ("('DT', 'NN', 'PRP', 'VBD', 'JJ', '.')", (1, 4)),                                # 145
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
        tok = [wd for wd in tokens[loc[0]:loc[1]+1] if re.sub(r'[^0-9a-zA-Z ]+', '', wd).isalnum() and
               len(re.sub(r'[^0-9a-zA-Z ]+', '', wd)) > 1]
        if len(tok) > 0:
            d.append(" ".join(tok))
    elif type(loc) is list:
        tok = [tokens[l] for l in loc if re.sub(r'[^0-9a-zA-Z ]+', '', tokens[l]).isalnum() and
               len(re.sub(r'[^0-9a-zA-Z ]+', '', tokens[l])) > 1]
        if len(tok) > 0:
            d.extend(tok)
    elif type(loc) is int:
        if re.sub(r'[^0-9a-zA-Z ]+', '', tokens[loc]).isalnum() and len(re.sub(r'[^0-9a-zA-Z ]+', '', tokens[loc])) > 1:
            d.append(tokens[loc])
    else:
        pass
    return d


def extract_two(tag_dict, interest_tags):
    noun_dict = Counter()
    adj_dict = Counter()
    for anpos in interest_tags:
        # print(anpos)
        pos, loc = anpos
        nloc, aloc = loc
        for sent in tag_dict[pos]:
            tokens = sent.lower().split()
            noun_dict.update(location(tokens, nloc))
            adj_dict.update(location(tokens, aloc))
    return noun_dict, adj_dict


def extract_one(tag_dict, interest_tags):
    pos_dict = Counter()
    for apos in interest_tags:
        print(apos)
        pos, loc = apos
        for sent in tag_dict[pos]:
            tokens = sent.lower().split()
            pos_dict.update(location(tokens, loc))
    return pos_dict


fuzzy_aspects = ["everything", "everythings", "nothing", "nothing everything", "thing", "things", "lot", "day", "all",
                 "others", "anything", "evrything", "hour", "part", "fun", "mess", "else", "bit", "night", "b", "way",
                 "super", "none", "wife", "pretty", "dislike", "complaints", "complaint", "everyone", "time", "joke"]


def process_noun(noun_dict):
    for wd in fuzzy_aspects:
        if wd in noun_dict:
            noun_dict.pop(wd)
    return noun_dict


def extract_jj_nn(rfile):
    tag_count, tag_dict = joblib.load(rfile)
    noun_dict = Counter()
    adj_dict = Counter()
    nouns, adjs = extract_two(tag_dict, interest_anpos)
    noun_dict.update(nouns)
    adj_dict.update(adjs)

    # nouns, adjs = extract_two(tag_dict, noisy_anpos)
    # noun_dict.update(nouns)
    # adj_dict.update(adjs)
    #
    # adjs = extract_one(tag_dict, noisy_apos)
    # adj_dict.update(adjs)
    #
    # nouns = extract_one(tag_dict, noisy_npos)
    # noun_dict.update(nouns)

    return noun_dict, adj_dict


if __name__ == "__main__":
    """
    python trip_extraction.py --rfile /media/data/hotels/kdd11/processed/extracted_tag/kdd11_tag_count_dict_c5.pkl --wfile /media/data/hotels/kdd11/processed/extracted_tag/tag_aspects_v2.csv
    """
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--rfile', help='read file',
                           default="/media/data/hotels/kdd11/processed/extracted_tag/kdd11_tag_count_dict_c5.pkl",
                           type=str)

    argparser.add_argument('--wfile', help='writen file',
                           default="/media/data/hotels/kdd11/processed/extracted_tag/tag_aspects.csv",
                           type=str)

    args = argparser.parse_args()

    tag_count, tag_dict = joblib.load(args.rfile)

    noun_dict, adj_dict = extract_jj_nn(args.rfile)

    write_csv_lines(args.wfile, noun_dict.most_common())
