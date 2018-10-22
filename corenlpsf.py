"""
Created on 2018-08-17
@author: duytinvo
"""
import re
import json
from nltk.tree import Tree
from sklearn.externals import joblib
from collections import defaultdict, Counter
from stanfordcorenlp import StanfordCoreNLP


class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

        self.ss = {
            'annotators': 'ssplit',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def ssplit(self, sentence):
        ss = json.loads(self.nlp.annotate(sentence, properties=self.ss))
        return [[token['originalText'] for token in s['tokens']] for s in ss['sentences']]

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens


def Extract3rdPhrases(myTree, phrase):
    """
    Extract NPs having height of 3 only
    """
    myPhrases = []
    if myTree.height() <= 3:
        if myTree.label() == phrase:
            myPhrases.append(myTree.copy(True))
    else:
        for child in myTree:
            if type(child) is Tree:
                list_of_phrases = Extract3rdPhrases(child, phrase)
                if len(list_of_phrases) > 0:
                    myPhrases.extend(list_of_phrases)
    return myPhrases


def extract_3rdnp(sent, sNLP=StanfordNLP()):
    """
    Extract 3-height NPs
    """
    pt = sNLP.parse(sent)
    parsedtree = Tree.fromstring(pt)
    NPs = Extract3rdPhrases(parsedtree, "NP")
    d = []
    for np in NPs:
        s, t = zip(*np.pos())
        s = list(s)
        t = list(t)
        if t[0] == "DT":
            s.pop(0)
            t.pop(0)
        d.append(" ".join(s))
    return d


def extract_3rdanp(sent, sNLP=StanfordNLP()):
    """
    Extract 3-height ADJ --> NPs
    """
    adj = ("JJ", "JJR", "JJS")
    pt = sNLP.parse(sent)
    parsedtree = Tree.fromstring(pt)
    NPs = Extract3rdPhrases(parsedtree, "NP")
    d = []
    for np in NPs:
        s, t = zip(*np.pos())
        s = list(s)
        t = list(t)
        if t[0] == "DT":
            s.pop(0)
            t.pop(0)
        if len(t) > 0 and t[0] in adj:
            d.append(" ".join(s))
    return d


def ExtractallPhrases(myTree, phrase):
    """
    Extract all non-overlap NPs
    """
    myPhrases = []
    if myTree.label() == phrase:
        myPhrases.append(myTree.copy(True))
    if len(myPhrases) == 0:
        for child in myTree:
            if type(child) is Tree:
                list_of_phrases = ExtractallPhrases(child, phrase)
                if len(list_of_phrases) > 0:
                    myPhrases.extend(list_of_phrases)
    return myPhrases


def extract_allnp(sent, sNLP=StanfordNLP()):
    """
    Extract all NPs
    """
    pt = sNLP.parse(sent)
    parsedtree = Tree.fromstring(pt)
    NPs = ExtractallPhrases(parsedtree, "NP")
    d = []
    for np in NPs:
        s, t = zip(*np.pos())
        s = list(s)
        t = list(t)
        c = 0
        for p in t:
            if p in ["NN", "NNS", "NNP", "NNPS"]:
                break
            else:
                c += 1
        for i in range(c):
            s.pop(0)
            t.pop(0)
        if len(s) > 0:
            d.append(" ".join(s))
    return d


noun_phases = [
               ('NNS', 'IN', 'DT', 'NN'), ('NNS', 'IN', 'DT', 'NNS'), ('NNS', 'IN', 'DT', 'NNP'), ('NNS', 'IN', 'DT', 'NNPS'),
               ('NNS', 'IN', 'NN'), ('NNS', 'IN', 'NN'), ('NNS', 'IN', 'NNS'), ('NNS', 'IN', 'NNP'), ('NNS', 'IN', 'NNPS'),
               ('NN', 'IN', 'DT', 'NN'), ('NN', 'IN', 'DT', 'NNS'), ('NN', 'IN', 'DT', 'NNP'), ('NN', 'IN', 'DT', 'NNPS'),
               ('NN', 'IN', 'NN'), ('NN', 'IN', 'NN'), ('NN', 'IN', 'NNS'), ('NN', 'IN', 'NNP'), ('NN', 'IN', 'NNPS'),
               ('NNP', 'IN', 'DT', 'NN'), ('NNP', 'IN', 'DT', 'NNS'), ('NNP', 'IN', 'DT', 'NNP'), ('NNP', 'IN', 'DT', 'NNPS'),
               ('NNPS', 'IN', 'NN'), ('NNPS', 'IN', 'NN'), ('NNPS', 'IN', 'NNS'), ('NNPS', 'IN', 'NNP'), ('NNPS', 'IN', 'NNPS')
               ]


def findsublist(haystack, needle):
    h = len(haystack)
    n = len(needle)
    skip = {needle[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    idx = []
    while i < h:
        for j in range(n):
            if haystack[i - j] != needle[-j - 1]:
                i += skip.get(haystack[i], n)
                break
        else:
            idx.append(i - n + 1)
            i += n
            # return i - n + 1
    return idx


def merge_np(tags, nps):
    for np in nps:
        if re.search(r"\b%s\b" % (" ".join(np)), " ".join(tags)):
            ids = findsublist(tags, np)
            if len(ids) > 0:
                for idx in ids:
                    tags[idx: idx + len(np)] = [np[0]] * len(np)
    return tags


def extract_nn(sent, sNLP=StanfordNLP()):
    pos = sNLP.pos(sent)
    nps = []
    cur = []
    s, t = zip(*pos)
    t = merge_np(list(t), noun_phases)
    for i in range(len(pos)-1):
        if t[i] in ["NN", "NNS", "NNP", "NNPS"]:
            cur += [s[i]]
            if t[i + 1] not in ["NN", "NNS", "NNP", "NNPS"]:
                nps.append(" ".join(cur).lower())
                cur = []
    if t[-1] in ["NN", "NNS", "NNP", "NNPS"]:
        cur += [s[-1]]
    if len(cur) != 0:
        nps.append(" ".join(cur).lower())
    return nps


def count_tags(wfile, rfile, sNLP=StanfordNLP()):
    pos_count = Counter()
    pos_dict = defaultdict(list)
    with open(rfile, "r") as f:
        for line in f:
            wds, pos = zip(*sNLP.pos(line.strip()))
            pos_count[str(pos)] += 1
            pos_dict[str(pos)].extend([" ".join(wds)])
    joblib.dump([pos_count, pos_dict], wfile)
    return pos_dict, pos_count


if __name__ == '__main__':
    """
    java --add-modules java.se.ee -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
    """
    sNLP = StanfordNLP(port=9000)








