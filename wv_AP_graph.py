"""
Created on Tue Mar  6 15:12:42 2018

@author: dtvo
"""
import argparse
import numpy as np
from corenlpsf import sNLP
import networkx as nx
import matplotlib.pyplot as plt


def norm_embs(emb_file):
    with open(emb_file, 'r') as f:
        vectors = {}
        words = []
        for line in f:
            vals = line.rstrip().split(' ')
            if len(vals) == 2:
                continue
            vectors[vals[0]] = [float(x) for x in vals[1:]]
            words.append(vals[0])
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word]] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


def emb_term(input_term, W, vocab):
    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
            print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term]])
            else:
                vec_result += W[vocab[term]]
        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return np.zeros(1)
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2, ) ** (0.5))
    vec_norm = (vec_result.T / d).T
    return vec_norm


def char_olp(wd1, wd2):
    return len(set(wd1).intersection(set(wd2)))/len(set(wd2))


def topNsim(input_term, W, vocab, ivocab, N):
    vec_norm = np.copy(W[vocab[input_term]])
    dist = np.dot(W, vec_norm.T)
    index = vocab[input_term]
    dist[index] = -np.Inf
    a = np.argsort(-dist)
    words = []
    for x in a:
        wd = ivocab[x]
        sim = dist[x]
        if sim <= 0.5 or len(words) >= N:
            break
        else:
            olp_scr = char_olp(input_term, wd)
            if olp_scr < 0.9:
                tags = sNLP.pos(wd)
                wd, tag = tags[0]
                if tag in ["NN", "NNS"] and wd.isalnum():
                    words.append((wd, sim))
    print(input_term, "\t", words)
    return words


def addgraph(G, input_term, W, vocab, ivocab, width=10, depth=10, scale=1.0):
    children = topNsim(input_term, W, vocab, ivocab, width)
    if depth == 0 or len(children) == 0:
        return G
    else:
        words, scrs = zip(*children)
        for i, wd in enumerate(words):
            scr = scrs[i]
            if G.has_node(wd):
                if G.out_degree(wd) == 0 and G.in_degree(wd) == 1:
                    G.remove_node(wd)
                else:
                    continue

            weight = scr*scale
            G.add_node(wd, scale=weight)
            G.add_edge(input_term, wd, weight=weight)
            G = addgraph(G, wd, W, vocab, ivocab, width, depth - 1, weight)
        return G


def add_emb2graph(G, W, vocab):
    count_wv = 0
    sum_wv = 0
    for node_name in G.nodes():
        emb = np.copy(W[vocab[node_name]])
        G.node[node_name]["wv"] = emb
        sum_wv += emb
        count_wv += 1

    vec_norm = np.zeros(sum_wv.shape)
    d = (np.sum(sum_wv ** 2, ) ** (0.5))
    vec_norm = (sum_wv.T / d).T
    G.graph["norm_wv"] = vec_norm
    G.graph["count_wv"] = count_wv
    return G


def build_graph(topics, W, vocab, ivocab, width=10, depth=3):
    name = "_".join(topics)
    G = nx.DiGraph(name=name + "_GROUP")
    for topic in topics:
        if G.has_node(topic):
            continue
        G.add_node(topic, scale=1.0)
        G = addgraph(G, topic, W, vocab, ivocab, width, depth, scale=1.0)
    G = add_emb2graph(G, W, vocab)
    nx.write_gpickle(G, "/Users/duytinvo/Projects/aspectSA/data/wv_groups/hotel_%s.gpickle" % G.graph["name"].lower())
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, width=1, with_labels=True, font_size=8)
    # plt.savefig("./data/wv_groups/%s.png" % (topic + "_group"))
    # plt.show()
    return G


def build_graphs(W, vocab, ivocab, group_terms):
    for g in group_terms:
        G = build_graph(group_terms[g], W, vocab, ivocab, width=40, depth=1)


if __name__ == "__main__":
    W, vocab, ivocab = norm_embs("/Users/duytinvo/Projects/aspectSA/data/w2v/sf_hotel.pro.v2.vec")
    group_terms = {
        "room": ["room", "rooms"],  # room
        "price": ["price", "bill", "cost", "discount",
                  "expenditure", "fare", "fee", "payment",
                  "rate", "tariffs"],  # value
        "service": ["staff", "support", "service"],  # Staff
        "ambiance": ["ambiance"],  # Ambiance
        "food": ["food"],  # food and beverage
        "location": ["location", "area", "district", "locale", "neighborhood", "region", "spot", "venue"],  # Location
        "amenity": ["amenities", "facilities"],  # Service and facility
    }
    build_graphs(W, vocab, ivocab, group_terms)




