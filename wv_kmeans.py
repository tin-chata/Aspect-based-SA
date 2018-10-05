"""
Created on Tue Mar  6 15:12:42 2018

@author: dtvo
"""
import argparse
import numpy as np


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
    vec_result = np.zeros(W.shape[1])
    for idx, term in enumerate(input_term):
        if term in vocab:
            vec_result += W[vocab[term]]
    vec_result = vec_result/len(input_term)
    if np.sum(vec_result) != 0:
        vec_norm = np.zeros(vec_result.shape)
        d = (np.sum(vec_result ** 2, ) ** (0.5))
        vec_norm = (vec_result.T / d).T
        return vec_norm
    else:
        return vec_result


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
            words.append((wd, sim))
    print(input_term, "\t", words)
    return words


def kmeanNsim(input_vec, W, ivocab, N):
    d = (np.sum(input_vec ** 2, ) ** (0.5))
    vec_norm = (input_vec.T / d).T
    dist = np.dot(W, vec_norm.T)
    a = np.argsort(-dist)
    words = []
    for x in a:
        wd = ivocab[x]
        sim = dist[x]
        if sim <= 0.0 or len(words) >= N:
            break
        else:
            words.append((wd, sim))
    return words


def readfile(filename, W, vocab):
    data = []
    with open(filename, "r") as f:
        for line in f:
            sent = line.strip().lower().split()
            if 5 < len(sent) < 100:
                sent_vec = emb_term(sent, W, vocab)
                data.append(sent_vec)
                if len(data) == 500000:
                    break
    return data


if __name__ == "__main__":
    from sklearn import cluster

    filename = "/media/data/hotel/hongmin_wang/kdd11_reviews_v2_2.pro.txt"
    W, vocab, ivocab = norm_embs("/Users/duytinvo/Projects/aspectSA/data/w2v/sf_hotel.pro.v2_2.vec")
    data = readfile(filename, W, vocab)
    NUM_CLUSTERS = 14
    kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
    kmeans.fit(data)
    #
    # labels = kmeans.labels_
    # centroids = kmeans.cluster_centers_





