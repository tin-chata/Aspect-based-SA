"""
Created on 2018-09-14
@author: duytinvo
"""
import csv
import numpy as np


def write_csv(data, file_name):
    data.sort(key=lambda x: len(x[0]), reverse=False)
    with open(file_name, "w", newline='') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(data)


def write_dataset(data_file, train_file, val_file, test_file, tr_ratio=0.9, val_ratio=0.95):
    corpus = []
    with open(data_file, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            corpus.append(tuple(line))
    corpus = list(set(corpus))
    np.random.shuffle(corpus)
    train, val, test = np.split(corpus, [int(tr_ratio * len(corpus)), int(val_ratio * len(corpus))])
    write_csv(train.tolist(), train_file)
    write_csv(val.tolist(), val_file)
    write_csv(test.tolist(), test_file)


if __name__ == "__main__":
    """
    python create_dataset.py --corpus_file /media/data/booking.com/aspect_merged.csv --train_file /media/data/aspectSA/booking_train_v3.csv --val_file /media/data/aspectSA/booking_val_v3.csv --test_file /media/data/aspectSA/booking_test_v3.csv
    """

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--corpus_file', help='corpus file',
                           default="/media/data/hotel/booking/aspect_all.csv",
                           type=str)
    argparser.add_argument('--train_file', help='training file',
                           default="/media/data/aspectSA/booking_train_v1.csv",
                           type=str)
    argparser.add_argument('--val_file', help='validating file',
                           default="/media/data/aspectSA/booking_val_v1.csv",
                           type=str)
    argparser.add_argument('--test_file', help='testing file',
                           default="/media/data/aspectSA/booking_test_v1.csv",
                           type=str)
    argparser.add_argument('--tr_ratio', help='splitting rate of training', default=0.8, type=float)
    argparser.add_argument('--val_ratio', help='splitting rate of validating', default=0.9, type=float)
    args = argparser.parse_args()

    write_dataset(args.corpus_file, args.train_file, args.val_file, args.test_file, args.tr_ratio, args.val_ratio)