"""
Created on 2018-08-13
@author: duytinvo
"""
import re
import csv
import itertools


class Txtfile(object):
    """
    Read txt file
    """
    def __init__(self, fname, word2idx=None, firstline=False, limit=-1):
        self.fname = fname
        self.firstline = firstline
        if limit < 0:
            self.limit = None
        else:
            self.limit = limit
        self.word2idx = word2idx
        self.length = None

    def __iter__(self):
        with open(self.fname, newline='', encoding='utf-8') as f:
            f.seek(0)
            if self.firstline:
                # Skip the header
                next(f)
            for line in itertools.islice(f, self.limit):
                sent = line.strip()
                sent = Txtfile.process_sent(sent)
                if self.word2idx is not None:
                    sent = self.word2idx(sent)
                yield sent

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length

    @staticmethod
    def process_sent(sent):
        sent = re.sub("[\'\"`]+", '', sent)
        sent = re.sub('[^0-9a-zA-Z ]+', ' ', sent)
        sent = sent.lower()
        return sent.strip().split()


def remove_symbols(sent):
    psent = re.sub("[\'\"`]+", '', sent)
    psent = re.sub(r'[^0-9a-zA-Z ]+', ' ', psent)
    psent = " ".join(psent.split())
    return psent


def write_csv_lines(filename, data):
    with open(filename, "w") as f:
        csvwriter = csv.writer(f)
        for line in data:
            csvwriter.writerow(line)


def write_txt_line(data, file_name):
    c = 0
    with open(file_name, "w") as f:
        for line in data:
            if len(line) >= 2 and len(" ".join(line)) >= 6:
                f.write(" ".join(line)+"\n")
                c += 1
                if c % 10000 == 0:
                    print("Write %d lines" % c)


def write_sorted_txt_lines(filename, data):
    data.sort(key=lambda x: len(x), reverse=False)
    with open(filename, "w") as f:
        for line in data:
            f.write(line + "\n")


def sorted_file(readfile, writefile):
    data = []
    c = 0
    with open(readfile, "r") as f:
        for line in f:
            sent = line.strip()
            psent = remove_symbols(sent)
            if len(psent.split()) >= 2 and len(psent) >= 6:
                data.append(sent)
                c += 1
                if c % 10000 == 0:
                    print("Processing %d sentences" % c)
    data = list(set(data))
    write_sorted_txt_lines(writefile, data)


def process_sent_ap(sent, asp):
    asp = remove_symbols(asp)
    sent = remove_symbols(sent)
    if len(asp.split()) >= 2:
        asp_rep = asp.replace(" ", "_")
        sent_rep = sent.replace(asp, asp_rep)
    else:
        asp_rep = asp
        sent_rep = sent
    sent_rep = sent_rep.lower()
    asp_rep = asp_rep.lower()
    return sent_rep, asp_rep


if __name__ == "__main__":
    """
    python process_data.py --inp_file /media/data/hotels/kdd11/extracted_rev/kdd11_all_sents.txt --out_file /media/data/hotels/kdd11/extracted_rev/kdd11_all_sents.pro.txt
    """
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--inp_file', help='Input file',
                           default="/media/data/hotels/kdd11/extracted_rev/kdd11_all_sents.txt",
                           type=str)
    argparser.add_argument('--out_file', help='Output file',
                           default="/media/data/hotels/kdd11/extracted_rev/kdd11_all_sents.pro.txt",
                           type=str)
    args = argparser.parse_args()

    sorted_file(args.inp_file, args.out_file)
