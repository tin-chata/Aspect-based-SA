"""
Created on 2018-09-20
@author: duytinvo
"""
from process_data import remove_symbols, write_sorted_txt_lines


def sftokenize(readfile, port=9000):
    from corenlpsf import StanfordNLP
    sNLP = StanfordNLP(port=port)
    data = []
    c = 0
    with open(readfile, "r") as f:
        for line in f:
            sents = sNLP.ssplit(line)
            for sent in sents:
                psent = remove_symbols(" ".join(sent))
                if len(psent.split()) >= 2 and len(" ".join(sent)) >= 6:
                    data.append(" ".join(sent))
                    c += 1
                    if c % 10000 == 0:
                        print("Processing %d sentences" % c)
    return set(data)


def clean_data(pos_rfile="/media/data/hotels/booking_v2/raw_data/booking.negative.txt",
               neg_rfile="/media/data/hotels/booking_v2/raw_data/booking.positive.txt",
               pos_wfile="/media/data/hotels/booking_v2/processed/extracted_sent/positive.set.txt",
               neg_wfile="/media/data/hotels/booking_v2/processed/extracted_sent/negative.set.txt"):
    pos_data = sftokenize(pos_rfile)
    neg_data = sftokenize(neg_rfile)
    inter_sents = pos_data.intersection(neg_data)
    pos_data.difference_update(inter_sents)
    neg_data.difference_update(inter_sents)
    write_sorted_txt_lines(pos_wfile, list(pos_data))
    write_sorted_txt_lines(neg_wfile, list(neg_data))


if __name__ == "__main__":
    """
    python extract_sent.py --pos_rfile /media/data/hotels/booking_v3/raw_data/canada_positive.txt --neg_rfile /media/data/hotels/booking_v3/raw_data/canada_negative.txt  --pos_wfile /media/data/hotels/booking_v3/processed/extracted_sent/canada_positive.set.txt --neg_wfile /media/data/hotels/booking_v3/processed/extracted_sent/canada_negative.set.txt
    """
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--pos_rfile', help='Input positive file',
                           default="/media/data/hotels/booking_v2/raw_data/booking.negative.txt",
                           type=str)

    argparser.add_argument('--neg_rfile', help='Input negative file',
                           default="/media/data/hotels/booking_v2/raw_data/booking.positive.txt",
                           type=str)

    argparser.add_argument('--pos_wfile', help='Output positive file',
                           default="/media/data/hotels/booking_v2/processed/extracted_sent/positive.set.txt",
                           type=str)
    argparser.add_argument('--neg_wfile', help='Output negative file',
                           default="/media/data/hotels/booking_v2/processed/extracted_sent/negative.set.txt",
                           type=str)

    argparser.add_argument('--sf_port', help='corenlp port',
                           default=9000,
                           type=int)

    args = argparser.parse_args()

    clean_data(args.pos_rfile, args.neg_rfile, args.pos_wfile, args.neg_wfile)

