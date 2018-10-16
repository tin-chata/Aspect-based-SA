"""
Created on 2018-08-31
@author: duytinvo
"""
from corenlpsf import count_tags, StanfordNLP

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--rfile', help='read file',
                           default="/media/data/hotels/kdd11/processed/extracted_sent/kdd11_all_sents.txt",
                           type=str)

    argparser.add_argument('--wfile', help='writen file',
                           default="/media/data/hotels/kdd11/processed/extracted_tag/kdd11_tag_count_dict.pkl",
                           type=str)

    argparser.add_argument('--port', help='port number', default=8000, type=int)

    args = argparser.parse_args()

    # rfile = "/media/data/hotels/kdd11/processed/extracted_sent/kdd11_all_sents.txt"
    # wfile = "/media/data/hotels/kdd11/processed/extracted_tag/tag_count_dict.pkl"
    sNLP = StanfordNLP(port=args.port)
    pos_dict, pos_count = count_tags(args.wfile, args.rfile, sNLP)
