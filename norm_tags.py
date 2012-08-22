import sys,os
import sqlite3

#import pickle
#import networkx as nx

from datetime import datetime
from optparse import OptionParser
from count_bigram_flickr import norm_tag, read_unigram #read_bigram(src_file, bg_dict)
from compare_cooc import read_bigram_list


def norm_words(in_wlist, db_file, addl_vocab=[]):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    wd = {}
    cn_vocab = {}
    null_count = 0
    for w in in_wlist:
        ww = w.split()
        vv = map(lambda s:norm_tag(s, cursor, addl_vocab), ww)
        vv = filter(lambda s: len(s), vv)
        wd[w] = vv
        null_count += (not vv)
        if vv :
            for v in vv:
                if v not in cn_vocab:
                    cn_vocab[v] = 1
            
    #wd = dict(izip(ww, vv))
    conn.close()
    
    cn_vocab = cn_vocab.keys()
    return cn_vocab


if __name__ == '__main__':
    argv = sys.argv 

    make_vocab(argv)    