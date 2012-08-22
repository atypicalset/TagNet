import sys,os
import sqlite3
#import divisi2
#import pickle
#import networkx as nx
import random
from operator import itemgetter
#from datetime import datetime
from optparse import OptionParser
from count_bigram_flickr import norm_tag #, read_unigram #read_bigram(src_file, bg_dict)
#from compare_cooc import read_bigram_list
from imgtags import cache_flickr_info

FLICKR_KEY_FILE = 'flickr.key.txt'

def rank_tags(argv):
    if len(argv)<2:
            argv = ['-h']
    parser = OptionParser(description='rank tags of a single image + compose sentences')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db and other data')
    parser.add_option('-n', '--num_output', dest='num_output', type="int", 
                      default=3, help='number of output images to examine')
    parser.add_option("", '--db_dict', dest='db_dict', default="dict.db", help='dictionary')
    parser.add_option("", '--vocab_score', dest='vocab_score', default="flickr_vscore.txt", 
                      help='file containing vocabulary count')
    parser.add_option("", '--tag_file', dest="tag_file", default="demo-data/24.cache", help="")
    parser.add_option("", '--wn_list', dest='wn_list', default="wnet-50.txt", help='')
    parser.add_option("", '--addl_vocab', dest='addl_vocab', default="places_etc.txt", help='')
    
    (opts, __args) = parser.parse_args(sys.argv)
    
    # intersect the two dictionaries first
    db_dict = os.path.join(opts.db_dir, opts.db_dict)
    
    tag_file = os.path.join(opts.db_dir, opts.tag_file)
    
    addl_vocab = open(os.path.join(opts.db_dir, opts.addl_vocab), 'rt').read().split()
    vocab_lines = open(os.path.join(opts.db_dir, opts.vocab_score), 'rt').read().split("\n")
    vocab_lines = filter(len, vocab_lines)
    vocab_score = {}
    for vl in vocab_lines:
        t = vl.split()
        # [word, score, prc]
        vocab_score[t[0]] = map(float, t[1:])
    
    # gulp all the tags
    vocab_lines = open(tag_file, 'rt').read().split("\n")
    vocab_lines = filter(len, vocab_lines)
    img_tag = {}
    for vl in vocab_lines:
        t = vl.split()        
        #print t
        img_tag[t[0]] = t[1]
    print "read %d tags, %d images" % ( len(vocab_score), len(img_tag) ) 
    
    id_list = img_tag.keys()
    if opts.num_output<0:
        random.shuffle(id_list)
        num_output = - opts.num_output
        id_select = id_list[:num_output*10]
    elif opts.num_output>1e5:
        id_select = [str(opts.num_output)]
        num_output = 1
    else:
        num_output = opts.num_output
        id_select = id_list[:num_output*10]
    
    
    icnt = 0
    
    api_keys = open(FLICKR_KEY_FILE, 'r').read().split()
    api_keys = map(lambda s: s.strip(), api_keys)
    
    conn = sqlite3.connect(db_dict)
    conn.text_factory = str
    cursor = conn.cursor()
    
    for cur_id in id_select:
        ww = img_tag[cur_id].split(",")
        vv = map(lambda s:norm_tag(s, cursor, addl_vocab), ww)
        vv = filter(lambda s: len(s), vv)
        #find the vscore of vv
        vs = map(lambda v: vocab_score[v], vv)
        
        # get flickr picture url
        #http://farm{farm-id}.staticflickr.com/{server-id}/{id}_{secret}.jpg
        if 1:
            cur_key = api_keys [random.randint(0, len(api_keys)-1)]
            jinfo = cache_flickr_info(cur_id, cur_key, rootdir="")
            p = jinfo['photo']
            imgurl = 'http://farm%s.staticflickr.com/%s/%s_%s.jpg' % (p["farm"], p["server"], p['id'], p['secret'])
        else:
            imgurl = ""
        #print zip(vv, vs)
        if len(vv) > 5:
            icnt += 1
            # print results
            print "\nimg: %s" % (imgurl if imgurl else cur_id)
            vtup = sorted(map(lambda s,t: (s, t[0], t[1]), vv, vs), key=itemgetter(2), reverse=True)
            outstr = ""
            for i, t in enumerate(vtup):
                outstr += "%s (%0.3f,%2.1f%%)\t"%(t[0],t[1],100*t[2])
                if (i+1)%3==0:
                    outstr += "\n" 
            print outstr
            """
            print "visual tags: " + ", ".join( map(lambda v, s: "%s (%0.3f)"%(v,s[0]) if s[1]>.9 else "", vv, vs ) )
            print "other      : " + ", ".join( map(lambda v, s: "%s (%0.3f)"%(v,s[0]) if s[1]<=.9 and s[1]>=.6 else "", vv, vs ) )
            print "non-visual : " + ", ".join( map(lambda v, s: "%s (%0.3f)"%(v,s[0]) if s[1]<.6 else "", vv, vs ) )
            """
            print ""
        else:
            pass
        
        if icnt >= num_output:
            break
        
    conn.close()
    
    
if __name__ == '__main__':
    argv = sys.argv 
    if '--rank_tags' in argv:
        argv.remove('--rank_tags')
        rank_tags(argv)
    else:
        print "no function name give, quit"
        pass      
    