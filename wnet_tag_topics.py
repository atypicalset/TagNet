

import sys,os
import sqlite3
#import urllib2
#import pickle
import codecs
#import re
#from pprint import pprint 
#from random import random
#from operator import itemgetter
#from itertools import groupby
from glob import glob
from scipy import io
from datetime import datetime
from optparse import OptionParser
from count_bigram_flickr import norm_tag #read_unigram, read_bigram(src_file, bg_dict)
from tag_popular_test import compile_synset_wordlist, print_synset_info
#from mlabwrap import mlab
from compare_cooc import parse_lines_fromfile
from count_bigram_flickr import accumulate_bg, sort_bg

import conceptnet.models as cm

import numpy as np
import math

from nltk.corpus import wordnet as wn

en = cm.Language.get('en')

def binary_mutual_info(N, x, y, xy):
    # given counts of total, and #x, #y, #xy, computer I(X;Y) =
    # \sum_{x,y} p(x,y) log2 [p(x,y) / p(x)p(y)]
    px = [1.*(N-x)/N, 1.*x/N]
    py = [1.*(N-y)/N, 1.*y/N]
    pxy = [N+xy-x-y, y-xy, x-xy, xy]
    pxy = map(lambda a: 1.*a/N, pxy)
    Ixy = 0
    for i in range(2):
        for j in range(2):
            ij = i*2+j
            if not pxy[ij] or not px[i] or not py[j]:
                continue
            if px[i]<0 or py[j]<0 or pxy[ij]<0:
                raise Exception("ValueError: probability less than 0!")
            
            Ixy += pxy[ij] * math.log( pxy[ij]/(px[i]*py[j]) , 2) 
    
    #print pxy, px, py
    return Ixy
    
def read_tag_cache(in_file):
    cache_dict = {}
    for cl in codecs.open(in_file, encoding='utf-8', mode="r"):
        tmp = cl.strip().split()
        if not tmp:
                continue
        cache_dict[tmp[0]] = tmp[1]
    return cache_dict

def cache_wnet_tag(opts):
    opts, _db_dict, _addl_vocab, _db_wn = options_get_wnet_tag(argv)
    file_list = glob(os.path.join(opts.data_home, opts.wnet_list_dir, 'n*[0-9].txt'))
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s start processing %d synsets " % (tt, len(file_list))
    scnt = 0
    
    lines = parse_lines_fromfile(os.path.join(opts.data_home, opts.db_dir, opts.usr_file) )
    img_usr = dict(lines)
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s read %d usr ids from %s" % (tt, len(img_usr), opts.usr_file)
    
    for f in file_list:
        scnt += 1
        wn = os.path.splitext( os.path.split(f)[1] )[0] # synset id
        wtag_file = os.path.join(opts.data_home, opts.wnet_list_dir, wn+'.tags.txt')
        
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print '\n%s synset # %d "%s"' % (tt, scnt, wn)
        get_wnet_tags(wn, wtag_file, opts, img_usr, force_reload=True)
    
    
def get_wnet_tags(wn, wtag_file, opts, img_usr=None, force_reload=False):
    """
        collect each image id, usr id, and tags for the input synset
    """
    
    if os.path.isfile(wtag_file) and not force_reload:
        imgid_list = []
        usr_list= []
        tag_dict = {}
        for cl in codecs.open(wtag_file, encoding='utf-8', mode="r"):
            tt = cl.strip().split()
            imgid_list.append(tt[0])
            usr_list.append(tt[1])
            tag_dict[tt[0]] = tt[2].split(",")
    else:
        wn_file = os.path.join(opts.data_home, opts.wnet_list_dir, wn+'.txt')
        imgid_list = []
        for cl in open(wn_file, "rt"):
            tmp = cl.strip().split()
            if not tmp:
                continue
            imgid_list.append(tmp[1])
        imgid_list.sort()
        numim = len(imgid_list)
        
        if not img_usr:
            lines = parse_lines_fromfile(os.path.join(opts.data_home, opts.db_dir, opts.usr_file) )
            img_usr = dict(lines)
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s read %d usr ids from %s" % (tt, len(img_usr), opts.usr_file)
        
        usr_list = map(lambda s: img_usr[s] if s in img_usr else "unk", imgid_list)
        
        cur_cache_id = []
        empty_cnt = 0
        del_id = []
        tag_dict = {}
        fh = codecs.open(wtag_file, encoding='utf-8', mode="w")
        for (imgid, uid, ii) in zip(imgid_list, usr_list, range(len(usr_list))):
            
            if not cur_cache_id==imgid[:2]:
                # read new cache file
                cur_cache_id = imgid[:2]
                cur_cache_file = os.path.join(opts.data_home, opts.tag_cache_dir, cur_cache_id+".cache")
                cache_dict = read_tag_cache(cur_cache_file)
                #print "\t read %s imgs from %s" % (len(cache_dict), cur_cache_id+".cache")
            else:
                pass
            
            if imgid in cache_dict:
                ww = cache_dict[imgid].split(",")
            else: # img has no tag
                ww = []
                empty_cnt += 1
            
            if ww:
                tag_dict[imgid] = ww
                fh.write("%s\t%s\t%s\n" % (imgid, uid, ",".join(ww)))
            else:
                del_id.append(ii)
                
        fh.close()
        del_id.sort(reverse=True) #pop bigger indexes first, trouble otherwise
        for ii in del_id:
            imgid_list.pop(ii)
            usr_list.pop(ii)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print '%s wnet "%s" - %d imgs, %d with tags, %d unique users' % (tt, wn, numim, len(imgid_list), len(set(usr_list)))
        
    return imgid_list, usr_list, tag_dict
            

def get_wnet_usr_tag(argv):
    # parser = OptionParser(description='compile tags for all imgs in a wnet synset')
    opts, db_dict, addl_vocab, db_wn = options_get_wnet_tag(argv)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s start processing '%s' " % (tt, opts.in_wnet_list)
    
    
    dbcn = sqlite3.connect(db_dict) # dictionary db
    dbcsr = dbcn.cursor()
    
    if os.path.isfile(opts.in_wnet_list):
        wnet_list = open(opts.in_wnet_list, 'rt').read().split()
    else:
        wnet_list = opts.in_wnet_list.split(",")
        
    for wn in wnet_list:
        
        wtag_file = os.path.join(opts.data_home, opts.wnet_list_dir, wn+'.tags.txt')
        imgid_list, usr_list, tag_dict = get_wnet_tags(wn, wtag_file, opts, False)
        
        usr_tag = {}
        tag_cnt = {}
        empty_cnt = 0
        
        #for u, utuple in groupby(zip(usr_list, imgid_list), lambda x: x[0]):
        for (imid, u) in zip(imgid_list, usr_list):
            if u not in usr_tag:
                usr_tag[u] = {}
                    
            vv = map(lambda s:norm_tag(s, dbcsr, addl_vocab,1), tag_dict[imid])
            vv = list(set(filter(len, vv)))
            for v in vv:
                if v in usr_tag[u]:
                    usr_tag[u][v] += 1
                else:
                    usr_tag[u][v] = 1
                    tag_cnt[v] = tag_cnt.get(v, 0)+1
            
            #for v in list(set(usr_tag[u])):
                
        
        tag_list = filter(lambda t: tag_cnt[t]>1, tag_cnt.keys())
        tag_list.sort()
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s obtained %d usrs, %d tags non-trival, %d empty imgs " % (tt, len(usr_tag), len(tag_list), empty_cnt)
        
        ulist = []
        tcnt_mat = np.zeros( (len(usr_list), len(tag_list)))
        icnt = 0
        #out_dat_fh = open(os.path.join(opts.data_home, opts.wnet_out_dir, wn+".dat"), "wt")
        for u in list(set(usr_list)):
            jj = -1
            outstr = ""
            if usr_tag[u]:
                for t in usr_tag[u]:
                    if t in tag_list:
                        jj  = tag_list.index(t)
                        tcnt_mat[icnt, jj] += 1
                        outstr += "%d:%d " % (jj+1, usr_tag[u][t])
            else:
                pass # empty list of tags, skip
           
            if outstr:
                #out_dat_fh.write( outstr + "\n" )
                icnt += 1
                ulist.append(u)
        
        tcnt = tcnt_mat[:icnt, :]
        
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print '%s wnet "%s": %d unique tags by %d users\n' % (tt, wn, len(tag_list), len(ulist))
        
        # get and print synset info
        td = dict([ (k, tag_cnt[k]) for k in tag_list ])
        syn_info, wlist = compile_synset_wordlist(wn, len(ulist), None, db_wn, db_dict, addl_vocab, td)
        hi_words = [syn_info['self']['words']] + syn_info['ancestor'].values() + syn_info['descendant'].values()
        hi_depth = [0 ] + syn_info['ancestor'].keys() + syn_info['descendant'].keys()
        other_words = list(set(tag_list) - set(reduce(lambda a,b: a+b, hi_words, [])) )
        print_synset_info(syn_info, wn, wlist, len(ulist))
        
        wn_out_mat = os.path.join(opts.data_home, opts.wnet_out_dir, wn+".mat")
        tcnt = tcnt_mat[:icnt, :]
        #mlab.save(wn_out_mat, 'img_list', 'tag_list', 'tcnt_mat')
        data = {'usr_list': ulist, 'tag_list': tag_list, 'tcnt':tcnt, 
                'other_words':other_words, 'hier_words':hi_words, 'hier_depth':hi_depth}
        io.savemat(wn_out_mat, data)
        
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print '%s saved to %s \n\n' % (tt, wn_out_mat)
        
        print "\n tag occurrence:"
        hi_words = reduce(lambda a,b: a+b, hi_words, [])
        lcnt = 0
        bg_dict = {}
        wn_tagf = open(os.path.join(opts.data_home, opts.wnet_out_dir, wn+".txt"), "wt")
        for u in usr_tag:
            vv = filter(lambda v: v in tag_list, usr_tag[u].keys())
            if vv:
                wn_tagf.write(u+"\t"+ ",".join(vv) + "\n")
                accumulate_bg(vv, bg_dict, None, None, addl_vocab=[])
                lcnt += 1
                if lcnt<20: print u+"\t"+ ",".join(vv) 
        wn_tagf.close()
        
        print "\nbigrams#\tMI \ttype\ttag1,tag2\trelations"
        lcnt = 0
        bg_tuples = sort_bg(bg_dict)
        wn_bgf = open(os.path.join(opts.data_home, opts.wnet_out_dir, wn+".bigram.txt"), "wt")
        for u, v, c in bg_tuples:
            assr = cm.Assertion.objects.filter(concept1__text=u, concept2__text=v,language=en)
            atxt = map(lambda a: str(a).strip("[]"), assr)
            atxt += map(lambda a: str(a).strip("[]"), 
                        cm.Assertion.objects.filter(concept1__text=v, concept2__text=u,language=en))
            mi = binary_mutual_info(len(usr_tag), tag_cnt[u], tag_cnt[v], c)
            if u in other_words and v in other_words:
                btype = "OO"
            elif u in hi_words and v in hi_words:
                btype = "HH"
            else:
                btype = "HO"
                
            outstr = "%5d\t%0.3f\t%s\t%s,%s\t%s" % (c, mi, btype, u, v, ";".join(atxt))
            wn_bgf.write(outstr +"\n")
            lcnt += 1
            if lcnt<50: print outstr
        wn_bgf.close()
        

def options_get_wnet_tag(argv):
    if len(argv)<2:
            argv = ['-h']
    parser = OptionParser(description='compile tags for all imgs in a wnet synset')
    parser.add_option('-d', '--data_home', dest='data_home', default="", help='parent dir of db, wnet-list, cache, etc')
    parser.add_option('-i', '--in_wnet_list', dest='in_wnet_list', default="", help='input file containing a list of wnet to process')
    parser.add_option('-c', '--tag_cache_dir', dest='tag_cache_dir', default="bigram", help='subdir storing tag cache by id')
    parser.add_option('-u', '--usr_file', dest='usr_file', default="flickr_usr_5M.txt.gz", help='file for usr-img mapping')
    parser.add_option('-w', '--wnet_list_dir', dest='wnet_list_dir', default="wnet", help='subdir storing wnet info')
    parser.add_option('-o', '--wnet_out_dir', dest='wnet_out_dir', default="wnet-tags", help='dir for wnet tag output')
    parser.add_option('', '--db_dir', dest='db_dir', default="db2", help='subdir storing various db')
    
    parser.add_option("", '--db_wn', dest='db_wn', default="wordnet_tag.db", help='db about wordnet words and tags')
    parser.add_option("", '--db_dict', dest='db_dict', default="dict.db", help='dictionary')
    parser.add_option("", '--addl_vocab', dest='addl_vocab', default="places_etc_v2.txt", help='')
    
    (opts, __args) = parser.parse_args(sys.argv)
    db_dict = os.path.join(opts.data_home, opts.db_dir, opts.db_dict)
    addl_vocab = open(os.path.join(opts.data_home, opts.db_dir, opts.addl_vocab), 'rt').read().split()
    db_wn = os.path.join(opts.data_home, opts.db_dir, opts.db_wn)
    
    return opts, db_dict, addl_vocab, db_wn

if __name__ == '__main__':
    argv = sys.argv 
    if "--cache_wnet_tag" in argv:
        argv.remove("--cache_wnet_tag")
        cache_wnet_tag(argv)
    elif "--get_wnet_usr_tag" in argv:
        argv.remove("--get_wnet_usr_tag")
        get_wnet_usr_tag(argv)
    else:
        #map_wn_words(argv)
        pass  