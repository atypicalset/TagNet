import sys,os
import sqlite3
import urllib2
import pickle
import codecs
import re
from pprint import pprint 
from random import random
from operator import itemgetter
from datetime import datetime
from optparse import OptionParser
from count_bigram_flickr import norm_tag #read_unigram, read_bigram(src_file, bg_dict)
#from compare_cooc import read_bigram_list
#import networkx as nx
import numpy as np
from scipy.optimize import fmin_slsqp
from nltk.corpus import wordnet as wn

def wn_get_hyponym(wnid, full=0):
    """ 
        input wnid, get its children (direct descendent)
        http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=[wnid]&full=1 
    """
    wn_struct_api = 'http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=%s&full=%d'
    ustr = urllib2.urlopen(wn_struct_api % (wnid, full)).read()
    wn_list = map(lambda s: s.strip().strip('-'), ustr.split())
    wn_list.remove(wnid)
    
    return wn_list


def get_descendant_words(self_syn, self_depth, db_cursor, addl_vocab):
    psyn = self_syn.hyponyms()
    if not psyn:
        return {}
    
    vlist = []
    for p in psyn:
        plemma = map(lambda s: s.name, p.lemmas)
        wlist = map(lambda s: s.lower().replace("_", " ").split(), plemma)
        ww = reduce(lambda u,v: u + v, wlist, [])
        vv = map(lambda s:norm_tag(s, db_cursor, addl_vocab,1), ww)
        vv = filter(len, vv)
        
        vlist += vv
    
    cur_d = self_depth + 1
    children_words = {}
    children_words[cur_d] = sorted(list(set(vlist)))
    #{'depth': self_depth-1,
    #    'words': sorted(list(set(vlist))) }
    
    for p in psyn:
        pw = get_descendant_words(p, cur_d, db_cursor, addl_vocab)
        # aggregate outcome by depth
        if pw:
            for d in pw.iterkeys():
                if d in children_words :
                    children_words[d] = sorted(list( set(children_words[d]) | set(pw[d]) ))
                else:
                    children_words[d] = pw[d]
    
    return children_words

def get_ancestor_words(self_syn, self_depth, db_cursor, addl_vocab):
    psyn = self_syn.hypernyms()
    if not psyn:
        return {}
    
    vlist = []
    for p in psyn:
        plemma = map(lambda s: s.name, p.lemmas)
        wlist = map(lambda s: s.lower().replace("_", " ").split(), plemma)
        ww = reduce(lambda u,v: u + v, wlist, [])
        vv = map(lambda s:norm_tag(s, db_cursor, addl_vocab,1), ww)
        vv = filter(len, vv)
        
        vlist += vv
    
    cur_d = self_depth-1
    parent_words = {}
    parent_words[cur_d] = sorted(list(set(vlist)))
    #{'depth': self_depth-1,
    #    'words': sorted(list(set(vlist))) }
    
    for p in psyn:
        pw = get_ancestor_words(p, cur_d, db_cursor, addl_vocab)
        # aggregate outcome by depth
        if pw:
            try:
                for d in pw.iterkeys():
                    if d in parent_words : # merge two dicts
                        parent_words[d] = sorted(list( set(parent_words[d]) | set(pw[d]) ))
                    else:
                        parent_words[d] = pw[d]
            except:
                print pw
                raise
    
    return parent_words

def compile_synset_wordlist(w, cw, wn_cnt=None, db_wn=None, db_dict=None, addl_vocab=None, tag_count=None):
    dbcn = sqlite3.connect(db_dict) # dictionary db
    dbcsr = dbcn.cursor()
    
    conn = sqlite3.connect(db_wn) # wnet tag info db
    cursor = conn.cursor()
    
    # ['dog', 'domestic dog', 'Canis familiaris']
    cursor.execute("SELECT words FROM wordnet WHERE wnid=?", (w,))
    wlist = map(lambda s: s.strip(), cursor.fetchone()[0].split(","))
    
    synset_exist = True
    syn_list = map(lambda s: wn.synsets(s.lower().replace(" ", "_"), pos="n"), wlist)
    self_syn = reduce(lambda u,v: v if u==-1 else list(set(u) & set(v)), syn_list, -1)
    
    if len(self_syn) > 1:
        print "%d synsets found for %s, take the 1st one: %s" % (len(self_syn), w, str(self_syn))
        self_syn = self_syn[0]
    elif len(self_syn) == 1:
        self_syn = self_syn[0]
    else: #len=0
        print "ERROR: no synset found for %s" % (w)
        synset_exist = False
        #return {'self': {'depth':None, 'words':[]} }, wlist
    
    if synset_exist:
        nltk_id = re.findall("Synset\('(.*)'\)", str(self_syn) )
        if not nltk_id:
            print "ERROR: synset parse empty for %s, %s" % (w, str(self_syn))
            synset_exist = False
    
    if tag_count:
        td = tag_count
    else:
        tcnt = cursor.execute("SELECT tag,count FROM wn_tag WHERE wnid=?", (w,)).fetchall()
        td = dict(tcnt)
        
    synset_info = {}
    synset_info['nltk_id'] = nltk_id
    synset_info['num_flickr'] = cw #wn_cnt[w]
    synset_info['tag_count'] = td
    children_wn = wn_get_hyponym(w, full=0)
    if wn_cnt:
        synset_info['children_wn'] = filter(lambda c: c in wn_cnt, children_wn)
    else:
        synset_info['children_wn'] = children_wn
    
    if not synset_exist:
        synset_info['self'] = {'depth': -1,
                              'words': [] }
        synset_info['ancestor'] = {}
        synset_info['descendant'] = {}
        synset_info['other'] = {}
    else:
        nltk_id = nltk_id[0]
        ww = reduce(lambda u,v: u + v.split(), wlist, [])
        vv = map(lambda s:norm_tag(s, dbcsr, addl_vocab,1), ww)
        vv = list(set(filter(len, vv)))
        
        synset_info['self'] = {'depth': self_syn.min_depth(),
                                  'words': vv }
        synset_info['ancestor'] = get_ancestor_words(self_syn, 0, dbcsr, addl_vocab)
        synset_info['descendant'] = get_descendant_words(self_syn, 0, dbcsr, addl_vocab)
        all_tags = set(synset_info['tag_count'].keys()) ;
        all_hierarchy = set(synset_info['self']['words'] 
                             + reduce(lambda a,b: a+b, synset_info['ancestor'].values(), []) 
                             + reduce(lambda a,b: a+b, synset_info['descendant'].values(), []) )
        synset_info['other'] = sorted(list( all_tags - all_hierarchy ))
    
    
    dbcn.close()
    conn.close()
    
    return synset_info, wlist

def print_synset_info(synset_info, w, wlist, syn_cnt):
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processing synset %04d, %s: %s" % (tt, syn_cnt, w, str(wlist))
    
    try:
        print "\t self D=%d, words: %s" % (synset_info['self']['depth'], repr(synset_info['self']['words']) )
    except:
        print w, wlist
        print  synset_info['self']
        raise
    
    if synset_info['ancestor']:
        print "\t depth=-1, words: " + repr(synset_info['ancestor'][-1])
    if synset_info['descendant']:
        print "\t depth=+1, words: " + repr(synset_info['descendant'][1])
        
    td = synset_info['tag_count']
    po = 1.*reduce(lambda a,s: a+ td[s] if s in td else a, 
                   synset_info['other'], 0) / sum(td.values()) if sum(td.values())>0 else 0
                   
    oth_cnt = sorted(map(lambda s: (s, td[s]), synset_info['other']), key=itemgetter(1), reverse=True)
    print ("\t other %0.3f : " % po) + str( oth_cnt[:10] )
    

def min_synset_prob(synset_info):
    
    # target prob vector 
    td = synset_info['tag_count']
    TQ = sorted(td.keys())  # all tags
    ttc = sum(td.values())
    Q = map(lambda w: 1.*td[w]/ttc, TQ)  # normalized tag count
    
    jh = filter(lambda j: TQ[j] not in synset_info['other'], range(len(TQ))) # index into orig tag list
    Qh = map(lambda j: Q[j], jh)  # reduced probabilities
    Th = map(lambda j: TQ[j], jh) # reduced tag list
    nh = len(jh)
    po = 1.*reduce(lambda a,j: a+ Q[j] if j not in jh else a, range(len(Q)), 0)
                   
    # pack P and W for optimization
    lc = 0
    Ph = []
    Pdepth = []
    W = []
    idx_W = []
    if synset_info['self']['words']:
        curw = map(lambda t: random() if t in synset_info['self']['words'] else 0., Th)
        sumw = sum(curw)
        if sumw:
            Ph.append( random() ) #Ph.append( 1. )
            Pdepth.append( 0 )
            curw = map(lambda x: x/sumw, curw)
            W += [curw]
            idx_W += [filter(lambda i: W[lc][i]>0, range(nh))]
            lc += 1
        else:
            print "\t empty word list for self: %s" % str(synset_info['self']['words']) 
        
    
    if synset_info["ancestor"]:
        kd = sorted(synset_info["ancestor"].keys(), reverse=True)
        for d in kd:
            if synset_info["ancestor"][d]:
                #print "\t", d, synset_info["ancestor"][d], Th#, W[0]
                #curw = map(lambda s: 1.0 if s in synset_info["ancestor"][d] else 0.0, Th)
                curw = map(lambda s: random() if s in synset_info["ancestor"][d] else 0.0, Th)
                sumw = sum(curw)
                if sumw:
                    curw = map(lambda x: x/sumw, curw)
                    if curw in W: # already exist, will make matrix singular
                        print "\t duplicate words at d=%d" % (d)
                    else:
                        Ph.append( random() ) #Ph.append( 1. )
                        Pdepth.append( d )
                        W += [ curw ]
                        idx_W += [ filter(lambda i: W[lc][i]>0, range(nh)) ]
                        lc += 1
                else:
                    print "\t ancestor words not found at d=%d" % (d)
            else:
                print "\t empty ancestor list at d=%d" % (d)
                
    if synset_info["descendant"]:
        kd = sorted(synset_info["descendant"].keys(), reverse=False)
        for d in kd:
            if synset_info["descendant"][d]:
                #curw = map(lambda s: 1. if s in synset_info["descendant"][d] else 0., Th)
                curw = map(lambda s: random() if s in synset_info["descendant"][d] else 0., Th)
                sumw = sum(curw)
                if sumw:
                    curw = map(lambda x: x/sumw, curw)
                    if curw in W: # already exist, will make matrix singular
                        print "\t duplicate words at d=%d" % (d)
                    else:
                        Ph.append( random() ) #Ph.append( 1. )
                        Pdepth.append( d )
                        W += [ curw ]
                        idx_W += [ filter(lambda i: W[lc][i]>0, range(nh)) ]
                        lc += 1
                else:
                    print "\t descendant words not found at d=%d" % (d)
            else:
                print "\t empty descendant list at d=%d" % (d)
    
    if not Pdepth:
        return 
    
    print "\t %d levels total from %d to %d" % (lc, min(Pdepth), max(Pdepth))
    L = lc
    tmp = sum(Ph)
    Ph = map(lambda x: (1-po)*x/tmp, Ph)
    
    # pack variables and bounds for optimization
    x0 = Ph + reduce(lambda u,v: u+v, W, [])
    wbound = []
    for i in range(L):
        for j in range(len(jh)):
            if j in idx_W[i]:
                if len(idx_W[i]) == 1:
                    wbound.append( (1., 1.) ) # only entry in W, must be 1
                else:
                    wbound.append( (0., 1.) )
            else:
                wbound.append( (0., 0.) ) # vocab not in this level
    
    xbound = [(0.,1.)] * len(Ph)  + wbound
    
    xres = fmin_slsqp(prob_obj, x0, args=(Qh, L, idx_W), #fprime=prob_obj_deriv,
                      f_eqcons = prob_eqcons, #fprime_eqcons=prob_eqcons_deriv,
                      bounds = xbound, full_output=False, acc=1e-8)
    
    # unpack P and W from optimization outcomes
    synset_info['pL'] = xres[:L]
    synset_info['pW'] = {}
    J = len(xres[L:])/L
    wtmp = np.resize(xres[L:], (L, J))
    
    #print "\t", synset_info['pL'] * np.asmatrix(wtmp)
    #print "\t", wtmp
    print "\t", "\t".join( map(lambda t: t[0]+":%0.4f"%t[1], zip(Th, Qh)) )
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s pL and W: " % tt
    print "\t", zip(Pdepth, map(lambda x: "%0.4f"%x, synset_info['pL']) )
    for i in range(L):
        synset_info['pW'][Pdepth[i]] = {}
        for j in range(J):
            if j in idx_W[i]:
                #print Pdepth[i], Th[j]
                synset_info['pW'][Pdepth[i]][Th[j]] = wtmp[i][j]
        
        print "\t", synset_info['pW'][Pdepth[i]]


def prob_obj(x, q, L, idw):
    J = len(x[L:])/L
    pl = np.array(x[:L])
    W = np.resize(x[L:], (L, J))
    err = 0
    for j in range(J):
        curp = 0
        for i in range(L):
            if j in idw[i]:
                curp += pl[i]*W[i][j]
        err += .5*(q[j] - curp)**2
        
    return err 

def prob_obj_deriv(x, q, L, idw):
    J = len(x[L:])/L
    pl = np.array(x[:L])
    W = np.resize(x[L:], (L, J))
    
    pd = np.zeros(pl.shape)
    Wd = np.zeros(W.shape)
    
    for j in range(J):
        
        for i in range(L):
            if j in idw[i]:
                cure = np.inner(pl, W[:,j]) - q[j]
                Wd[i,j] = pl[i] * cure
                #else Wd[i,j] = 0
                pd[i] += cure*W[i,j]
    
    deriv = np.concatenate( (pd, Wd.flatten() ) )
    
    return deriv
    
def prob_eqcons(x, q, L, idw):
    J = len(x[L:])/L
    pl = np.array(x[:L])
    W = np.resize(x[L:], (L, J))
    y = [0.]
    y[0] = sum(pl) - sum(q)
    for i in range(L):
        sum_wi = 0
        for j in range(J):
            if j in idw[i]:
                sum_wi += W[i,j]
            #else:
            #    y.append( W[i,j] )
        y.append( sum_wi - 1 )
    
    y = np.array(y)
    return y

def prob_eqcons_deriv(x, q, L, idw):
    J = len(x[L:])/L
    #pl = np.array(x[:L])
    #W = np.resize(x[L:], (L, J))

    dy = np.zeros((L+1, len(x)))
    
    for i in range(L):
        dy[0,i] = 1 #np.ones(L)
        for j in range(J):
            if j in idw[i]:
                dy[i+1, L+j] = 1
            #else:   
    
    return dy 


def map_wn_words(argv):
    if len(argv)<2:
            argv = ['-h']
    parser = OptionParser(description='build a mapping of wordnet struct and words')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db and other data')
    #parser.add_option('', '--wn_tagfreq', dest='wn_tagfreq', default="wordnet_tagsfreq.txt", help='input wordnet frequency file')
    parser.add_option("", '--cnt_db', dest='cnt_db', default="wordnet_tag.db", help='dictionary')
    parser.add_option("", '--wnet_list_file', dest='wnet_list_file', default="wnet-50.txt", 
                  help='list of wordnet ids + img cnts with at least 50 flickr imgs associated')
    parser.add_option("", '--db_dict', dest='db_dict', default="dict.db", help='dictionary')
    parser.add_option("-n", '--num_start', dest='num_start', type='int', default=0, help='start from n-th synset, for debugging')
    parser.add_option('', '--TOTAL_NUM_IMG', dest='TOTAL_NUM_IMG', type="int", default=4517527, help='# of flickr imgs / docs')
    parser.add_option("", '--addl_vocab', dest='addl_vocab', default="places_etc.txt", help='')
    
    (opts, __args) = parser.parse_args(sys.argv)
    
    db_dict = os.path.join(opts.db_dir, opts.db_dict)
    addl_vocab = open(os.path.join(opts.db_dir, opts.addl_vocab), 'rt').read().split()
    
    # load synset counts
    cnt_file = os.path.join(opts.db_dir, opts.wnet_list_file)
    tmp_lines = filter(lambda s:len(s), map(lambda s: s.strip(), open(cnt_file, "rt").read().split("\n") ) )
    wn_cnt = dict(map(lambda t: (t.split()[1], int(t.split()[0])), tmp_lines) )
    print "read %d wordnet counts" % (len(wn_cnt))
    
    db_wn = os.path.join(opts.db_dir, opts.cnt_db)
    pkl_file = os.path.join(opts.db_dir, 'synset_info.pkl')
    
    synset_info = {}
    syn_cnt = 0
    for w, cw in wn_cnt.items(): #row in cursor.execute("SELECT wnid,words FROM wordnet"):
        #if random() > 0.3:
        #    continue
        syn_cnt += 1
        if syn_cnt < opts.num_start:            
            continue
        
        synset_info[w],wlist = compile_synset_wordlist(w, cw, wn_cnt, db_wn, db_dict, addl_vocab)
        
        print_synset_info(synset_info[w], w, wlist, syn_cnt)
        
        min_synset_prob(synset_info[w])
        
        print "\n"
        
        if syn_cnt % 200 ==0:
            pickle.dump( synset_info, open(pkl_file, "wb" ) )
        
        

def norm_tag_file(argv):
    """
        clean tags for one single file
    """
    if len(argv)<2:
            argv = ['-h']
    parser = OptionParser(description='rank tags of a single image + compose sentences')
    parser.add_option('-i', '--in_file', dest='in_file', default="", help='input file')
    parser.add_option('-o', '--out_dir', dest='out_dir', default="db", help='ouptut format (db or pkl)')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db and other data')
    parser.add_option("", '--db_dict', dest='db_dict', default="dict.db", help='dictionary')
    parser.add_option("", '--addl_vocab', dest='addl_vocab', default="places_etc.txt", help='')
    
    (opts, __args) = parser.parse_args(sys.argv)
    db_dict = os.path.join(opts.db_dir, opts.db_dict)
    addl_vocab = open(os.path.join(opts.db_dir, opts.addl_vocab), 'rt').read().split()
    
    out_fh = open(os.path.join(opts.out_dir, os.path.split(opts.in_file)[1]), 'wt')
    conn = sqlite3.connect(db_dict)
    cursor = conn.cursor()
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processing %s ..." % (tt, opts.in_file)
    cnt = 0
    ecnt = 0
    tcnt = 0
    for cl in codecs.open(opts.in_file, encoding='utf-8'):
        imid = cl.split()[0]
        w = cl.split()[1]
        ww = w.split(",")
        vv = map(lambda s:norm_tag(s, cursor, addl_vocab), ww)
        vv = filter(lambda s: len(s), vv)
        if vv:
            #tag_dict[imid] = vv
            out_fh.write("%s\t%s\n" % (imid, ",".join(vv)))
            tcnt += len(vv)
        else:
            ecnt += 1
        
        cnt += 1
        if cnt % 5000 == 0:
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s %d img-id processed, %d tags, %d empty" % (tt, cnt, tcnt, ecnt)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s %d img-id processed, %d tags, %d empty. Done.\n" % (tt, cnt, tcnt, ecnt)
    conn.close()



if __name__ == '__main__':
    argv = sys.argv 
    if "--norm_tag_file" in argv:
        argv.remove("--norm_tag_file")
        norm_tag_file(argv)
    elif "--eval_tag_prob" in argv:
        argv.remove("--eval_tag_prob")
        #eval_tag_prob(argv)
    else:
        map_wn_words(argv)
        pass  

"""        
    ########## ########## ########## ##########
    UNUSED FUNCTIONS
"""
def eval_tag_prob(argv):
    if len(argv)<2:
            argv = ['-h']
    parser = OptionParser(description='test the significance of tags in wordnet synset (compared to its descendents)')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db and other data')
    parser.add_option('', '--wn_tagfreq', dest='wn_tagfreq', default="wordnet_tagsfreq.txt", help='input wordnet frequency file')
    parser.add_option('-q', '--qry_root', dest='qry_root', default="", help='root wnid for querying')
    parser.add_option("-p", '--p_value', dest='p_value', type="float", default=1e-6, help='dictionary')
    parser.add_option("", '--cnt_db', dest='cnt_db', default="wordnet_tag.db", help='dictionary')    
    parser.add_option("", '--wnet_list_file', dest='wnet_list_file', default="wnet-50.txt", 
                  help='list of wordnet ids + img cnts with at least 50 flickr imgs associated')
    parser.add_option("", "--flickr_unigram", dest="flickr_unigram", default="flickr.unigram.txt", 
                      help="list of flickr tag frequencies format: (#\t tag)")
    parser.add_option('-n', '--TOTAL_NUM_IMG', dest='TOTAL_NUM_IMG', type="int", default=4517527, help='root wnid for querying')
    
    (opts, __args) = parser.parse_args(sys.argv)
     
    db_file = os.path.join(opts.db_dir, opts.db_out)
    ug_file = os.path.join(opts.db_dir, opts.flickr_unigram)
    tmp_lines = filter(lambda s:len(s), map(lambda s: s.strip(), open(ug_file, "rt").read().split("\n") ) )
    tag_cnt = map(lambda t: (t.split()[1], int(t.split()[0])), tmp_lines) 
    tag_prior = dict( map(lambda t: (t[0], float(t[1])/opts.TOTAL_NUM_IMG ), tag_cnt) )
    print "read %d tag prior" % (len(tag_prior))
    
    # load synset counts
    cnt_file = os.path.join(opts.db_dir, opts.wnet_list_file)
    tmp_lines = filter(lambda s:len(s), map(lambda s: s.strip(), open(cnt_file, "rt").read().split("\n") ) )
    wn_cnt = dict(map(lambda t: (t.split()[1], int(t.split()[0])), tmp_lines) )
    print "read %d wordnet counts" % (len(wn_cnt))
    
    wn_list = wn_get_hyponym(opts.qry_root, full=1)
    print "direct query: %d children nodes descending from %s" % (len(wn_list), opts.qry_root)
    
    pdict,cdict = build_wn_from_root(opts.qry_root)
    print "recursive query: %d, %d nodes descending from %s" % (len(pdict), len(cdict), opts.qry_root)
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    wn_sig_tags = dict.fromkeys(wn_list)
    wn_sig_words = dict.fromkeys(wn_list)
    wn_sig_children = dict.fromkeys(wn_list)
    wn_sig_descendent = dict.fromkeys(wn_list)
    
    for wnid in wn_list:
        # find words
        cursor.execute("SELECT (words) FROM wordnet WHERE wnid=?", wnid)
        wlist = cursor.fetchone()[0].split(",")
        
        # find out all significant tags
        wn_sig_tags[wnid] = []
        for tag, tcnt, tprc in cursor.execute("SELECT (tag, count, percentage) FROM wn_tag WHERE wnid=?", wnid):
            # >>> from scipy.stats import binom
            # >>> prb = binom.cdf(3, 50, 1e-4)
            imcnt = round(tcnt/tprc)
            prb = 1 #- binom.cdf(tcnt, imcnt, tag_prior[tag])
            if prb >= opts.p_value:
                wn_sig_tags[wnid] += [tag]
            
    conn.close()          
    
def build_wn_from_root(qry, parent_dict={}, children_dict={}):
        
    wn_list = wn_get_hyponym(qry)
    
    if not qry in children_dict:
        children_dict[qry] = wn_list
    
    if not qry in parent_dict:
        parent_dict[qry] = []
        
    for w in wn_list:
        if not w in parent_dict:
            parent_dict[w] = qry
        else:
            print "error %s already exist for %s" % (w, qry)
            print parent_dict
                                          
        build_wn_from_root(w, parent_dict, children_dict)
    
    return parent_dict, children_dict

def collect_tags(argv):
    """
        reads a txt file of cleaned tags, insert into db or save as dict
        seems redundant work , using "wordnet_tagsfreq.txt" is enough
    """
    if len(argv)<2:
            argv = ['-h']
    parser = OptionParser(description='rank tags of a single image + compose sentences')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db and other data')
    parser.add_option('-o', '--out_format', dest='out_format', default="db", help='ouptut format (db or pkl)')
    #parser.add_option("", '--db_dict', dest='db_dict', default="dict.db", help='dictionary')
    parser.add_option("", '--tag_file', dest="tag_file", default="all_tags.cache", help="")
    #parser.add_option("", '--id_file' , dest="id_file" , default="all_id.cache", help="")
    #parser.add_option("", '--addl_vocab', dest='addl_vocab', default="places_etc.txt", help='')
    
    (opts, __args) = parser.parse_args(sys.argv)
    
    id_file = os.path.join(opts.db_dir, os.path.splitext(opts.tag_file)[0]+".id")
    all_id = open(id_file, 'rt').read().split()
    tag_file = os.path.join(opts.db_dir, opts.tag_file)
    
    if opts.out_format=="pkl":
        out_pkl = os.path.join(opts.db_dir, os.path.splitext(opts.tag_file)[0]+".pkl")
        out_cn = None
    else:
        out_db = os.path.join(opts.db_dir, os.path.splitext(opts.tag_file)[0]+".db")
        out_cn = sqlite3.connect(out_db)
        out_cr = out_cn.cursor()
        out_cr.execute('CREATE TABLE IF NOT EXISTS "flickr_tags" (imgid TEXT, tag TEXT)')
        out_cr.execute("CREATE INDEX IF NOT EXISTS imidx ON flickr_tags(imgid)")
        out_cr.execute("CREATE INDEX IF NOT EXISTS tagidx1 ON flickr_tags(tag)")
        out_cr.execute("DELETE FROM flickr_tags")
        out_cn.commit()
        
    tag_dict = dict.fromkeys(all_id)
    
    cnt = 0
    tcnt= 0
    num_im = len(all_id)
    for cl in codecs.open(tag_file, encoding='utf-8'):
        imid = cl.split()[0]
        w = cl.split()[1]
        ww = w.split(",")
        nw = len(ww)
        
        #vv = map(lambda s:norm_tag(s, cursor, addl_vocab), ww)
        #vv = filter(lambda s: len(s), vv)
        if opts.out_format=="pkl":
            tag_dict[imid] = ww
        else:
            out_cn.executemany("INSERT INTO flickr_tags (imgid, tag) VALUES (?,?)", zip([imid]*nw, ww) )
            
        cnt += 1
        tcnt += nw
        if cnt % 10000 == 0:
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s %d / %d imgs processed, %d tags" % (tt, cnt, num_im, tcnt)
            if out_cn:
                out_cn.commit()
            
    if opts.out_format=="db":
        out_cn.close()
    else:
        pickle.dump(tag_dict, open(out_pkl, "wb"))    