
import sys,os
import sqlite3
import json
import divisi2
import pickle
import gzip, tarfile
import operator
from pysparse import sparse
from datetime import datetime
from itertools import izip
from optparse import OptionParser
from count_bigram_flickr import norm_tag #read_bigram(src_file, bg_dict)

def get_conceptnet_words(db_file):
    A = divisi2.network.conceptnet_matrix('en')
    wa = A.row_labels
    
    ww = filter(lambda s: s==s.split()[0], wa) # keep 1-word terms only
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()    
    vv = map(lambda s:norm_tag(s, cursor), ww)
    wd = dict(izip(ww, vv))
    conn.close()
    
    return wd
    
def compute_sim_conceptnet(word_dict, vocab):
    A = divisi2.network.conceptnet_matrix('en')
    """ subset rows of A to correspond to vocab """
    """ """
    row_labB = filter(lambda s: len(s)>0 and s in vocab, list(set(word_dict.values())) )
    ib_list = []
    ## figure out the row indexes of new_rows
    #new_id = filter(lambda i: A.row_label(i) in new_rows, range(A.shape[0]))
    #nl = A[new_id, :].named_lists() # an ugly method for converting the args
    #row_labB = word_dict.values()
    #Bp = divisi2.SparseMatrix(nrow=len(row_labB), ncol=A.shape[1], sizeHint=A.nnz)
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s B has %d rows" % (tt, len(row_labB))
            
    Bp = divisi2.SparseMatrix([], row_labels=row_labB, col_labels=A.col_labels)
    rcnt = 0
    for ia, ar in enumerate(A.row_labels):
        if ar not in word_dict:
            continue
        elif word_dict[ar] in row_labB:
            ib = row_labB.index(word_dict[ar])
            #Bp[ib, :] = Bp[ib, :] + A[ia, :]
            aa = A[ia, :].to_dict()
            rn = word_dict[ar]
            Bp = Bp + divisi2.SparseMatrix.from_named_lists(aa.values(), [rn]*len(aa), aa.keys())
            rcnt += 1
            if not ib in ib_list:
                ib_list.append(ib)
        
        if (ia+1) % 100 == 0:
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s processed %5d / %d rows, %d added" % (tt, ia+1, A.shape[0], rcnt)
    
    print " %s matrix B -- %d rows, %d populated " % (repr(Bp.shape), len(row_labB), len(ib_list))
    #B = divisi2.SparseMatrix(Bp, row_labels=row_labB, col_labels=A.col_labels)
    """ DOESN"T WORK FOR NOW, SKIP
        bd = B.data()
        ## now add in the rows of the dict items that map to one of new_rows
        for k,v in word_dict.iteritems():
            if not k==v and len(v):
                ib = B.row_index(v)
                bd[ib,:] = bd[ib,:] + A.row_named(k).data()
        B.replacedata(bd)
    """
    B = Bp.normalize_all()
    U, S, _V = B.svd(k=100)
    #sim = divisi2.reconstruct_similarity(U, S, post_normalize=False)
    
    #assert sim.row_labels == new_rows, " ERR in the row labels of the reconstruction matrix! "
    ## possible PROBLEM: empty row labels
    
    return U, S

def get_flickr_words(db_file, thresh=5):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()   
    cursor.execute("SELECT word FROM unigram WHERE freq>%d" % thresh)
    
    words = map(lambda r: r[0], cursor)
    conn.close()
    return words

def get_flickr_unigram(ug_file, vocab=[], thresh=-1):
    
    word = []
    freq = []
    for cl in open(ug_file, "rt"):
        tmp = cl.strip().split()
        w = tmp[0]
        c = int(tmp[1])
        if vocab and w not in vocab:
            continue
        if thresh>0 and c<thresh:
            continue
        word.append(w)
        freq.append(c)
    
    return dict(zip(word,freq))

def read_bigram_subset(bg_file, vocab, thresh=-1):
    bg_dict = {}
    ## each line of the file
    # count word1 word2
    line_cnt = 0 
    v_cnt = 0
    bg_cnt = 0
    for line in open(bg_file, 'rt'):
        tmp = line.strip().split()
        c = int(tmp[0])
        w1 = tmp[1]
        w2 = tmp[2] ## assuming w1<w2
        line_cnt += 1
        if w1 in vocab and w2 in vocab:
            v_cnt += 1
            if thresh<0 or c >= thresh:
                bg_cnt += 1
                if w1 not in bg_dict:
                    bg_dict[w1] = {}
                    bg_dict[w1][w2] = c
                else:
                    assert w2 not in bg_dict[w1], "bigram keys should not repeat! %s " % line
                    bg_dict[w1][w2] = c
        
        if line_cnt % 5000 == 0:
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s read %d lines, %d in vocab, %d pass threshold" % (tt, line_cnt, v_cnt, bg_cnt)
                    
    print "read %d lines, %d in vocab, %d pass thresh=%d" % (line_cnt, v_cnt, bg_cnt, thresh)
    
    return bg_dict

def get_wordnet_words(db_wn, wnid_file, db_dict, vocab):
    wn_freq = {}
    for line in open(wnid_file):
        t = line.strip().split()
        wn_freq[t[1]] = int(t[0])
    
    wn_words = {}
    
    condict = sqlite3.connect(db_dict)
    curdict = condict.cursor()
    
    conn = sqlite3.connect(db_wn)
    cursor = conn.cursor()
    wcnt = 0
    vcnt = 0
    for k in wn_freq.iterkeys():
        #stmt = ("SELECT W.wnid, W.word1, WW.word " + 
        #    "FROM wordnet as W, wordnet_word AS WW " + 
        #    "WHERE W.wnid='%s' AND W.wnid=WW.wnid" % k )
        stmt = "SELECT word FROM wordnet_word WHERE wnid='%s'" % k
        cursor.execute(stmt)
        ww = map(lambda t: t[0], cursor.fetchall())
        vv = map(lambda s:norm_tag(s.lower(), curdict), ww)
        vv = filter(lambda s: len(s) and s in vocab, vv)
        vv = list(set(vv))
        wcnt += 1
        if vv:
            wn_words[k] = vv
            vcnt += 1
        if wcnt % 1000==0:
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s %d/%d synsets processsed, %d has tags" % (tt, wcnt, len(wn_freq), vcnt)
        
    conn.close()
    condict.close()
    uniq_v = reduce(lambda v, s: v+s, wn_words.itervalues())
    uniq_v = list( set(uniq_v) )
    print " %d / %d synsets has non-empty mapping to %d words" % (len(wn_words), len(wn_freq), len(uniq_v))
    
    return wn_words

def intersect_vocab(db_dict, tag_file, db_wn, wn_list):
    cn_words = get_conceptnet_words(db_dict)
    fr_words,__ = get_flickr_unigram(tag_file, thresh=5)
    
    print "ConceptNet: %d words, %d cleaned" % (len(cn_words), len(set(cn_words.values())) )
    print "Flickr %d tags, %d in common with ConceptNet " % (len(fr_words), len(set(cn_words.values()) & set(fr_words)) )
    vocab = list( set(cn_words.values()) & set(fr_words) )
    vocab = filter(lambda s: len(s)>0, vocab)
    print " %d words in the intersected vocab" % len(vocab)
    
    # now deal with wordnet
    wn_words = get_wordnet_words(db_wn, wn_list, db_dict, vocab)
    
    return vocab, wn_words

def record_svd(argv):
    if len(argv)<2:
            argv = ['-h']
    parser = OptionParser(description='construct+compare conceptnet and flickr word similarities')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db files')
    parser.add_option("", '--db_dict', dest='db_dict', default="dict.db", help='dictionary')
    parser.add_option("", '--vocab_file', dest='vocab_file', default="vocab.txt", help='vocabulary file')
    parser.add_option("", '--out_prefix', dest='out_prefix', default="conceptnet_sim_", help='')
    (opts, __args) = parser.parse_args(sys.argv)
    
    db_dict = os.path.join(opts.db_dir, opts.db_dict)
    vocab_file = os.path.join(opts.db_dir, opts.vocab_file)
    vocab = open(vocab_file, 'r').read().split("\n")
    cn_words = get_conceptnet_words(db_dict)
    U, S = compute_sim_conceptnet(cn_words, vocab)
    
    import numpy
    out_file_u = os.path.join(opts.db_dir, opts.out_prefix+"U.txt")
    out_file_rows = os.path.join(opts.db_dir, opts.out_prefix+"rowlabel.txt")
    out_file_s = os.path.join(opts.db_dir, opts.out_prefix+"S.txt")
    numpy.savetxt(out_file_u, U.to_scipy(), fmt='%.5e', delimiter=' ')
    numpy.savetxt(out_file_s, S, fmt='%.5e', delimiter=' ')
    open(out_file_rows, "wt").write("\n".join(list(U.row_labels)) )
    
    
def make_vocab(argv):
    if len(argv)<2:
            argv = ['-h']
    parser = OptionParser(description='construct+compare conceptnet and flickr word similarities')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db files')
    parser.add_option("", '--db_dict', dest='db_dict', default="dict.db", help='dictionary')
    parser.add_option("", '--unigram_file', dest='unigram_file', default="unigram.txt", help='unigrams file %word count%')
    parser.add_option("", '--wn_list', dest='wn_list', default="wnet-50.txt", help='')
    parser.add_option("", '--db_wordnet', dest='db_wordnet', default="wordnet.db", help='')
    #parser.add_option("", '--bigram_file', dest='bigram_file', default="bigram_filtered.txt", help='')
    (opts, __args) = parser.parse_args(sys.argv)
    
    # intersect the two dictionaries first
    db_dict = os.path.join(opts.db_dir, opts.db_dict)
    ug_file = os.path.join(opts.db_dir, opts.unigram_file)
    db_wn = os.path.join(opts.db_dir, opts.db_wordnet)
    wn_list = os.path.join(opts.db_dir, opts.wn_list)
    
    vocab, wn_tags = intersect_vocab(db_dict, ug_file, db_wn, wn_list)
    
    open(os.path.join(opts.db_dir, 'vocab.txt'), "wt").write("\n".join(vocab))
    
    fo = open(os.path.join(opts.db_dir, 'wordnet_tags.txt'), "wt")
    for k, v in wn_tags.iteritems():
        fo.write("%s\t%s\n" % (k, ",".join(v)))
    fo.close()

def read_bigram_list(bg_file, vocab=[], numtype='int'):
    lines = open(bg_file, "rt").read().split("\n")
    word1 = []
    word2 = []
    freq = []
    
    for cl in lines:
        if not cl.strip():
            continue
        tmp = cl.split()
        if vocab and (tmp[1] not in vocab or tmp[2] not in vocab):
            continue
        if numtype=='int':
            freq.append(int(tmp[0]))
        elif numtype=='float':
            freq.append(float(tmp[0]))
        else:
            raise Exception("ERR", "ERR unknown numtype %s! " % numtype)
        word1.append(tmp[1])
        word2.append(tmp[2])
    
    print "done loading %d bigram pairs from %d lines" % (len(word1), len(lines))
    return word1, word2, freq

def make_bigram(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='construct+compare conceptnet and flickr word similarities')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db files')
    #parser.add_option("", '--db_dict', dest='db_dict', default="dict.db", help='dictionary')
    parser.add_option("", '--unigram_file', dest='unigram_file', default="unigram.txt", help='unigrams file %word count%')
    parser.add_option("", '--bigram_file', dest='bigram_file', default="bigram_filtered.txt", help='')
    parser.add_option("", '--vocab_file', dest='vocab_file', default="vocab.txt", help='')
    (opts, __args) = parser.parse_args(sys.argv)
    
    # build stuff from flickr bigrams
    ug_file = os.path.join(opts.db_dir, opts.unigram_file)
    bg_file = os.path.join(opts.db_dir,opts.bigram_file)
    vocab = open(os.path.join(opts.db_dir, opts.vocab_file), "rt").read().split("\n")
    
    ug_dict = get_flickr_unigram(ug_file, vocab)
    word1, word2, freq = read_bigram_list(bg_file, vocab)
    
    BG = divisi2.SparseMatrix.square_from_named_lists(freq, word1, word2 )
    BG = BG + BG.T  # make symmertic
    print " done constructing bigram matrix of shape %s" % repr(BG.shape)
    # normalize
    for idx in range(BG.shape[0]):
        rn = BG.row_label(idx)
        BG[idx, :] = BG[idx, :]/ug_dict[rn]
    
    pickle.dump((BG,ug_dict), open(os.path.join(opts.db_dir, "bigram.pkl"), "wb"))
    return BG

def parse_lines_fromfile(in_file_name):
    
    if os.path.splitext(in_file_name)[1]==".gz":
        fh = gzip.open(in_file_name, "rb")
    else:
        fh = open(in_file_name, "r")
        
    lines = filter(len, fh.read().split("\n") )     # skip empty line if any
    lines = [cl.strip().split() for cl in lines]              
    return lines

def make_wn_tag(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='construct WN-tag tables')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing data files')
    parser.add_option('-w', '--wnet_dir', dest='wnet_dir', 
                      default="/home/users/xlx/vault-xlx/imgnet-flickr/wnet", help='dir w. wnet-flickrid mapping')
    parser.add_option("-n", '--wnet_list_file', dest='wnet_list_file', default="wnet-50.txt", 
                      help='list of wordnet ids with at least 50 flickr imgs associated')
    parser.add_option("-u", '--img_usr_file', dest='img_usr_file', default="flickr_usr_5M.txt", 
                      help='list of flickr ids and their usr ids')
    parser.add_option("-t", '--img_tag_file', dest='img_tag_file', default="flickr_tags_5M.txt", 
                      help='list of flickr ids and their tags')
    parser.add_option("-v", '--vocab_file', dest='vocab_file', default="vocab.txt", 
                      help='list of vocab words')
    parser.add_option("", '--wordnet_def_file', dest='wordnet_def_file', default="wordnet_tags.txt", 
                      help='list of wordnet ids and (cleaned) words in their synset')
    parser.add_option("-o", '--out_file_prefix', dest='out_file_prefix', default="wordnet_tagsfreq", 
                      help='file name (w/o extension) of the output file')
    (opts, __args) = parser.parse_args(sys.argv)
    
    lines = parse_lines_fromfile(os.path.join(opts.db_dir, opts.wnet_list_file) )
    wnet_cnt = dict ( map(lambda tt: [tt[1], int(tt[0])], lines ) ) # flip the key and val
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s read %d synset counts from %s" % (tt, len(wnet_cnt), opts.wnet_list_file)
    
    lines = parse_lines_fromfile(os.path.join(opts.db_dir, opts.img_usr_file) )
    img_usr = dict(lines)
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s read %d usr ids from %s" % (tt, len(img_usr), opts.img_usr_file)
    
    #img_tag = {}
    #if 0:
    lines = parse_lines_fromfile(os.path.join(opts.db_dir, opts.img_tag_file) )
    img_tag = dict ( map(lambda tt: [tt[0], tt[1].split(",")], lines ) ) # imgid: list of tags
    
    total_tag_cnt = sum(map(lambda tl:len(tl), img_tag.values() ))
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s read %d imgs, %d tags total from %s" % (tt, len(img_tag), total_tag_cnt, opts.img_tag_file)
    
    #lines = parse_lines_fromfile(os.path.join(opts.db_dir, opts.wordnet_def_file) )
    #wn_def = dict ( map(lambda tt: [tt[0], tt[1].split(",")], lines ) )
    
    vocab = filter(len, open(os.path.join(opts.db_dir, opts.vocab_file), "r").read().split("\n"))
    vocab.sort()
    rows_wn = sorted(wnet_cnt.keys())
    #WT = divisi2.SparseMatrix( )
    if os.path.isfile(opts.wnet_dir):
        tarw = tarfile.open(opts.wnet_dir, "r:gz")
    
    WT = sparse.PysparseMatrix(nrow=len(rows_wn), ncol=len(vocab), sizeHint=len(rows_wn)*20)
    fo = open(os.path.join(opts.db_dir, opts.out_file_prefix+".txt"), "wt")
    fj = open(os.path.join(opts.db_dir, opts.out_file_prefix+".json"), "wt")
    fe = open(os.path.join(opts.db_dir, opts.out_file_prefix+"_ERR.txt"), "wt")
    for ir, wnid in enumerate(rows_wn):
        if os.path.isdir(opts.wnet_dir):
            wn_list_file = os.path.join(opts.wnet_dir, wnid+".txt")
            wh = open(wn_list_file, "r")
        else: # is a tar file
            wh = tarw.extractfile("wnet/"+wnid+".txt")
        
        #try:
        lines = filter(len, wh.read().split("\n"))
        #nfields = map(lambda s: len(s.split()), lines)
        #lines = map(operator.itemgetter(1), filter(lambda t: nfields[t[0]]<3, enumerate(lines)) )
        imgid_list = map(lambda s: s.split()[1] if len(s.split())==3 else "", lines)
        usr_list = map(lambda imid: img_usr[imid] if imid in img_usr else "", imgid_list)
        tag_list = map(lambda imid: img_tag[imid] if imid in img_tag else [], imgid_list)
        #except:            
        #    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        #    print "%s ERR processing synset#%d %s" % (tt, ir, wnid)
        #    fe.write("%s\t%d\t%s\n" % (tt, ir, wnid))
        #    continue
        tag_u = []
        #print "  ", len(imgid_list), len(usr_list), len(tag_list), sum(map(len, tag_list))
        for u in set(usr_list):
            if not u:
                continue
            iu = filter(lambda ii: usr_list[ii]==u, range(len(usr_list)) )
            tag_u.append( list(set( reduce(lambda u,v: u + tag_list[v], iu, []) )) ) 
        
        tag_u = reduce(lambda u,v: u+v, tag_u)
        tcnt = dict((t, tag_u.count(t)) for t in tag_u)
        icol = []
        for t in tcnt.keys():
            icol.append( vocab.index(t) )
        WT.put(tcnt.values(), [ir]*len(tcnt), icol)
        
        tcnt_sorted = sorted(tcnt.iteritems(), key=operator.itemgetter(1), reverse=True)
        fo.write( "%s\t%s\n" % (wnid, " ".join( map(lambda t: "%s:%d"%(t[0], t[1]), tcnt_sorted) ) ) )
        fj.write(json.dumps({wnid: dict(tcnt_sorted) }, sort_keys=True, indent=2))
        
        if ir % 100 == 0:            
            if len(tcnt_sorted)>10:
                tcnt_sorted = tcnt_sorted[:10]
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            #wstr = ",".join(wn_def[wnid]) if wnid in wn_def else " "
            print ('%s synset#%5d %s: %d imgs, %d users, %d tags'
                % (tt, ir, wnid, len(usr_list), len(set(usr_list)), sum(map(len, tag_list)) ) )
            print "\t%s\n" % " ".join( map(lambda t: "%s:%d"%(t[0], t[1]), tcnt_sorted) )
    
    fe.close()
    fo.close()
    fj.close()
    WTd = divisi2.SparseMatrix(WT, row_labels=rows_wn, col_labels=vocab)
    pickle.dump(WTd.to_state(), open(os.path.join(opts.db_dir, opts.out_file_prefix+".pkl"), "wb"))
    
    return WT

if __name__ == '__main__':
    argv = sys.argv 
    if '--make_vocab' in argv:
        argv.remove('--make_vocab')
        make_vocab(argv)
    elif "--record_svd" in argv:
        argv.remove('--record_svd')
        record_svd(argv)
    elif "--make_wn_tag" in argv:
        argv.remove('--make_wn_tag')
        make_wn_tag(argv)
    else:  
        pass
    
    """    
    elif "--make_bigram" in argv:
        argv.remove('--make_bigram')
        make_bigram(argv)
    """
            
    #bg_dict = read_bigram_subset(bg_file, vocab, thresh=5)
    #pickle.dump(bg_dict, open('bg.pkl', 'wb'))
    # read in conceptnet
    """       sim = compute_sim_conceptnet()
    """
    #sim = compute_sim_conceptnet(cn_words)
    #pickle.dump(sim, open('sim.pkl', 'wb'))
    
    # compare the two
    
    
    