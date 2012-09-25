
import sys, os
import json
import sqlite3
import codecs
import pickle
from glob import glob
from datetime import datetime
from optparse import OptionParser
from operator import itemgetter

"""
    walk a dir of flickr json files
    ... read/write to cache file
    ... clean tags
    ... count bigrams
"""
def bigram_flickr(argv):
    if len(argv)<2:
        argv = ['-h']
    
    parser = OptionParser(description='parse flickr json files')
    parser.add_option("-i", "--in_dir", dest="in_dir", 
        default='', help="input dir containing all json files")
    parser.add_option("-o", "--out_dir", dest="out_dir", 
        default='', help="out txt")
    parser.add_option("-r", "--redo_data", dest="redo_data", action="store_true", 
                      default=False, help="re-extract docs, ignore existing cache")
    parser.add_option('-d', '--db_file', dest='db_file', 
        default='dict.db', help='file containing dictionary db')
    parser.add_option('-a', '--addl_vocab', dest="addl_vocab", default="")
    
    opts, __ = parser.parse_args(argv)
    
    if opts.addl_vocab: #additional out-of-dictionary words 
        addl_vocab = open(opts.addl_vocab, 'rt').read().split()
    else:
        addl_vocab = []
    
    __, dir_name = os.path.split(opts.in_dir.strip(os.sep))
    out_file = os.path.join(opts.out_dir, dir_name+".bigram")
    uni_file = os.path.join(opts.out_dir, dir_name+".unigram")
    cache_file = os.path.join(opts.out_dir, dir_name+".cache")
    if not os.path.exists(opts.out_dir):
        os.makedirs(opts.out_dir)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processing dir %s" % (tt, opts.in_dir)
    
    """ ouptut struct """
    bg_dict = {}
    uni_dict = {}
    """ get dictionary from SQlite """
    conn = sqlite3.connect(opts.db_file)
    cursor = conn.cursor()
    jcnt = 0
    errcnt = 0
    emtcnt = 0
    if not opts.redo_data and os.path.exists(cache_file):
        print "reading from file: %s" % cache_file
        for cl in codecs.open(cache_file, encoding='utf-8', mode='rt'):
            tmp = cl.split()[1]
            tag_raw = tmp.split(',')
            #tag_cleaned = map(lambda s: norm_tag(s, cur), tag_raw)
            accumulate_bg(tag_raw, bg_dict, uni_dict, cursor, addl_vocab=addl_vocab)
            jcnt += 1
            if jcnt % 1000 == 0:
                    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                    print "%s %7d docs done" % (tt, jcnt)
            
    else:
        cfh = codecs.open(cache_file, encoding='utf-8', mode='wt')
        for cp, __cd, fn in os.walk(opts.in_dir):         # UnusedVariable
            jn = filter(lambda s: s.find(".json")>0, fn)
            for j in jn:
                meta_name = os.path.join(cp, j)
                jstr = codecs.open(meta_name, encoding='utf-8', mode='rt').read()
                try:
                    jinfo = json.loads( jstr )
                except Exception, e:
                    print " ERR parsing json from %s" % j 
                    print e
                    print ""
                    
                if 'stat' not in jinfo or not jinfo['stat']=='ok' :
                    errcnt += 1
                else:                    
                    tag_raw = map(lambda s:s["_content"], jinfo["photo"]["tags"]["tag"])
                    if tag_raw:
                        jcnt += 1
                        accumulate_bg(tag_raw, bg_dict, uni_dict, cursor)
                        #print repr(bg_dict)
                        # write to cache file
                        imid,__ = os.path.splitext(j)
                        cfh.write("%s\t%s\n" % (imid, ",".join(tag_raw)))
                        if jcnt % 1000 == 0:
                            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                            print "%s %7d docs done" % (tt, jcnt)
                    else:
                        emtcnt += 1
        cfh.close()
        
    conn.close()
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s: %d docs with tags, %d empty, %d failed, %d words " % (tt, jcnt, emtcnt, errcnt, len(uni_dict))
    
    write_unigram_file(uni_dict, uni_file)
    write_bigram_file(bg_dict, out_file)
    
    # DONE

def write_unigram_file(uni_dict, uni_file):
    ufh = codecs.open(uni_file, encoding='utf-8', mode='wt')
    
    sk = sorted(uni_dict.keys())
    for j1 in sk:
        ufh.write("%5d\t%s\n" % (uni_dict[j1], j1) )
    ufh.close()
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s: %d unigrams wrote to \n\t%s" % (tt, len(uni_dict), uni_file)


def write_bigram_file(bg_dict, out_file):    
    ofh = codecs.open(out_file, encoding='utf-8', mode='wt')
    bcnt = 0
    sk = sorted(bg_dict.keys())
    for j1 in sk:
        sk2 = sorted(bg_dict[j1].keys())
        for j2 in sk2:
            ofh.write("%5d\t%s\t%s\n" % (bg_dict[j1][j2], j1, j2) )
            bcnt += 1
    ofh.close()
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s: %d bigrams wrote to \n\t%s " % (tt, bcnt, out_file)
    

def sort_bg(bg_dict):
    bg_tuples = []
    for u in bg_dict:
        for v in bg_dict[u]:
            bg_tuples.append((u,v,bg_dict[u][v]))
    
    bg_tuples.sort(key=itemgetter(2), reverse=True)
    
    return bg_tuples
    
""" is bg dict mutable """
def accumulate_bg(tr, bg_dict, uni_dict, cursor, addl_vocab=[]):
    if cursor:
        tt = map(lambda s: norm_tag(s, cursor, addl_vocab=addl_vocab), tr)
        tt = filter(lambda s: len(s)>1, tt)
        tt = list(set(tt))
    else:
        tt = list(set(tr)) 
        
    tt.sort()
    for i in range(len(tt)):
        if uni_dict:
            if tt[i] not in uni_dict:
                uni_dict[tt[i]] = 1
            else:
                uni_dict[tt[i]] += 1
        for j in range(i):
            if tt[j] not in bg_dict:
                bg_dict[tt[j]] = {}
            if tt[i] not in bg_dict[tt[j]]:
                bg_dict[tt[j]][tt[i]] = 1
            else:
                bg_dict[tt[j]][tt[i]] += 1
    """ bg_dict is modified inside """
    #print repr(tr) + "--->" + repr(tt)
    #print "in:" + repr(bg_dict)
                
    
def norm_tag(in_tag, cur, addl_vocab=[],filter_stopword=0):
    out_tag = ''
    
    stmt = "SELECT string,word FROM dict WHERE string='%s'" % in_tag
    try:
        #for row in cur.execute(stmt):
        for row in cur.execute("SELECT string,word,is_stopword FROM dict WHERE string=?", [in_tag]):
            assert not out_tag, "one in_tag should have only one match!"
            out_tag = row[1]
            if filter_stopword and row[2]==1:
                out_tag = ""
    except:
        print 'err in SQL: "%s"' % stmt
        raise
    
    if not out_tag and addl_vocab and in_tag in addl_vocab:
        out_tag = in_tag
           
    return out_tag

""" not used  using read_bigram suffices 
"""
def merge_bigram(dest_dict, src_dict):
    """ dest[w1][w2] = cnt"""
    for w1, d1 in src_dict.iteritems():
        if w1 not in dest_dict:
            dest_dict[w1] = {}
        for w2, c in d1.iteritems():
            if w2 not in dest_dict[w1]:
                dest_dict[w1][w2] = c
            else:
                dest_dict[w1][w2] += c
    # DONE
""" not used  using read_bigram suffices """

def read_one_bigram_line(linetxt):
    tmp = linetxt.strip().split()
    c = int(tmp[0])
    w1 = min(tmp[1:])
    w2 = max(tmp[1:])
    return c,w1,w2

def read_unigram(src_file, ug_dict):
    new_cnt = 0
    line_cnt = 0
    for cl in open(src_file, 'r'):
        a = cl.strip().split()
        uc = int(a[0])
        uw = a[1]
        if uw not in ug_dict:
            ug_dict[uw] = uc
            new_cnt += 1
        else:
            ug_dict[uw] += uc
        line_cnt += 1
        
    return new_cnt,line_cnt
    
def read_bigram(src_file, bg_dict):
    new_cnt = 0
    line_cnt = 0
    for cl in open(src_file, 'r'):
        """
        tmp = cl.strip().split()
        c = int(tmp[0])
        w1 = min(tmp[1:])
        w2 = max(tmp[1:])
        """
        c,w1,w2 = read_one_bigram_line(cl)
        if w1 not in bg_dict:
            bg_dict[w1] = {}
            bg_dict[w1][w2] = c
            new_cnt += 1
        else:
            if w2 not in bg_dict[w1]:
                bg_dict[w1][w2] = c
                new_cnt += 1
            else:
                bg_dict[w1][w2] += c
        line_cnt += 1
        
    return new_cnt,line_cnt
    #return bg_dict

def bigram_reduce(argv):
    if len(argv)<2:
        argv = ['-h']
    
    parser = OptionParser(description='parse flickr json files')
    parser.add_option("-i", "--in_dir", dest="in_dir", 
        default='', help="input dir containing input stat files")
    parser.add_option("-g", "--in_grep", dest="in_grep", 
        default='', help="input grep pattern for unigram/bigram files")
    parser.add_option("-u", "--proc_unigram", dest="proc_unigram",
                      default="", help="ingest/combine unigram, '':do nothing, 'm': in memory, 'd': in database")
    parser.add_option("-b", "--proc_bigram", dest="proc_bigram",  
                      default="", help="ingest/combine bigram, '':do nothing, 'm': in memory, 'd': in database")
    parser.add_option('-d', '--db_file', dest='db_file', 
        default='dict.db', help='file containing dictionary db')
    opts, __ = parser.parse_args(argv)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processing dir %s" % (tt, opts.in_dir)
    
    """ find files to process """
    #from glob import glob
    ufile = glob(os.path.join(opts.in_dir, opts.in_grep+'*.unigram'))
    bfile = glob(os.path.join(opts.in_dir, opts.in_grep+'*.bigram'))
    
    """ open SQlite """
    conn = sqlite3.connect(opts.db_file)
    cursor = conn.cursor()
    
    if opts.proc_unigram and opts.proc_unigram[0].lower()=='m':
        """ do this in memory """
        ug_dict = {}
        if os.path.exists(opts.db_file):
            ug_cnt, line_cnt = read_unigram(opts.db_file, ug_dict)
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s load %d unigrams, from %d lines in %s " % (tt, ug_cnt, line_cnt, opts.db_file )
        else:
            ug_cnt = 0
            print "target file do not exist, creat new %s" % opts.db_file
        
        fcnt = 0
        for uf in ufile:
            new_cnt,line_cnt = read_unigram(uf, ug_dict)
            ug_cnt += new_cnt
            fcnt += 1
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s '%s'-file# %3d/%3d, total unigrams %d, %d new from %d lines" % (tt, os.path.split(uf)[1], fcnt, len(ufile), ug_cnt, new_cnt, line_cnt )
        
        write_unigram_file(ug_dict, opts.db_file)
        
    elif opts.proc_unigram:
        """ open SQlite """
        conn = sqlite3.connect(opts.db_file)
        cursor = conn.cursor()
        conn.execute("""DELETE FROM unigram""")
        conn.commit()
        for uf in ufile:
            ucnt = 0
            for ul in codecs.open(uf, mode="r", encoding='utf-8'):
                tmp = ul.strip().split()
                cur_freq = int(tmp[0])
                word = tmp[1]
                ucnt += 1
                cursor.execute("SELECT word,freq FROM unigram WHERE word='%s'" % word)
                r = cursor.fetchone()
                if r:
                    ufreq = cur_freq + int(r[1])
                    cursor.execute("UPDATE unigram SET freq=%d WHERE word='%s'" % (ufreq, word) )
                else:
                    ufreq = cur_freq
                    cursor.execute("INSERT INTO unigram (word,freq) VALUES (?,?)", (word, ufreq) )
            
            conn.commit()
            cursor.execute("SELECT COUNT(*) FROM unigram")
            r = int(cursor.fetchone()[0])
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s total unigrams %d, added %d in %s " % (tt, r, ucnt, os.path.split(uf)[1] )

        print "DONE ingesting unigram stats\n"
        conn.close()
    

    
    if opts.proc_bigram and opts.proc_bigram[0].lower()=='m':
        """try to do this in memory """
        bg_dict = {}
        if os.path.exists(opts.db_file):
            bg_cnt, line_cnt = read_bigram(opts.db_file, bg_dict)
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s load %d bigrams, from %d lines in %s " % (tt, bg_cnt, line_cnt, opts.db_file )
        else:
            print "target file do not exist, creat new %s" % opts.db_file
        
        fcnt = 0
        for bf in bfile:
            new_cnt,line_cnt = read_bigram(bf, bg_dict)
            bg_cnt += new_cnt
            fcnt += 1
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s file# %3d/%3d, total bigrams %d, %d new from %d lines in %s " % (tt, fcnt, len(bfile), bg_cnt, new_cnt, line_cnt, os.path.split(bf)[1] )
        
        write_bigram_file(bg_dict, opts.db_file)
        # DONE
         
    elif opts.proc_bigram and opts.proc_bigram[0].lower()=='d':
        """ this become too slow, not used now"""
        """ open SQlite """
        conn = sqlite3.connect(opts.db_file)
        cursor = conn.cursor()
        conn.execute("""DELETE FROM bigram""")
        conn.commit()
        
        for bf in bfile:
            bcnt = 0
            for bl in codecs.open(bf, mode="r", encoding='utf-8'):
                tmp = bl.strip().split()
                cur_freq = int(tmp[0])
                word1 = tmp[1]
                word2 = tmp[2]
                bcnt += 1
                cursor.execute("SELECT word1,word2,freq FROM bigram WHERE word1='%s' AND word2='%s'" % (word1,word2) )
                r = cursor.fetchone()
                if r:
                    bfreq = cur_freq + int(r[2])
                    cursor.execute("UPDATE bigram SET freq=%d WHERE word1='%s' AND word2='%s'" % (bfreq, word1, word2) )
                else:
                    bfreq = cur_freq
                    cursor.execute("INSERT INTO bigram (word1,word2,freq) VALUES (?,?,?)", (word1, word2, bfreq) )
                if bcnt % 5000 ==0:
                    conn.commit()
                    
            conn.commit()
            cursor.execute("SELECT COUNT(*) FROM bigram")
            r = int(cursor.fetchone()[0])
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s total bigrams %d, added %d in %s " % (tt, r, bcnt, os.path.split(bf)[1] )
    
    
        conn.close()

def bigram_ingest(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='parse flickr json files')
    parser.add_option("-i", "--in_file", dest="in_file", 
        default='', help="input stat files")
    parser.add_option('-d', '--db_file', dest='db_file', 
        default='word_freq.db', help='file containing dictionary db')
    opts, __ = parser.parse_args(argv)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processing file %s into %s" % (tt, opts.in_file, opts.db_file)
    
    """ open SQlite """
    conn = sqlite3.connect(opts.db_file)
    cursor = conn.cursor()
    conn.execute("""DELETE FROM bigram""")
    conn.commit()
    
    linecnt = 0
    for cl in open(opts.in_file):
        c,w1,w2 = read_one_bigram_line(cl)
        cursor.execute("INSERT INTO bigram (word1,word2,freq) VALUES (?,?,?)", (w1, w2, c) )
        
        linecnt += 1
        if linecnt % 10000 == 0:
            conn.commit()
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s processed %d tuples" % (tt, linecnt)
    
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM bigram")
    r = int(cursor.fetchone()[0])
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s total bigrams %d from %s " % (tt, r, opts.in_file )
    conn.close()

def flickr_hash_dir(imgid, meta_dir, hash_level=2, chars_per_hash=2):
    hdir = []
    for i in range(hash_level):
        curs = i*chars_per_hash
        cure = curs + chars_per_hash
        hdir.append(imgid[curs:cure])
    meta_name = os.path.join(meta_dir, '/'.join(hdir), imgid+".json")
    return meta_name

def collect_user(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='parse flickr tag cache, noramlize, save to pickle file')
    parser.add_option("-i", "--in_id_file", dest="in_id_file", 
        default='', help="input flickr id file")
    parser.add_option('-o', '--out_file', dest='out_file', 
        default='', help='output file containing id-user mapping')
    parser.add_option('-m', '--meta_dir', dest='meta_dir', 
        default='/home/users/xlx/vault-xlx/imgnet-flicr/json', help='root dir containing metadata')
    opts, __ = parser.parse_args(argv)
    
    assert os.path.exists(opts.meta_dir), " ERR meta data dir must exist: %s" % opts.meta_dir
    
    id_list = open(opts.in_id_file, 'rt').read()
    id_list = id_list.split("\n")
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processing %d entries from %s " % (tt, len(id_list), opts.in_id_file)
    
    usr_dict = dict.fromkeys(id_list)
    fo = open(opts.out_file, 'wt')
    cnt = 0
    for imgid in id_list:
        if not imgid:
            continue
        meta_name = flickr_hash_dir(imgid, opts.meta_dir)
        jstr = codecs.open(meta_name, encoding='utf-8', mode='rt').read()
        try:
            jinfo = json.loads( jstr )
        except Exception, e:
            print " ERR parsing json for %s" % imgid
            print e
            print ""
        pinfo = jinfo["photo"]
        usr = pinfo["owner"]["nsid"]
        fo.write("%s\t%s\n" % (imgid, usr))
        usr_dict[imgid] = usr
        cnt += 1
        if cnt % 5000 ==0:
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s processed %d out of %d entries " % (tt, cnt, len(id_list))
    
    fo.close()
    fp = os.path.splitext(opts.out_file)[0] + ".pkl"
    pickle.dump(open(fp, 'wb'), usr_dict)
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s DONE %d " % (tt, cnt, len(id_list))
    
def collect_tags(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='parse flickr tag cache, noramlize, save to pickle file')
    parser.add_option("-i", "--in_file_glob", dest="in_file_glob", 
                      default='/home/users/xlx/vault-xlx/imgnet-flickr/bigram/*.cache', help="input cache file glob pattern")
    parser.add_option('-l', '--id_index', dest='id_index', 
        default='/home/users/xlx/vault-xlx/imgnet-flickr/flickr_id_5M.txt', help='file containing all flickr ids')
    parser.add_option('-d', '--db_dict', dest='db_dict', 
        default='dict.db', help='dictionary mapping')
    parser.add_option('-v', '--vocab_file', dest='vocab_file', 
        default='vocab_flickr.txt', help='vocabulary file')
    parser.add_option("", "--addl_words", dest='addl_words', 
        default='places_etc.txt', help='additional vocabulary file')
    parser.add_option("-a", "--aux_dir", dest="aux_dir", 
        default="/home/users/xlx/vault-xlx/imgnet-flickr/db2", help="dir for aux and output files")
    opts,__ = parser.parse_args(argv)
    
    #id_list = open(opts.in_id_file, 'rt').read()
    #id_list = id_list.split("\n")
    #tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    #print "%s processing %d entries from %s " % (tt, len(id_list), opts.in_id_file)
    
    vocab = open(os.path.join(opts.aux_dir, opts.vocab_file), "rt").read().split("\n")
    #print len(vocab)
    #print opts.db_dict
    conn = sqlite3.connect(os.path.join(opts.aux_dir, opts.db_dict))
    cursor = conn.cursor()
    eng_dict = {}
    for row in cursor.execute("SELECT string,word,is_stopword FROM dict"):
        if int(row[2])==1:
            continue # stop word, skip
        eng_dict[row[0]] = row[1] 
    conn.close()
    if opts.addl_words:
        for cl in open(os.path.join(opts.aux_dir, opts.addl_words), 'rt'):
            w = cl.strip()
            if w not in eng_dict:
                eng_dict[w] = w;
    
    #tag_dict = dict.fromkeys(id_list)
    fo = open(os.path.join(opts.aux_dir, "flickr_tags_5M.txt"), "wt")
    tag_file_list = glob(opts.in_file_glob)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s %d valid tags, %d output tags" % (tt, len(eng_dict), len(set(eng_dict.values())) )
    print "%s processing %d file from %s" % (tt, len(tag_file_list), opts.in_file_glob)
    vcnt = 0
    for tf in tag_file_list:
        tc = open(tf, 'rt').read().split("\n")
        for line in tc:
            tmp = line.strip().split()
            if len(tmp)<2:
                continue
            
            k = tmp[0]
            vv = tmp[1].split(",")
            #print str(vv) + "\n" + repr(tmp)            
            if len(vv) and type(vv) is list:
                v = map(lambda s: eng_dict[s] if s in eng_dict else "", vv)
                v = filter(lambda s: len(s)>0 and s in vocab, v)
                v = list(set(v))
            else:
                v = []
            #print v
            #tag_dict[k] = v
            if v:
                fo.write("%s\t%s\n" % (k, ",".join(v)))
                vcnt += 1
            
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s wrote %d entries from %d lines in %s " % (tt, vcnt, len(tc), os.path.split(tf)[1])

if __name__ == '__main__':  
    argv = sys.argv 
    if '--reduce' in argv:
        argv.remove('--reduce')
        bigram_reduce(argv)
    elif '--ingest' in argv:
        argv.remove('--ingest')
        bigram_ingest(argv)
    elif '--collect_tags' in argv:
        argv.remove('--collect_tags')
        collect_tags(argv)
    elif '--collect_user' in argv:
        argv.remove('--collect_user')
        collect_user(argv)
    else:  
        bigram_flickr(argv)
        