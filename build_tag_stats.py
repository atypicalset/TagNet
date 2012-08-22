

import sys,os
import sqlite3
import pickle
import networkx as nx
import divisi2
from nltk.corpus import wordnet as wn

from datetime import datetime
from optparse import OptionParser
from count_bigram_flickr import norm_tag, read_unigram #read_bigram(src_file, bg_dict)
from compare_cooc import read_bigram_list

def get_conceptnet_words(db_file, addl_vocab=[]):
    A = divisi2.network.conceptnet_matrix('en')
    wa = A.row_labels
    #wb = A.col_labels
    #all_t2 = map(lambda s: s[2], list(wb))
    #ww = filter(lambda s: s==s.split()[0], wa) # keep 1-word terms only
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    wd = {}
    cn_vocab = {}
    null_count = 0
    for w in wa:
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
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s mapped %d concept-net nodes, %d empty" % (tt, len(wa), null_count)
    
    return wd, cn_vocab

""" not used now, replace with the contents in make_vocab() """
def intersect_vocab(db_dict, tag_file, addl_vocab=[], db_wn='', wn_list=[]):
    cn_words, cn_vocab = get_conceptnet_words(db_dict, addl_vocab)
    print "ConceptNet: %d words, %d cleaned" % (len(cn_words), len(cn_vocab) )
    
    fr_words = {}
    cnt,__ = read_unigram(tag_file, fr_words)
    fr_words = filter(lambda k: fr_words[k]>5, fr_words.keys())
    
    vocab = list( set(cn_vocab) & set(fr_words) )
    vocab = filter(lambda s: len(s)>0, vocab)
    print "Flickr kep %d/%d tags, %d in common with ConceptNet " % (len(fr_words), cnt, len(vocab) )
    print " %d words in the intersected vocab" % len(vocab)
    
    # now deal with wordnet
    #wn_words = get_wordnet_words(db_wn, wn_list, db_dict, vocab)
    
    return vocab, fr_words

def make_vocab(argv):
    if len(argv)<2:
            argv = ['-h']
    parser = OptionParser(description='construct+compare conceptnet and flickr word similarities')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db files')
    parser.add_option("", '--db_dict', dest='db_dict', default="dict.db", help='dictionary')
    parser.add_option("", '--unigram_file', dest='unigram_file', default="unigram.txt", help='unigrams file %word count%')
    parser.add_option("", '--wn_list', dest='wn_list', default="wnet-50.txt", help='')
    parser.add_option("", '--addl_vocab', dest='addl_vocab', default="places_etc.txt", help='')
    #parser.add_option("", '--db_wordnet', dest='db_wordnet', default="wordnet.db", help='')
    #parser.add_option("", '--bigram_file', dest='bigram_file', default="bigram_filtered.txt", help='')
    (opts, __args) = parser.parse_args(sys.argv)
    
    # intersect the two dictionaries first
    db_dict = os.path.join(opts.db_dir, opts.db_dict)
    ug_file = os.path.join(opts.db_dir, opts.unigram_file)
    addl_vocab = open(os.path.join(opts.db_dir, opts.addl_vocab), 'rt').read().split()
    #db_wn = os.path.join(opts.db_dir, opts.db_wordnet)
    #wn_list = os.path.join(opts.db_dir, opts.wn_list)
    
    #vocab, fr_words = intersect_vocab(db_dict, ug_file, addl_vocab=addl_vocab)
    
    cn_words, cn_vocab = get_conceptnet_words(db_dict, addl_vocab)
    print "ConceptNet: %d words, %d cleaned" % (len(cn_words), len(cn_vocab) )
    
    fr_words = {}
    cnt,__ = read_unigram(ug_file, fr_words)
    fr_words = filter(lambda k: fr_words[k]>5, fr_words.keys())
    
    vocab = list( set(cn_vocab) & set(fr_words) )
    vocab = filter(lambda s: len(s)>0, vocab)
    print "Flickr kep %d/%d tags, %d in common with ConceptNet " % (len(fr_words), cnt, len(vocab) )
    print " %d words in the intersected vocab" % len(vocab)
    
    #open(os.path.join(opts.db_dir, 'vocab.txt'), "wt").write("\n".join(vocab))
    fr_words.sort()
    open(os.path.join(opts.db_dir, 'vocab_flickr.txt'), "wt").write("\n".join(fr_words))
    
    fo = open(os.path.join(opts.db_dir, 'vocab_conceptnet.txt'), "wt")
    for k, v in cn_words.iteritems():
        if v:
            fo.write("%s\t%s\n" % (k, ",".join(v)) )
    fo.close()


def add_conceptnet_edges(G, conceptnet_words, x1, x2):
    for w1 in conceptnet_words[x1]:
        for w2 in conceptnet_words[x2]:
            ow = G[w1][w2]['weight'] if w1 in G and w2 in G[w1] else 0
            cw = ow + 1
            G.add_edge(w1, w2, weight=cw)

def write_output_graph(fo, G, ginfo):
    fo.write("#%s\n" % (",".join(ginfo))) # comment line, type+relation
    for u,v in G.edges_iter():
        fo.write("%0.1f\t%s\t%s\n" % (G[u][v]['weight'], u, v) )
    fo.write("----\n\n")
    
def find_subgraph_star(Gd, rel, pair_tuple, conceptnet_words, Tdict, Tnode_dict, direction='out'):
    w1 = pair_tuple[0]
    w2 = pair_tuple[1]
    skip_graph = False
    root = ''
    if (w1, w2) in Gd[rel].edges(): 
        if direction=='out' and Gd[rel].out_degree(w1) > 1:
            # w1 is the star center
            root = w1
        elif direction=='in' and Gd[rel].in_degree(w1) > 1:
            root = w1
        else:
            skip_graph = True
    elif (w2, w1) in Gd[rel].edges() :
        if direction=='out' and Gd[rel].out_degree(w2) > 1:
            # w1 is the star center
            root = w2
        elif direction=='in' and Gd[rel].in_degree(w2) > 1:
            root = w2
        else:
            skip_graph = True
    else:
        print " not an edge: (%s, %s), %s" % (w1, w2, rel)
        skip_graph = True
    
    if root and root in Tdict[rel]:
        print " graph already exist: %s, %s" % (root, rel)
        skip_graph = True
        
    if skip_graph:
        return [], [], [] # already exist
    
    if direction=='out' :
        leaves = Gd[rel][root].keys() # weights are ignored, only term count
    elif direction =='in':
        leaves = Gd[rel].predecessors(root)
    
    all_nodes = tuple(sorted([root]+leaves))
    if all_nodes in Tnode_dict:
        print " node dup exists: %s, %s" % (root, rel)
        return [], [], [] # already exist
    else:
        Tnode_dict[all_nodes] = 'S'
    
    curg = nx.DiGraph()
    
    add_conceptnet_edges(curg, conceptnet_words, root, root)
    for wv in leaves:
        add_conceptnet_edges(curg, conceptnet_words, root, wv)
        add_conceptnet_edges(curg, conceptnet_words, wv, root)
        add_conceptnet_edges(curg, conceptnet_words, wv, wv)    # add self
    
    return curg, root, leaves

def find_common_clique(Gd, curr, wk, conceptnet_words, Tdict, Tnode_dict):
    gr = nx.Graph(Gd[curr]) # convert to undirected graph
    w1 = min(wk)
    w2 = max(wk)
    wk = (w1, w2)
    w1 = wk[0]
    w2 = wk[1]
    cli1 = sorted(nx.cliques_containing_node(gr, w1), key=lambda s:len(s), reverse=True)
    cli2 = sorted(nx.cliques_containing_node(gr, w2), key=lambda s:len(s), reverse=True)
    
    if wk in Tdict[curr]:
        print " graph already exist: %s, %s" % (curr, repr(wk))
        return [], [], []
    
    # find the largest common clique, size >= 3
    cnodes = []
    for i1 in range(len(cli1)):
        if len(cli1[i1])<3:
            break
        for i2 in range(len(cli2)):
            if len(cli2[i2]) == len(cli1[i1]) and sorted(cli2[i2])==sorted(cli1[i1]):
                cnodes = sorted(cli2[i2])
                break
        if len(cnodes) > 0:
            break
    
    if len(cnodes) > 0:
        all_nodes = tuple(sorted(cnodes))
        if all_nodes in Tnode_dict:
            print " node dup exists: %s, %s" % (repr(wk), curr)
            return [], [], [] # already exist
        else:
            Tnode_dict[all_nodes] = 'C'
        
        curg = nx.DiGraph()

        for u in range(len(cnodes)):
            un = cnodes[u]
            add_conceptnet_edges(curg, conceptnet_words, un, un)
            for v in range(len(cnodes)):
                vn = cnodes[v]
                if not vn==un:
                    add_conceptnet_edges(curg, conceptnet_words, un, vn)
    else:
        print " no good cliques found: %s, %s" % (repr(wk), curr)
        return [], [], []
    
    return curg, wk, cnodes

def get_top_bigrams(argv):
    """
    input:  two word-pair files (bigram, conceptnet), conceptnet graph, sorted reverse order by correlation
    output: slices of relations
    """
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='construct+compare conceptnet and flickr word similarities')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db files')
    parser.add_option("-n", '--num_pairs', dest='num_pairs', type='int', default=200, help="")
    parser.add_option("-q", '--query_rel', dest='query_rel', default='', help="")
    #parser.add_option("-w", '--write_graph', dest='write_graph', type='int', default=1, help="write or only print graph to screen")
    
    #parser.add_option("", '--addl_vocab', dest='addl_vocab', default="places_etc.txt", help='')
    parser.add_option("", '--bigram_file', dest='bigram_file', default="flickr.bigram.ge5.txt", help='')    
    parser.add_option("", '--cn_pair_file', dest='cn_pair_file', default='conceptnet_sim_2M.txt', help='')
    parser.add_option("", '--graph_piece_pkl', dest='graph_piece_pkl', default='Gd.pkl', help='')
    parser.add_option("", '--cn_vocab_pkl', dest='cn_vocab_pkl', default='vocab_conceptnet_invertidx.pkl', help='')
    parser.add_option("", '--out_graph_pkl', dest='out_graph_pkl', default='Tdict.pkl', help="")
    parser.add_option("", '--out_graph_txt', dest='out_graph_txt', default='Tgraphs.txt', help="")
    (opts, __args) = parser.parse_args(sys.argv)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s loading files" % (tt)
    # load graph pieces
    Gd = pickle.load(open(os.path.join(opts.db_dir, opts.graph_piece_pkl), 'rb'))
    _conceptnet_invidx, conceptnet_words = pickle.load(open(os.path.join(opts.db_dir, opts.cn_vocab_pkl), 'rb'))
    #a = pickle.load(open(os.path.join(opts.db_dir, opts.cn_vocab_pkl), 'rb'))
    #print len(a)
    # load bigrams, sorted descending
    bg_file = os.path.join(opts.db_dir, opts.bigram_file)
    bg_w1,bg_w2,bg_fq = read_bigram_list(bg_file)
    # load conceptnet similariy, sorted descending
    cnpair_file = os.path.join(opts.db_dir, opts.cn_pair_file)
    cp_w1,cp_w2,cp_fq = read_bigram_list(cnpair_file, numtype="float")
    # cpw12 = zip(cp_w1, cp_w2) #?? slow
    cpw12 = dict( zip(zip(cp_w1, cp_w2), range(len(cp_w1)) ) )
    
    lb = len(bg_w1)
    lc = len(cp_w1)
    pair_rscore = list(cp_fq)
    pair_nscore = list(cp_fq)
    print lb, lc
    print id(pair_rscore), id(pair_nscore)
    pcnt = 0
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s ranking %d pairs" % (tt, lc)
    for ib in range(lc): #zip(range(len(bg_w1)), bg_w1, bg_w2):
        w1 = bg_w1[ib]
        w2 = bg_w2[ib]
        if (w1,w2) in cpw12:
            #iw = cpw12.index( (w1,w2) )
            iw = cpw12[ (w1,w2) ]
            pair_rscore[ib] = .5*( (lb-ib)/(1.*lb)   + (lc-iw)/(1.*lc) )
            pair_nscore[ib] = .5*( ib/(1.*lb)        + (lc-iw)/(1.*lc) )
            pcnt += 1
        else:
            pair_rscore[ib] = -1.
            pair_nscore[ib] = -1.
        if (ib+1) % 500000 == 0 :
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s %d pairs processed, %d exist" % (tt, ib+1, pcnt)
    
    jr_pair = sorted(range(len(pair_rscore)), key=pair_rscore.__getitem__, reverse=True)
    #jn_pair = sorted(range(len(pair_nscore)), key=pair_nscore.__getitem__, reverse=True)
    #num = 200
    
    concept_cols = {}
    # define graph type for each relation
    rel_gtype = {"PartOf": ['o'], "AtLocation": ['o'], "LocatedNear":['o'], "ConceptuallyRelatedTo": ['c'], 
                 "HasA": ['o', 'i', 'c'], 'HasProperty':['o', 'i', 'c'], 
                 'UsedFor':['o', 'i', 'c'], 'CapableOf':['o', 'i', 'c'], 'IsA':['o', 'i']}
    # dict of output
    Tdict = dict.fromkeys(rel_gtype.keys(), {})
    Tnode_dict = {} # dict for existing graphs, hashed as sort(nodes):'S'/'C' -- star or clique
    # output text file
    fo = open(os.path.join(opts.db_dir, opts.out_graph_txt), 'wt')
    gcnt = 0
    
    print "\nr-combined\ts-combined\tr-bigram\tr-concept\ts-bigram\ts-concept\tword1\tword2\t"
    print "-------------  -------------  -------------\n"
    for j in range(opts.num_pairs):
        jb = jr_pair[j]
        w1 = bg_w1[jr_pair[j]]
        w2 = bg_w2[jr_pair[j]]
        js = pair_rscore[jr_pair[j]]
        # find the corresponding graph component(s): star on "AtLocation" and "PartOf"
        jc = cpw12[ (w1,w2) ]
        print "%d\t%f\t%d\t%d\t%0.1f\t%0.4f\t%s\t%s" % (j, js, jb, jc, bg_fq[jb], cp_fq[jc], w1, w2)
        
        # query and explore a relation
        if opts.query_rel:
            curr = opts.query_rel
            Tdict[curr] = {}
            
            curg,root,leaves = find_subgraph_star(Gd, curr, (w1,w2), conceptnet_words, Tdict, Tnode_dict, direction="out")
            if curg:
                tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                print "%s Graph #%d, rel=%s, root(out) = %s, size=%s" % (tt, gcnt, curr, root, repr([len(leaves)+1, len(curg)]))
                print [root, leaves]
                #print curg.nodes()
                print "-------------  -------------  -------------\n" 
            
            curg,root,leaves = find_subgraph_star(Gd, curr, (w1,w2), conceptnet_words, Tdict, Tnode_dict, direction="in")
            if curg:
                tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                print "%s Graph #%d, rel=%s, root(in) = %s, size=%s" % (tt, gcnt, curr, root, repr([len(leaves)+1, len(curg)]))
                print [root, leaves]
                #print curg.nodes()
                print "-------------  -------------  -------------\n" 
                    
            curg, wk, cnodes = find_common_clique(Gd, curr, (w1, w2), conceptnet_words, Tdict, Tnode_dict)
            if cnodes:
                print "%s Graph #%d, rel=%s, root = %s, size=%s" % (tt, gcnt, curr, repr(wk), repr([len(cnodes), len(curg)]))
                print cnodes
                #print curg.nodes()
                print "-------------  -------------  -------------\n"
            continue    
            
        """
            star component on "partof" and "at location"
        """
        for curr, gtype in rel_gtype.iteritems(): #["PartOf", "AtLocation"]:
            if 'o' in gtype:
                curg,root,leaves = find_subgraph_star(Gd, curr, (w1,w2), conceptnet_words, Tdict, Tnode_dict, direction="out")
                if curg:
                    Tdict[curr][root] = curg
                    for s in curg.nodes():
                        concept_cols[s]=1
                    gcnt += 1
                    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                    print "%s Graph #%d, rel=%s, root(out) = %s, size=%s" % (tt, gcnt, curr, root, repr([len(leaves)+1, len(curg)]))
                    print [root, leaves]
                    #print curg.nodes()
                    print "-------------  -------------  -------------\n" 
                    write_output_graph(fo, curg, ["%d"%gcnt,'star', curr, root, repr([len(leaves)+1, len(curg)])])
            if 'i' in gtype:
                curg,root,leaves = find_subgraph_star(Gd, curr, (w1,w2), conceptnet_words, Tdict, Tnode_dict, direction="in")
                if curg:
                    Tdict[curr][root] = curg
                    for s in curg.nodes():
                        concept_cols[s]=1
                    gcnt += 1
                    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                    print "%s Graph #%d, rel=%s, root(in) = %s, size=%s" % (tt, gcnt, curr, root, repr([len(leaves)+1, len(curg)]))
                    print [root, leaves]
                    #print curg.nodes()
                    print "-------------  -------------  -------------\n" 
                    write_output_graph(fo, curg, ["%d"%gcnt,'star', curr, root, repr([len(leaves)+1, len(curg)])])
        
            """
                cliques in "ConceptuallyRelatedTo
            """
            curg, wk, cnodes = find_common_clique(Gd, curr, (w1, w2), conceptnet_words, Tdict, Tnode_dict)
            if cnodes:
                Tdict[curr][wk] = curg
                for s in curg.nodes():
                    concept_cols[s]=1
                gcnt += 1
                tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                print "%s Graph #%d, rel=%s, root(clique) = %s, size=%s" % (tt, gcnt, curr, repr(wk), repr([len(cnodes), len(curg)]))
                print cnodes
                #print curg.nodes()
                print "-------------  -------------  -------------\n"
                write_output_graph(fo, curg, ["%d"%gcnt, 'clique', curr, repr(wk), repr([len(cnodes), len(curg)])])
        
        print "\n" # end of a pair
        
    pickle.dump(Tdict, open(os.path.join(opts.db_dir, opts.out_graph_pkl), 'wb') ) 
    fo.close()
    
    print "writing %d words as col/row labels" % len(concept_cols.keys())
    open(os.path.splitext(os.path.join(opts.db_dir, opts.out_graph_txt))[0]+".cols.txt", 'wt').write("\n".join(sorted(concept_cols.keys())) )
    
    """
    print "\nr-combined\ts-combined\tr-bigram\tr-concept\ts-bigram\ts-concept\tword1\tword2\t"
    print "-------------  -------------  -------------\n"
    for j in range(num):
        jb = jn_pair[j]
        w1 = bg_w1[jn_pair[j]]
        w2 = bg_w2[jn_pair[j]]
        js = pair_rscore[jr_pair[j]]
        #jc = cpw12.index( (w1,w2) )
        try:
            jc = cpw12[ (w1,w2) ]
            print "%d\t%3e\t%d\t%d\t%0.1f\t%0.4f\t%s\t%s" % (j, js, jb, jc, bg_fq[jb], cp_fq[jc], w1, w2)
        #print "%s\t%s\t%d\t%d\t%d\t%0.2f\t%0.4f" % (w1, w2, j, jb, jc, bg_fq[jb], cp_fq[jc])
        except:
            print "%d\t%e\t%d\t%0.1f\t %s\t%s" % (j, js, jb, bg_fq[jb], w1, w2)
    """
    return
    
def compute_sim_conceptnet(argv) : #word_dict, cn_words):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='construct+compare conceptnet and flickr word similarities')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db files')
    parser.add_option("", '--db_dict', dest='db_dict', default="dict.db", help='dictionary')
    parser.add_option("", '--addl_vocab', dest='addl_vocab', default="places_etc.txt", help='')
    parser.add_option("", '--svd_dim', dest='svd_dim', type='int', default=150, help='')
    parser.add_option("-K", '--num_pairs', dest='num_pairs', type='int', default=1e6, help='')
    #parser.add_option("", '--bigram_file', dest='bigram_file', default="bigram_filtered.txt", help='')
    (opts, __args) = parser.parse_args(sys.argv)

    # intersect the two dictionaries first
    db_dict = os.path.join(opts.db_dir, opts.db_dict)
    addl_vocab = os.path.join(opts.db_dir, opts.addl_vocab)
        
    cn_words, cn_vocab = get_conceptnet_words(db_dict, addl_vocab)
    # build inverted index (words --> CN terms)
    cn_ii = dict.fromkeys(cn_vocab) 
    for w in cn_words.keys():
        for v in cn_words[w]:
            if not cn_ii[v]:
                cn_ii[v] = [w]
            else:
                cn_ii[v].append(w)
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s finished compiling inverted index for %d words from %d concept-net terms" % (tt, len(cn_words), len(cn_vocab))
    pickle.dump((cn_ii, cn_words), open(os.path.join(opts.db_dir, 'vocab_conceptnet_invertidx.pkl'), 'wb') )
    
    #print "done"
    #return

    # read conceptnet, write the top K similar pairs (by SVD) to file
    A = divisi2.network.conceptnet_matrix('en')
    B = A.normalize_all()
    U, S, _V = B.svd(k = opts.svd_dim)
    sim = divisi2.reconstruct_similarity(U, S, post_normalize=False).to_dense()
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s finished %dD-SVD for %s dimensions" % (tt, opts.svd_dim, repr(sim.shape))

    sim_out = open(os.path.join(opts.db_dir, 'conceptnet_sim_out.txt'), 'wt')

    nv = len(cn_vocab)
    ne = int(.5*nv*(nv-1))
    #vsim = [-.5]*ne #np.empty([.5*nv*(nv-1), 1], dtype=float)
    #vidx = ne*[[1, 2]]
    vcnt = 0
    for i in range(nv):
        for j in range(i):
            #vidx = [i, j]
            vsim = 0

            for wi in cn_ii[cn_vocab[i]]:
                for wj in cn_ii[cn_vocab[j]]:
                    if wi == wj:
                        continue
                    ii = sim.col_index(wi) # index of CN terms
                    jj = sim.col_index(wj)
                    vsim += sim[ii, jj]
            ww = sorted([cn_vocab[i], cn_vocab[j]])
            if vsim > 0:
                sim_out.write("%0.8f\t%s\t%s\n" % (vsim, ww[0], ww[1]) )
            vcnt += 1
            if vcnt % 10000==0:
                tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                print "%s %d / %d similarities computed" % (tt, vcnt, ne)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s Done.  %d / %d similarities computed" % (tt, vcnt, ne)
    sim_out.close()
    return   
         
    """
    # printing everything instead, too slow to compute+sort
    sidx = sorted(vidx, reverse=True, key=lambda i: vsim[i])
    vsim.sort(reverse=True)
    thresh = -1
    oc = 0
    while oc < opts.num_pairs or vsim[oc] >= thresh:
        sim_out.write("%0.5f\t%6d%6d\n" % (vsim[oc], sidx[oc][0], sidx[oc][1]) )
        if oc == opts.num_pairs-1:
            thresh = vsim[oc]
        oc += 1
        if oc % 10000==0:
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s %d / %d similarities printed" % (tt, oc, opts.num_pairs)
    """
    
def ingest_wnet_tag(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='construct+compare conceptnet and flickr word similarities')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db files')
    parser.add_option("", '--db_out', dest='db_out', default="wordnet_tag.db", help='dictionary')
    parser.add_option("-n", '--wnet_list_file', dest='wnet_list_file', default="wnet-50.txt", 
                  help='list of wordnet ids + img cnts with at least 50 flickr imgs associated')
    parser.add_option("-i", '--in_cnts_file', dest='in_cnts_file', default="wordnet_tagsfreq.txt", 
                  help='input file of the format "wnetid  [tag:cnt]"')
    parser.add_option("-w", '--words_file', dest='words_file', default="words.txt", 
                  help='input file of the format "wnetid  words"')
    
    (opts, __args) = parser.parse_args(sys.argv)
    
    in_file = os.path.join(opts.db_dir, opts.in_cnts_file)
    db_file = os.path.join(opts.db_dir, opts.db_out)
    cnt_file = os.path.join(opts.db_dir, opts.wnet_list_file)
    words_file = os.path.join(opts.db_dir, opts.words_file)
    
    cnt_lines = filter(lambda s:len(s), map(lambda s: s.strip(), open(cnt_file, "rt").read().split("\n") ) )
    #print cnt_lines[0:3]
    #print cnt_lines[0].split()
    wn_cnt = dict(map(lambda t: (t.split()[1], int(t.split()[0])), cnt_lines) )
    print "read %d wordnet counts" % (len(wn_cnt))
    
    w_lines = filter(lambda s:len(s), map(lambda s: s.strip(), open(words_file, "rt").read().split("\n") ) )
    #print cnt_lines[0:3]
    #print cnt_lines[0].split()
    wn_words = map(lambda t: (t.split("\t")[0], t.split("\t")[1]), w_lines)
    tmpcnt = len(wn_words)
    wn_words = dict( filter (lambda t: t[0] in wn_cnt, wn_words ) )
    print "read %d wordnet defs, filter down to %d" % (tmpcnt, len(wn_words))
    
    fh = open(in_file, 'rt')
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM wn_tag")
    cursor.execute("DROP TABLE wordnet")
    conn.commit()
    conn.execute('CREATE TABLE "wordnet" (wnid TEXT PRIMARY KEY, words TEXT)')
    conn.commit()
    conn.executemany("INSERT INTO wordnet (wnid, words) VALUES (?,?)", 
                     zip(wn_words.keys(), wn_words.values()) )
    conn.commit()
    cursor.execute("SELECT count(*) FROM wordnet")
    tmpcnt = cursor.fetchone()[0]
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s %d wnet definitions inserted" % (tt, tmpcnt)
        
    lcnt = 0 
    ecnt = 0
    for cl in fh:
        clist = cl.split()
        wn = clist[0]
        tcnt = map(lambda s: (s.split(":")[0], int(s.split(":")[1])), clist[1:]) 
        tmpcnt = len(tcnt)
        tcnt = dict( filter(lambda t: t[1]>1, tcnt) )
        ecnt += (tmpcnt-len(tcnt))
        tfrac = map(lambda k: float(tcnt[k])/wn_cnt[wn], tcnt.keys() )
        conn.executemany("INSERT INTO wn_tag (wnid, tag, count, percentage) VALUES (?,?,?,?)", 
                         zip([wn]*len(tcnt), tcnt.keys(), tcnt.values(), tfrac) )
        lcnt += 1
        if lcnt % 500 == 0:
            conn.commit()
            cursor.execute("SELECT count(*) FROM wn_tag")
            tmpcnt = cursor.fetchone()[0]
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s %d / %d wnet ids inserted, %d tag entries, %d filtered" % (tt, lcnt, len(wn_cnt), tmpcnt, ecnt)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s %d / %d wnet tags inserted" % (tt, lcnt, len(wn_cnt))
    
    #cursor.execute("CREATE INDEX IF NOT EXISTS wnidx1 ON wn_tag(wnid)")
    cursor.execute("CREATE INDEX IF NOT EXISTS tagidx1 ON wn_tag(tag)")
    cursor.execute("CREATE INDEX IF NOT EXISTS tagcnt_idx ON wn_tag(count)")
    cursor.execute("CREATE INDEX IF NOT EXISTS tagprc_idx ON wn_tag(percentage)")
    #cursor.execute("CREATE INDEX IF NOT EXISTS wnidx0 ON wordnet(wnid)")
    conn.commit()
    
    conn.close()
    fh.close()
    
    return

def traverse(graph, start, node):
    node_name = node.name.split('.')[0]
    graph.depth[node_name] = node.shortest_path_distance(start)
    for child in node.hyponyms():
        graph.add_edge(node_name, child.name.split('.')[0]) # [_add-edge]
        traverse(graph, start, child) # [_recursive-traversal]
        
def ingest_tag_struct(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='construct+compare conceptnet and flickr word similarities')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db files')
    parser.add_option("", '--db_out', dest='db_out', default="taginfo.db", help='dictionary')
    
    (opts, __args) = parser.parse_args(sys.argv)
    
    db_file = os.path.join(opts.db_dir, opts.db_out)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    pos_list = ["n", "v", "a", "r"]
    """
        G = {}
        root_list = {}
        for p in pos_list:
            G[p] = nx.DiGraph() 
            G[p].depth = {}
            root_list[p] = []
    """ 
    
    cursor.execute("DROP TABLE tag_wn")
    cursor.execute("CREATE TABLE tag_wn (tagid INTEGER,  pos TEXT,  min_depth INTEGER,  max_depth INTEGER)")
    conn.commit()
    
    tlist =  cursor.execute("SELECT id, tag FROM tag_score").fetchall()
    tcnt = 0
    for ti, tt in tlist : #cursor.execute("SELECT id, tag FROM tag_score"):
        #ti = row[0]
        #tt = row[1]
        for p in pos_list:
            syn = wn.synsets(tt, pos=p)
            if syn:
                min_d = min(map(lambda s: s.min_depth(), syn))
                max_d = max(map(lambda s: s.min_depth(), syn))
                cursor.execute("INSERT INTO tag_wn (tagid, pos, min_depth, max_depth) VALUES (?,?,?,?)", (ti,p,min_d,max_d))
            else:
                pass
                #print '  empty synset for "%s", "%s"' %(tt, p)
        
        tcnt += 1
        if tcnt % 1000 == 0:
            conn.commit()
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s inserted %d tags " % (tt, tcnt)
        
        """
        for p in pos_list:
            for s in wn.synsets(tt, pos=p):
                for h in s.root_hypernyms():
                    if h.name not in root_list[p]:
                        traverse(G[p], h, h)
                        root_list[p].append(h.name)
                    else:
                        pass # root already exist
        """
    
    conn.commit()
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')          
    print "%s inserted %d tags, Done." % (tt, tcnt)
    conn.close()
    
    return
    
if __name__ == '__main__':
    argv = sys.argv 
    if '--make_vocab' in argv:
        argv.remove('--make_vocab')
        make_vocab(argv)
    elif '--compute_sim_conceptnet' in argv:
        argv.remove('--compute_sim_conceptnet')
        compute_sim_conceptnet(argv)
    elif '--get_top_bigrams' in argv:
        argv.remove('--get_top_bigrams')
        get_top_bigrams(argv)
        #elif '--norm_vocab' in argv:
        #    argv.remove('--norm_vocab')
        #   get_top_bigrams(argv)    
    elif '--ingest_wnet_tag' in argv:
        argv.remove('--ingest_wnet_tag')
        ingest_wnet_tag(argv)
    elif '--ingest_tag_struct' in argv:
        argv.remove('--ingest_tag_struct')
        ingest_tag_struct(argv)
    else:
        pass        