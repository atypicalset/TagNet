
import divisi2
import networkx as nx
#import matplotlib.pyplot as plt
import pickle
import os

A = divisi2.network.conceptnet_matrix('en')
wa = A.row_labels
wb = A.col_labels
# wa and wb are ordered sets
# wb -- OrderedSet([('right', u'AtLocation', u'wood'), ('left', u'AtLocation', u'fawn'), ('right', u'IsA', u'deer')])

all_rel = map(lambda s: s[1], list(wb))
node_cnt_dict = dict.fromkeys(wa)

rel_cnt_dict = dict.fromkeys(all_rel)
Gd = dict.fromkeys(all_rel)

for ik in A.keys():
    """
    A.keys() Returns a list of tuples, giving the indices of non-zero entries.
    A.keys()[:5] = [(0, 0), (0, 2), (0, 4), (0, 5), (0, 7)]
    """
    v = A[ik[0],ik[1]]
    r = wb[ik[1]][1]
    a = wa[ik[0]]
    b = wb[ik[1]][2]
    
    if v < 0:
        continue
    
    if not Gd[r]:
        Gd[r] = nx.DiGraph()
        rel_cnt_dict[r] = 0
        
    #print ik, a, b, r, node_cnt_dict
        
    if not node_cnt_dict[a]:
        node_cnt_dict[a] = 0
    #if not node_cnt_dict[b]:
    #    node_cnt_dict[b] = [0, 0]
    
    if wb[ik[1]][0]=='right': 
        # A is symmetric w.r.t. left and right relations
        rel_cnt_dict[r] += 1
        node_cnt_dict[a] += 1
        Gd[r].add_edge(b, a, weight=v)
    """
    if a==u'apple' or b==u'apple':
        print a + ", " + repr(wb[ik[1]])
     """
     
print "" 
print rel_cnt_dict   
print sum(rel_cnt_dict.values()) 
print ""
#print node_cnt_dict

for r in Gd.keys():
    if rel_cnt_dict[r]<500:
        del rel_cnt_dict[r]

for r in rel_cnt_dict.keys():
    #gc = nx.connected_components(Gd[r])
    #print ' R="%s", graph nodes %d, edges %d, components %d ' % (r, len(Gd[r]), Gd[r].size(), len(gc))
    #print '\t sizes [%s]' % repr(map(lambda x:len(x), gc) )
    gml_file = os.path.join('/Users/xlx/tmp', r+'.graphml')
    Gc = Gd[r]
    nx.write_graphml(Gc, gml_file)
    """
    if r == u"AtLocation":
        es = Gd[r].edges()
        rlabel = list(set(map(lambda e: e[0], es)))
        clabel = list(set(map(lambda e: e[1], es)))
    """
    
print "total number of edges %d" % sum(map(lambda r:Gd[r].size(), Gd.keys()))

pickle.dump(Gd, open('/Users/xlx/proj/ImageNet/db2/Gd.pkl', 'wb'))

"""
print ""
print len(rlabel)
print rlabel
print ""
print len(clabel)
print clabel
"""

#nx.draw_spring(Gd[u'DefinedAs']); plt.show()

"""
nx.draw(G)
>>> nx.draw_random(G)
>>> nx.draw_circular(G)
>>> nx.draw_spectral(G)    
"""
