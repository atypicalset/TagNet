import os
#from datetime import datetime
#import sqlite3
from pysqlite2 import dbapi2 as sqlite3

data_dir = '/Users/xlx/proj/ImageNet/db'
in_words_file = 'words.txt'
in_gloss_file = "gloss.txt"
wnet_13k = "wnet-50.txt"
sqlite_db_file = 'wordnet_fts.db'

""" store to SQlite """
conn = sqlite3.connect(os.path.join(data_dir, sqlite_db_file))
cur = conn.cursor()

conn.execute("""DROP TABLE wordnet""")
#conn.execute("""DELETE FROM wordnet_word""")
conn.execute("""CREATE VIRTUAL TABLE wordnet USING fts3(wnid, words, gloss);""")
conn.commit()

wn_list = map(lambda s: s.strip().split()[1], open(os.path.join(data_dir, wnet_13k), 'rt'))

gdict = {}
for cl in open(os.path.join(data_dir, in_gloss_file), 'rt'):
    tmp = cl.split('\t')
    assert tmp[0] not in gdict, "duplicate wnid in gloss! "
    if tmp[0] in wn_list:
        gdict[tmp[0]] = tmp[1].strip()

print " load %d gloss items" % len(gdict)

for out in conn.execute("""SELECT COUNT(*) FROM wordnet""", ):
    print out
    
conn.executemany("INSERT INTO wordnet (wnid,gloss) values (?,?)", gdict.items())
conn.commit()

for out in conn.execute("""SELECT COUNT(*) FROM wordnet""", ):
    print out
    
wdict = {}
for cl in open(os.path.join(data_dir, in_words_file), "rt"):
    tmp = cl.split('\t')
    assert tmp[0] not in wdict, "duplicate wnid! "
    if tmp[0] in wn_list:
        w = tmp[0]
        wdict[w] = map(lambda s: s.strip(), tmp[1].strip().split(","))
        cur.execute("UPDATE wordnet SET words=? WHERE wnid=?", (','.join(wdict[w]), w) )

print " load %d words" % len(wdict)
    
#conn.executemany("UPDATE wordnet SET words=? WHERE wnid=?",
#                (map(lambda w: ','.join(wdict[w]), wdict.keys()), wdict.keys() ) ) 

conn.commit()

conn.close()

"""
wcnt = 0
for w, wlist in wdict.iteritems():
    conn.executemany("INSERT INTO wordnet_word (wnid,word) values (?,?)", 
                     zip([w]*len(wlist), wlist) )
    wcnt += 1
    if wcnt%100==0:
        conn.commit()

print "inserted %d words with %d wnid" % (reduce(lambda x, y:x+y, map(len,wdict.values())), wcnt)
"""



