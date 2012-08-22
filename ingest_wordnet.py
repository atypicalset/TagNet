import os
#from datetime import datetime
import sqlite3

data_dir = '/Users/xlx/proj/ImageNet'
in_words_file = 'words.txt'
in_gloss_file = "gloss.txt"
sqlite_db_file = 'imgnet.db'

""" store to SQlite """
conn = sqlite3.connect(os.path.join(data_dir, sqlite_db_file))
conn.execute("""DELETE FROM wordnet""")
conn.execute("""DELETE FROM wordnet_word""")

gdict = {}
for cl in open(os.path.join(data_dir, in_gloss_file), 'rt'):
    tmp = cl.split('\t')
    assert tmp[0] not in gdict, "duplicate wnid in gloss! "
    gdict[tmp[0]] = tmp[1].strip()

print " load %d gloss items" % len(gdict)


for out in conn.execute("""SELECT COUNT(*) FROM wordnet""", ):
    print out
    
conn.executemany("INSERT INTO wordnet (wnid,gloss) values (?,?)", gdict.items())
conn.commit()

wdict = {}
for cl in open(os.path.join(data_dir, in_words_file), "rt"):
    tmp = cl.split('\t')
    assert tmp[0] not in wdict, "duplicate wnid! "
    wdict[tmp[0]] = map(lambda s: s.strip(), tmp[1].strip().split(","))
print " load %d words" % len(wdict)
    
conn.executemany("UPDATE wordnet SET word1=?, allwords=? WHERE wnid=?", 
                zip(map(lambda w:wdict[w][0], wdict.keys()), 
                    map(lambda w:u','.join(wdict[w]), wdict.keys()), wdict.keys()) )

conn.commit()


wcnt = 0
for w, wlist in wdict.iteritems():
    conn.executemany("INSERT INTO wordnet_word (wnid,word) values (?,?)", 
                     zip([w]*len(wlist), wlist) )
    wcnt += 1
    if wcnt%100==0:
        conn.commit()

print "inserted %d words with %d wnid" % (reduce(lambda x, y:x+y, map(len,wdict.values())), wcnt)

conn.close()


