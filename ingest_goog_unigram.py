
import os
from datetime import datetime
import sqlite3
import zipfile
import glob
import re
import codecs
 
year_range = [2004, 2008]
data_home = '/home/users/xlx/vault-xlx/goog-ngram'
data_set = 'eng-1m'

db_dir = '/home/users/xlx/vault-xlx/proj'
sqlite_db_file = 'imgnet.db'

wpat = re.compile('^[a-zA-Z]')

""" store to SQlite """
conn = sqlite3.connect(os.path.join(db_dir, sqlite_db_file))
conn.execute("""DELETE FROM google_wordcount""")
conn.commit

file_list = glob.glob(os.path.join(data_home, data_set, 'googlebooks-eng-*.zip'))
cnt = 0

for cur_file in file_list:
    tt =  datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processing %s" % (tt, cur_file)
    try:
        root = zipfile.ZipFile(cur_file, "r")
    except:
        root = "."
      
    word=u''; mw=0; pw=0; vw=0
    for name in root.namelist():
        """ extract the file to disk """
        tt =  datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s extracting %s ..." % (tt, name)
        
        ename = os.path.join(data_home, data_set, name)
        root.extract(name, os.path.join(data_home, data_set))
        lcnt = 0 # line count in the current file
        try:
            for cl in codecs.open(ename, encoding='utf-8'):
                lcnt += 1
                tmp = cl.strip().split('\t')
                try:
                    if not tmp[0]==word:
                        """ commit current result """
                        if word and mw>2 and wpat.search(word) :
                            conn.execute("INSERT INTO google_wordcount (word,match_count,page_count,volume_count) VALUES (?,?,?,?)", 
                                         (word, mw, pw, vw) )
                            cnt += 1
                            if cnt%5000 == 0:
                                conn.commit()
                                tt =  datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                                print "%s %8d words recorded. last entry (%s,%d,%d,%d)" % (tt, cnt, word, mw, pw, vw )
                        mw = 0 # matchin count
                        pw = 0 # page count
                        vw = 0 # volume count
                        word = tmp[0]
                except:
                    print tmp
                    print word, mw, pw, vw
                    print cnt
                    
                if int(tmp[1])>=2004:
                    mw += int(tmp[2])
                    pw += int(tmp[3])
                    vw += int(tmp[4])
        except:
            tt =  datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s error reading %s, at line# %d, %s" % (tt, ename, lcnt, cl)
            
        tt =  datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s done. %d words so far, remove file %s" % (tt, cnt, ename)
        os.remove(ename)
        
    #break ## DEBUG use, only proc the first file