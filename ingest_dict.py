import os
import sqlite3
import string
import re

data_dir = '/Users/xlx/proj/ImageNet/word-list'
base_dict_file = ['12dicts-5.0/2+2lemma.txt'] #, '12dicts-5.0/neol2007.txt']
other_list_file = ['list-UWL.txt', 'list-GSL.txt', 'list-BE1500.txt', 'list_VOA.txt']
other_list_var = ['from_UWL', 'from_GSL', 'from_BE1500', 'from_VOA']
stopword_file = "stoplist-SEO.txt"

sqlite_db_file = 'dict.db'

""" store to SQlite """
conn = sqlite3.connect(os.path.join(data_dir, sqlite_db_file))
conn.execute("""DELETE FROM dict""")
conn.commit()
cur = conn.cursor()

""" read a list of stop words """
stop_list = open(os.path.join(data_dir, stopword_file), 'rt').read().split()

wordpat = re.compile('^[a-zA-Z]*')
alpha_pat = re.compile('^[a-zA-Z]*$')

"""
    parse 2+2lemma-style word lists, as in http://wordlist.sourceforge.net/12dicts-readme-r5.html
"""
wcnt = 0
ecnt = 0
for bd in base_dict_file:
    cur_base = ''
    for bl in open(os.path.join(data_dir, bd), 'rt'):
        if bl[0] not in string.whitespace:
            cur_base = bl.split()[0].strip()
            cur_base = wordpat.match(cur_base).group(0) # filter out symbols 
            cw = [cur_base]
            is_base = [1]
        else:
            cw = map(lambda s: s.strip(), bl.split(',')) # list of non-base words
            cw = map(lambda s: wordpat.match(s).group(0), cw) # filter out symbols 
            assert cur_base, "error: current wordbase cannot be empty!" % bl
            is_base = [0]*len(cw)
        
        if len(cur_base)<2:
            continue # ignore single-letter words
        
        cb = [cur_base]*len(cw)
        is_stop = [cur_base in stop_list]*len(cw)
        from_22lemma = [1]*len(cw)
        wcnt += len(cw)
        """ commit to SQLite """
        for ct in zip(cw, cb, is_base, is_stop, from_22lemma):
            try:
                cur.execute("INSERT INTO dict (string, word, is_base, is_stopword, from_22lemma) VALUES (?,?,?,?,?)", ct)
            except:
                conn.commit()
                print ct
                #print ct[0]
                for row in cur.execute("SELECT string,word FROM dict WHERE string='%s' " % ct[0]):
                    print row
                
                ecnt += 1
        
conn.commit()
print " %d words read, %d errs" % (wcnt, ecnt)

"""
    parse various other lists
"""     
for ofile, ofield in zip(other_list_file, other_list_var):
    new_cnt = 0
    update_cnt = 0
    words = open(os.path.join(data_dir, ofile), 'rt').read().split()
    for cl in words:
        cw = cl.strip().lower() # the current word
        if len(cw)<2 or alpha_pat.match(cw) is None:
            continue # skip compound words and so on
        
        is_stop = cw in stop_list
        cur.execute("SELECT string,word FROM dict WHERE string='%s' " % cw)
        row = cur.fetchall()
        if row: # update the entry
            cur.execute("UPDATE dict SET %s=1 WHERE string='%s'" % (ofield, cw))
            update_cnt += 1
        else: # insert this word
            cur.execute("INSERT INTO dict (string, word, is_base, is_stopword, %s) VALUES (?,?,?,?,?)" % ofield, 
                        (cw, cw, 1, is_stop, 1))
            new_cnt += 1
            print "\t '%s' is_stop=%d " % (cw, is_stop)
    
    print " done processing %s. %d new, %d updated, %d total" % (ofile, new_cnt, update_cnt, len(words))
conn.commit()

field_str = "count(*), sum(from_VOA), sum(from_UWL), sum(from_GSL), sum(from_BE1500), sum(from_22lemma), sum(is_stopword), sum(is_base) "
cur.execute("SELECT "+field_str+" FROM dict")

print "\n"+field_str+"=\n\t"+str(cur.fetchall())
conn.close()