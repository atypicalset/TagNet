import sys, os
import re
import random
from datetime import datetime
from optparse import OptionParser
import sqlite3

from imgtags import get_substree, get_imgnet_url, cache_flickr_info

DEBUG = 1
FLICKR_XML_DIR = '/data1/vault-xlx/imgnet-flickr/json'

DB_DIR = '/home/users/xlx/vault-xlx/proj'
#DB_DIR = '/home/users/xlx/db'
DB_FILE = 'imgnet.db'

def ingest_flickr_info(url_list, conn, argv):
    
    parser = OptionParser(description='return co-occurring tag counts for a given list of flickr URLs')
    parser.add_option('-w', '--wordnet_id', dest='wordnet_id', default='', help='current wnid')
    parser.add_option('-j', '--json_dir', dest='json_dir', default=FLICKR_XML_DIR, 
                    help='dir to cache json metadata of each photo')
    parser.add_option('-k', '--flickr_key_file', dest='flickr_key_file', 
        default='flickr.key.txt', help='file containing a list of API keys, one per line')
    """ 'http://static.flickr.com/2088/[id]_94dbc23839.jpg' """
    parser.add_option("-p", "--id_pattern", dest="id_pattern", 
        default="[^/]*//.*/[0-9]*/(?P<flickrid>[0-9]*)\_([0-9a-z]*).*", help="regexp to get flickr id")
    (opts, __args) = parser.parse_args(argv)
    
    api_keys = open(opts.flickr_key_file, 'r').read().split()
    api_keys = map(lambda s: s.strip(), api_keys)
    id_pattern = re.compile(opts.id_pattern)
    
    csr = conn.cursor()
    
    icnt = 0
    ist_cnt = 0
    img_nct = 0
    fail_cnt = 0
    id_list = []
    for cur_u in url_list:
        try:
            m = id_pattern.match(cur_u)
            imgid = m.group('flickrid')
        except:
            print "\t err parsing URL" + cur_u
            continue
        icnt += 1        
        
        cur_key = api_keys [random.randint(0, len(api_keys)-1)]
        jinfo = cache_flickr_info(imgid, cur_key, rootdir=opts.json_dir)
        
        flickrid = int(imgid)
        id_list.append(flickrid)
        wnf_exist = -1
        csr.execute("SELECT COUNT(*) FROM imagenet_flickr WHERE wnid=? AND flickrid=?", 
                    (opts.wordnet_id, flickrid))
        for row in csr:
            wnf_exist = int(row[0])
                
        if 'stat' not in jinfo or not jinfo['stat']=='ok' :
            fail_cnt += 1
            if wnf_exist>0:
                csr.execute("INSERT INTO imagenet_flickr (wnid, flickrid,status) VALUES (?,?,?)", 
                         (opts.wordnet_id, flickrid, 0))
            if icnt%100 == 0 or DEBUG>2:
                tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                print "%s %5d/%5d flickr img not found: %s" % (tt, icnt, len(url_list), flickrid)
                
        else:
            if not wnf_exist==1:
                ist_cnt += 1
                csr.execute("INSERT INTO imagenet_flickr (wnid, flickrid,status) VALUES (?,?,?)", 
                         (opts.wordnet_id, flickrid, 1))
                
                img_exist = 0
                csr.execute("SELECT COUNT(*) FROM flickr_info WHERE flickrid=%d" % flickrid )            
                for row in csr:
                    img_exist = int(row[0])
                    
                """
                    INSERT OR REPLACE INTO Employee (id,name,role) 
                      VALUES (  1, 
                                'Susan Bar',
                                coalesce((select name from Employee where id = 1),'Benchwarmer')
                              );
                """    
                
                if not img_exist:
                    img_nct += 1
                    pinfo = jinfo["photo"]
                    userid = pinfo["owner"]["nsid"]
                    taken_time = pinfo["dates"]["taken"]
                    image_url = cur_u
                    title = pinfo["title"]["_content"]
                    description = pinfo["description"]["_content"]
                    
                    finfo = (flickrid,userid,taken_time,image_url,title,description)
                    csr.execute("INSERT INTO flickr_info VALUES (?,?,?,?,?,?)", finfo)
                    
                    tmp = pinfo["tags"]["tag"]
                    tg = map(lambda s: s["_content"], tmp)
                    if tg:                    
                        csr.executemany("INSERT INTO flickr_tag (flickrid,tag) VALUES (?,?)", 
                                        zip([flickrid]*len(tg), tg))
                else:
                    print "  img # %d already exist: %d" % (icnt, flickrid)
                    
                if DEBUG>2:
                    print icnt, opts.wordnet_id, flickrid
                    print repr(finfo)
                    print repr( zip([flickrid]*len(tg), tg) )
                    print ""
            else:
                # what case is this?
                print "  entry already exist:"
                csr.execute("SELECT * FROM imagenet_flickr WHERE wnid=? AND flickrid=?", 
                    (opts.wordnet_id, flickrid))            
                for row in csr:
                    print "  " + repr(row)
                #csr.execute("SELECT * FROM flickr_info WHERE flickrid=%d" % flickrid)
                #for row in csr:
                #    print row
                #print flickrid in id_list
                
                
            
        if icnt%100 == 0: 
            conn.commit()
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s %d/%d urls processed, %d new records, %d new images, %d failed" \
                % (tt, icnt, len(url_list), ist_cnt, img_nct, fail_cnt)
            
                
    conn.commit()
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s %d/%d urls processed, %d new records, %d new images, %d failed \n\n" \
        % (tt, icnt, len(url_list), ist_cnt, img_nct, fail_cnt)
        
    return

if __name__ == '__main__':  
    """ store to SQlite """
    conn = sqlite3.connect(os.path.join(DB_DIR, DB_FILE))
    #conn.execute("""DELETE FROM google_wordcount""")
    #conn.commit
    wlist = get_substree(sys.argv[1])
    wcnt = 0
    for ww in wlist:
        wcnt += 1
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s Processing %d of %d synsets" % (tt, wcnt, len(wlist))
         
        url_list,n_allurl = get_imgnet_url(['-w', ww]) #'n09470027' fast debug # 'n09421951' sandbar
        if len(sys.argv)<=2:
            tag_cnt = ingest_flickr_info(url_list, conn, ['-w', ww])
        else:
            tag_cnt = ingest_flickr_info(url_list, conn, ['-w', ww, '-j', sys.argv[1]])
        #break
   
    conn.commit()
    conn.close() 

        