import sys, os
import random
import sqlite3
import urllib2
from glob import glob
from datetime import datetime
from time import sleep
from optparse import OptionParser


def download_synset_flickr(synset_file, img_root_dir, hash_level=2, chars_per_hash=2):
    
    ss = os.sep
    exist_cnt = 0
    err_cnt = 0
    good_cnt = 0
    cnt = 0
    syn_lines = open(synset_file, 'rt').read()
    syn_lines = syn_lines.split("\n")
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processing %d urls from %s"  % (tt, len(syn_lines), os.path.split(synset_file)[1] )
    for line in syn_lines:
        if not line.strip() or len(line.strip().split())<3:
            continue
        tmp = line.strip().split()
        imgid = tmp[1]
        imgurl = tmp[2]
        cnt += 1
        
        curs = 0
        cure = chars_per_hash
        hdir = []
        for i in range(hash_level):
            curs += i*chars_per_hash
            cure = curs + chars_per_hash
            hdir.append(imgid[curs:cure])
        outdir = os.path.join(img_root_dir, ss.join(hdir))
        imfile = os.path.join(outdir, imgid+".jpg")
        
        if not os.path.exists(imfile):
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            try:
                buf = urllib2.urlopen(imgurl).read()
                fh = open(imfile, 'wb')
                fh.write(buf)
                fh.close()
                good_cnt += 1
            except:
                print "  ERR downloading img %s from %s" % (imgid, imgurl)
                err_cnt += 1
        else:
            exist_cnt += 1
        
        if cnt % 100 == 0:
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s processed %d urls: %d new, %d exist, %d err \n\t"  % (tt, cnt, good_cnt, exist_cnt, err_cnt)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processed %d urls: %d new, %d exist, %d err \n\t from %s"  % (tt, cnt, good_cnt, exist_cnt, err_cnt, synset_file)
    
def db_try_out(stmt, cur, args=[], quiet=0):
    num_tries = 0
    success = False
    t0 = datetime.now()
    print "%s trying sqlite3: %s" % (datetime.strftime(t0, '%Y-%m-%d %H:%M:%S'), stmt)
    while num_tries<10 and (not success):
        num_tries += 1
        try:
            if not args:
                cur.execute(stmt)
            else:
                cur.excute(stmt, args)
            success = True
        except Exception, e:            
            twait = 1.1 #random.randint(1, 5)
            if 1: #not quiet:
                print "  sql-fail# %d, msg %s. waiting %0.1f seconds .." % (num_tries, str(e), twait)
                print '  stmt: "%s", %s' % (stmt, str(args))
            sleep(twait)
    
    if not success:
        dt = datetime.now() - t0
        dt = dt.total_seconds()
        print "Failed accessing DB after %d tries and %0.3f seconds" % (num_tries, dt)
        raise
    else:
        if not quiet:
            print "%s sqlite3: ok" % datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    
    return
            

def check_status(wnid, db_file, request='CHECKOUT'):
    """ store to SQlite """
    conn = sqlite3.connect(os.path.join(db_file))
    cur = conn.cursor()
    stmt = "SELECT working,done FROM synset_status WHERE wnid='%s'"% wnid
    #print stmt 
    db_try_out(stmt, cur)
    row = cur.fetchone()
    
    if not row:
        if request=='CHECKOUT':
            #cur.execute('INSERT INTO synset_status (wnid, working,done) VALUES (?,?,?)', (wnid, 1, 0))
            db_try_out("INSERT INTO synset_status (wnid,working,done) VALUES ('%s',%d,%d)"%(wnid, 1, 0), cur)
            conn.commit()
            req_status = 1
        elif request=='CHECKIN':
            req_status = -1
            print "ERR checking in wnid=%s, Status does not exit!" % wnid
        else:
            req_status = -1
            print "ERR unkonwn request " + request
    else:
        working=int(row[0])
        done = int(row[1])
        if request=='CHECKOUT' and (working or done) :
            req_status = 0 # already worked on, move on
        elif request=='CHECKOUT' :
            req_status = -1
            print "ERR checkout item already exist, status invalid (%s,%d,%d) " % (wnid, working, done)
        elif request=='CHECKIN':
            if working==1 and done==0:   
                stmt = "UPDATE synset_status SET working=%d, done=%d where wnid='%s'" % (0, 1, wnid)
                db_try_out(stmt, cur)
                #cur.execute(stmt)
                conn.commit()
                req_status = 1 # SUCCESS checking in
                #cur.execute('SELECT count(*), sum(working), sum(done) FROM synset_status' )
                stmt = 'SELECT count(*), sum(working), sum(done) FROM synset_status'
                db_try_out(stmt, cur)
                row = cur.fetchone()
                print " %s CHECKED IN. DB status (total, working, done) = %s" % (wnid, repr(row))
            else:
                print "ERR checkin, status invalid (%s,%d,%d) " % (wnid, working, done)
                req_status = -1
        else:
            req_status = -1
            print "ERR unkonwn request " + request
    
    
    conn.close()
    return req_status
    

if __name__ == '__main__':  
    argv = sys.argv
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='download flickr images for imagenet')
    parser.add_option("-i", "--in_dir", dest="in_dir", 
        default='', help="input dir containing wnid-flickrid mappings")
    parser.add_option('-o', '--img_root_dir', dest='img_root_dir', 
        default='', help='root dir where images should be stored')
    parser.add_option("-d", "--db_file", dest="db_file", default="flickr_na.jpg", 
                      help = "path to example flickr NOTFOUND img")
    parser.add_option("-t", "--time_limit", dest="time_limit", type='float', default=2., 
                      help = "time limit in hours")
    parser.add_option("-c", "--clean_ongoing_synsets", dest="clean_ongoing_synsets", action='store_true',  
                      default=False, help = "clean up unfinished synsets from db (need to be in single process) ")
    opts, __ = parser.parse_args(argv)
    
    fi = 0
    ti = datetime.now()
    dt = 0
    dcnt = 0
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    if opts.clean_ongoing_synsets:
        conn = sqlite3.connect(opts.db_file); cur=conn.cursor()
        db_try_out("SELECT wnid FROM synset_status WHERE working=1 AND done=0", cur)
        list_todo = map(lambda r: r[0], cur)
        conn.close()
        print "Found %d unfinished synsets " % len(list_todo)
        print list_todo 
        print ""
        for wnid in list_todo:
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s processing %d of %d synsets: %s" % (tt, fi+1, len(list_todo), wnid)
            
            wn_file = os.path.join(opts.in_dir, wnid+".txt")
            download_synset_flickr(wn_file, opts.img_root_dir)
            status = check_status(wnid, opts.db_file, request='CHECKIN')
            assert status==1, "db status err! for %s" % wnid
                
            dd = datetime.now() - ti
            #dt = dd.total_seconds()
            dt = (dd.microseconds + 1.*(dd.seconds + dd.days * 24 * 3600) * 10**6) / 10**6
            fi += 1
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s elapsed time %0.4f seconds \n" % (tt, dt)
    
    else:
        all_files = glob(os.path.join(opts.in_dir,"*.txt"))
        random.shuffle(all_files)
        wn_list = map(lambda s: os.path.splitext(os.path.split(s)[1])[0], all_files)
        """ see + store how many are already taken """
        conn = sqlite3.connect(opts.db_file); cur=conn.cursor()
        #cur.execute("SELECT wnid FROM synset_status WHERE working=1 OR done=1")
        db_try_out("SELECT wnid FROM synset_status WHERE working=1 OR done=1", cur)
        list_taken = map(lambda r: r[0], cur)
        conn.close()
        print "%d / %d synsets already taken \n" % (len(list_taken), len(wn_list))
        
        while fi < len(all_files) and dt < (opts.time_limit-1)*3600:
            wnid = wn_list[fi]
            if wnid in list_taken:
                print " %s in the list taken, skip" % wnid
                fi += 1
                continue # skip this
            
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s processing %d of %d synsets from %s" % (tt, fi+1, len(all_files), all_files[fi])
            #wnid = os.path.splitext(os.path.split(all_files[fi])[1])[0]
            
            status = check_status(wnid, opts.db_file, request='CHECKOUT')
            if status ==0:
                print "\t synset %s already worked on, skip to the next" % wnid
            elif status == 1:
                download_synset_flickr(all_files[fi], opts.img_root_dir)
                sleep(0.1) # do nothing
                status = check_status(wnid, opts.db_file, request='CHECKIN')
                assert status==1, "db status err! for %s" % wnid
                dcnt += 1
            else:
                assert 0, "\t synset %s DB status err! exist" % wnid
                
            dd = datetime.now() - ti
            #dt = dd.total_seconds()
            dt = (dd.microseconds + 1.*(dd.seconds + dd.days * 24 * 3600) * 10**6) / 10**6
            fi += 1
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s elapsed time %0.4f seconds \n" % (tt, dt)
    
    print "%s quit. num-synsets-processed %d, downloaded %d, time used %0.3f hours \n" % (tt, fi, dcnt, dt/3600)
        
    
    