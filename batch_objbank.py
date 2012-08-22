import sys, os
import re
from datetime import datetime
import sqlite3
from glob import glob 
#from random import random
from optparse import OptionParser
from download_synsets_flickr import db_try_out

DB_FILE_NAME="proc_objbank.db"

""" pick new dirs from db, create jobs + qsub """
def run_obj_job(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='count img files')
    parser.add_option('-i', '--in_dir', dest='in_dir', default="", help='root dir of input img files')
    parser.add_option('-o', '--out_dir', dest='out_dir', default="", help='output dir containing feature files')
    parser.add_option('-n', '--num_job', dest='num_job', type='int', 
                      default=50, help='number of qsub jobs to generate')
    
    parser.add_option('-j', '--job_file', dest='job_file', default="/home/659/lxx659/short/qsub-jobs/objbank.job", 
                      help='output job script (temporary)')
    parser.add_option('-t', '--job_template', dest='job_template', default="/home/659/lxx659/short/qsub-jobs/test_ob.job", 
                      help='default: test_ob.db')
    parser.add_option('', '--db_file', dest='db_file', default=DB_FILE_NAME, help='default: proc_objbank.db')
    parser.add_option("-m", '--max_imgs', dest='max_imgs', type="int", default=1500,
                      help='number of imgs to pack in a job')

    
    parser.add_option("", '--num_models', dest='num_models', type="int", default=177,
                      help='number of model ran on data')
    (opts, __args) = parser.parse_args(sys.argv)
    
    job_script = open(opts.job_template,'r').read().split("\n")
    
    db_file = os.path.join(os.path.split(opts.out_dir)[0], opts.db_file)
    print db_file
    conn = sqlite3.connect(db_file); cur=conn.cursor()
    cur.execute("SELECT dir_name,num_imgs FROM status WHERE done==-1 ORDER BY dir_name")
    list_todo = dict(map(lambda r: [r[0], r[1]], cur))
    lkey = sorted(list_todo.keys())
    cur_imgcnt = 0
    cur_dlist = []
    cur_script = job_script[:10] # header info
    job_cnt = 0
    curp = os.getcwd()
    
    for curd in lkey:
        if cur_imgcnt + list_todo[curd] > opts.max_imgs:
            #print "\n".join(cur_script)
            open(opts.job_file, "w").write("\n".join(cur_script))
            os.chdir(os.path.split(opts.job_file)[0])
            os.system("qsub %s" % opts.job_file)
            os.chdir(curp)
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s job with %d images from %d dirs: %s" % (tt, cur_imgcnt, len(cur_dlist), str(cur_dlist))
            job_cnt += 1
            if job_cnt >= opts.num_job:
                break
            cur_imgcnt = 0
            cur_dlist = []
            cur_script = job_script[:10] # header info
        
        cur_out = os.path.join(opts.out_dir, curd)
        if not os.path.exists(cur_out):
            print 'mkdir '+ cur_out 
            os.makedirs(cur_out)
        cur_script += ["curdir=%s" % curd]
        cur_script += [job_script[11] + "\n"] # the cmd line
        
        print "UPDATE status SET done=0 WHERE dir_name='%s'" % curd
        cur.execute("UPDATE status SET done=0 WHERE dir_name='%s'" % curd )
        conn.commit()
        
        cur_dlist += [curd]
        cur_imgcnt += list_todo[curd]
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s done. submitted %d jobs\n" % (tt, job_cnt)
    conn.close()
    
""" NOT USED """    
def mkdir_objbank(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='count img files')
    parser.add_option('-i', '--in_dir', dest='in_dir', default="", help='root dir of input img files')
    parser.add_option('-f', '--feat_dir', dest='feat_dir', default="", help='dir containing feature files')
    parser.add_option('-n', '--num_items', dest='num_items', default="", help='dir containing feature files')
    
    parser.add_option('-j', '--job_file', dest='job_file', default="~/short/qsub-jobs/objbank.job", help='default: proc_objbank.db')
    parser.add_option('', '--db_file', dest='db_file', default=DB_FILE_NAME, help='default: proc_objbank.db')
    parser.add_option("", '--num_models', dest='num_models', type="int", default=177,
                      help='number of model ran on data')
    (opts, __args) = parser.parse_args(sys.argv)
    """
    db_file = os.path.join(os.path.split(opts.in_dir)[0], opts.db_file)
    conn = sqlite3.connect(db_file); cur=conn.cursor()
    cur.execute("SELECT dir_name FROM status WHERE done>-1")
    list_taken = map(lambda r: r[0], cur)
    """
    for pp, _dd, _ff in os.walk(opts.in_dir):
        dname = pp.replace(opts.in_dir, "").strip(os.sep)
        
        cur_out = os.path.join(opts.feat_dir, dname)
        if not os.path.exists(cur_out):
            print 'mkdir '+ cur_out 
            os.makedirs(cur_out)
        cmd = './OBmain %s %s' % (pp+os.sep, cur_out+os.sep)
        #print cmd

    #conn.close()
""" NOT USED """

def collect_img_feat(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='count feature files')
    #parser.add_option('-i', '--in_dir', dest='in_dir', default="", help='root dir of input img files')
    parser.add_option('-f', '--feat_dir', dest='feat_dir', default="", help='dir containing feature files')
    parser.add_option('', '--db_file', dest='db_file', default=DB_FILE_NAME, help='default: proc_objbank.db')
    #parser.add_option("", '--num_models', dest='num_models', type="int", default=177,
    #                  help='number of model ran on data')
    (opts, __args) = parser.parse_args(sys.argv)
    
    #is_feat = re.compile("^[0-9]*.jpg.feat$")
    db_file = os.path.join(os.path.split(opts.in_dir)[0], opts.db_file)
    conn = sqlite3.connect(db_file); cur=conn.cursor()
    
    cur.execute("SELECT dir_name,num_imgs FROM status WHERE done==0 ORDER BY dir_name")
    list_todo = dict(map(lambda r: [r[0], r[1]], cur))
    lkey = sorted(list_todo.keys())
    
    for curd in lkey:
        cur_out = os.path.join(opts.out_dir, curd)
        ff = glob(os.path.join(cur_out, '*.jpg.feat'))
        if len(ff):
            nf = len(ff)
            print "dirname: \t num_img=%d\t num_feat=%d " % (curd, list_todo[curd], nf)
            cur.execute("UPDATE status SET done=0 WHERE dir_name='%s'" % (nf, curd) )
            conn.commit()
        else:
            print "dirname: %s empty" % curd
    
    conn.close()
    
def count_imgs(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='count img files')
    parser.add_option('-i', '--in_dir', dest='in_dir', default="", help='root dir of input img files')
    parser.add_option('-f', '--feat_dir', dest='feat_dir', default="", help='dir containing feature files')
    parser.add_option('', '--db_file', dest='db_file', default=DB_FILE_NAME, help='default: proc_objbank.db')
    parser.add_option("", '--num_models', dest='num_models', type="int", default=177,
                      help='number of model ran on data')
    (opts, __args) = parser.parse_args(sys.argv)
    
    is_jpg = re.compile("^[0-9]*.jpg$")
    db_file = os.path.join(os.path.split(opts.in_dir)[0], opts.db_file)
    conn = sqlite3.connect(db_file); cur=conn.cursor()
    db_try_out("DELETE FROM status", cur, quiet=1)
    
    total_img = 0
    max_img = 0
    num_dir = 0
    for pp, _dd, ff in os.walk(opts.in_dir):
        imgf = filter(lambda s: is_jpg.match(s), ff)
        if imgf:
            dname = pp.replace(opts.in_dir, "").strip(os.sep)
            #descp = dname.replace("/", "//")
            num_im = len(imgf)
            num_dir += 1
            total_img += num_im
            if max_img < num_im:
                max_img = num_im
            print "%s\t%d" % (dname, num_im)
            cur.execute("INSERT INTO status (dir_name, num_imgs) VALUES (?, ?)", (dname, num_im))
            #if num_dir % 10 ==0:
            conn.commit()
            
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s #dirs: %d, # imgs: %d, max-#img/dir: %d" % (tt, num_dir, total_img, max_img)
    
    conn.close()

if __name__ == '__main__':
    argv = sys.argv 
    if '--count_imgs' in argv:
        argv.remove('--count_imgs')
        count_imgs(argv)
        """
            invoked this way:
            python SCRIP_NAME.py --count_imgs -d IN_DIR
        """
    elif '--collect_img_feat' in argv:
        argv.remove('--collect_img_feat')
        collect_img_feat(argv)
        
    elif '--run_obj_job' in argv:
        argv.remove('--run_obj_job')
        run_obj_job(argv)
    
    elif '--outputdir' in argv: 
        """ NOT USED """
        argv.remove('--outputdir')
        mkdir_objbank(argv)
    
    else:  
        print "usage: python %s --OP ARGLIST" % argv[0]
        print "\t OP: parse_features, ..." 