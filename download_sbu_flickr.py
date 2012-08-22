import sys, os

import urllib2
import random
from datetime import datetime
#from time import sleep
from optparse import OptionParser
from imgtags import cache_flickr_info

FLICKR_KEY_FILE = 'flickr.key.txt'

def download_sbu_imgs(url_file, id_file, img_root_dir, startnum=0, endnum=50, hash_level=2, chars_per_hash=2):
    ss = os.sep
    exist_cnt = 0
    err_cnt = 0
    good_cnt = 0
    cnt = 0
    url_lines = open(url_file, 'rt').read().split("\n")
    id_lines = open(id_file, 'rt').read().split("\n")
    
    api_keys = open(FLICKR_KEY_FILE, 'r').read().split()
    api_keys = map(lambda s: s.strip(), api_keys)
    
    json_dir = os.path.join(os.path.split(img_root_dir)[0], 'sbu-json')
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processing #%d - #%d of %d urls"  % (tt, startnum, endnum, len(url_lines))
    ii = startnum 
    while ii < endnum and ii < len(url_lines):
    #for (ul, imid) in enumerate(url_lines, id_lines):

        imgid = id_lines[ii]
        imgurl = url_lines[ii]
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
        
        cur_key = api_keys [random.randint(0, len(api_keys)-1)]
        _jinfo = cache_flickr_info(imgid, cur_key, rootdir=json_dir)
        #if 'stat' not in jinfo or not jinfo['stat']=='ok' :
        #    err_cnt += 1  
        #    continue 
                
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
                print "  ERR downloading url #%d, img %s from %s" % (ii, imgid, imgurl)
                err_cnt += 1
        else:
            exist_cnt += 1
        
        ii += 1
        if cnt % 100 == 0:
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s processed %d urls: %d new, %d exist, %d err \n\t"  % (tt, cnt, good_cnt, exist_cnt, err_cnt)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processed %d urls: %d new, %d exist, %d err \n\t from %s"  % (tt, cnt, good_cnt, exist_cnt, err_cnt, url_file)
    

if __name__ == '__main__':
    argv = sys.argv
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='download flickr images for imagenet')
    parser.add_option("-u", "--url_file", dest="url_file", 
        default='', help="input file containing flickr urls")
    parser.add_option("-i", "--id_file", dest="id_file", 
        default='', help="input file containing flickrids corresponding to the above")
    parser.add_option('-o', '--img_root_dir', dest='img_root_dir', 
        default='', help='root dir where images should be stored')

    parser.add_option("-s", "--startnum", dest="startnum", type='int', default=0, 
                      help = "start number")
    parser.add_option("-e", "--endnum", dest="endnum", type='int', default=10,   
                      help = "end line number ")
    opts, __ = parser.parse_args(argv)
    
    download_sbu_imgs(opts.url_file, opts.id_file, opts.img_root_dir, 
                      startnum=opts.startnum, endnum=opts.endnum)
    