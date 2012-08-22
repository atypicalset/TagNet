
import sys, os
import urllib2
import hashlib
from datetime import datetime
from optparse import OptionParser


def download_nuswide(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='download images in NUS wide dataset')
    parser.add_option("-i", "--in_file", dest="in_file", 
        default='', help="input file")
    parser.add_option('-o', '--out_dir', dest='out_dir', 
        default='', help='file containing dictionary db')
    parser.add_option("-j", "--junk_img", dest="junk_img", default="flickr_na.jpg", 
                      help = "path to example flickr NOTFOUND img")
    opts, __ = parser.parse_args(argv)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s downloading entries in %s into %s" % (tt, opts.in_file, opts.out_dir)
    
    sig_na = hashlib.new("md5", open(opts.junk_img, 'rb').read()).digest()
    
    linecnt = 0
    errcnt = 0
    for cl in open(opts.in_file):
        if cl[1]=="%":
            continue # skip comment line
        linecnt += 1
        """  line format 
        %%%%%%%%%%Photo_file    Photo_id    url_Large   url_Middle   url_Small  url_Original%%%%%%%%%%%%%%%%%
        C:\ImageData\Flickr\actor\0001_2124494179.jpg   2124494179  http://farm3.static.flickr.com/2244/2124494179_b039ddccac_b.jpg  http://farm3.static.flickr.com/2244/2124494179_b039ddccac.jpg  http://farm3.static.flickr.com/2244/2124494179_b039ddccac_m.jpg  null
        C:\ImageData\Flickr\actor\0002_174174086.jpg   174174086  http://farm1.static.flickr.com/44/174174086_a0b4e9d9cf_b.jpg  http://farm1.static.flickr.com/44/174174086_a0b4e9d9cf.jpg  http://farm1.static.flickr.com/44/174174086_a0b4e9d9cf_m.jpg  null
        """
        tmp = cl.strip().split()
        dest_file = tmp[0].replace("C:\\ImageData\\Flickr\\", "")
        dest_file = os.path.join(opts.out_dir, dest_file.replace("\\", os.sep))
        dest_dir,__ = os.path.split(dest_file)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        imgurl = tmp[3]
        if imgurl=="null":
            errcnt += 1
            continue
        
        try:
            buf = urllib2.urlopen(imgurl).read()
            sig = hashlib.new("md5", buf).digest()
            if not sig== sig_na:
                fh = open(dest_file, 'wb')
                fh.write(buf)
                fh.close()
            else:
                errcnt += 1
                print " img not available at line# %d url %s" % (linecnt, imgurl)
        except:
            errcnt += 1
            print " err downloading line# %d url %s" % (linecnt, imgurl)
            print cl
            print ""
            if os.path.exists(dest_file):
                os.remove(dest_file)
            
        if linecnt % 100 == 0:
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s processed %6d images, %4d error" % (tt, linecnt, errcnt)
    

    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s total %d, %d err. from %s " % (tt, linecnt, errcnt, opts.in_file )

if __name__ == '__main__':  
    argv = sys.argv 
    download_nuswide(argv)