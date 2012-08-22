import sys, os
import re
import random
from datetime import datetime
from optparse import OptionParser

from imgtags import get_substree, get_imgnet_url, cache_flickr_info

DEBUG = 1
FLICKR_XML_DIR = '/home/users/xlx/vault-xlx/imgnet-flickr/json'
WNET_OUT_DIR = '/home/users/xlx/vault-xlx/imgnet-flickr/wnet'

#DB_DIR = '/home/users/xlx/vault-xlx/proj'
#DB_DIR = '/home/users/xlx/db'
#DB_FILE = 'imgnet.db'

def compile_flickr_info(url_list, argv):
    
    parser = OptionParser(description='return co-occurring tag counts for a given list of flickr URLs')
    parser.add_option('-o', '--out_dir', dest='out_dir', default=WNET_OUT_DIR, help='output dir for wordnet-flickr log file')
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
    
    out_file = os.path.join(opts.out_dir, opts.wordnet_id+".txt")
    if os.path.exists(out_file):
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s output '%s' already exist, RETURN \n\n" % (tt, out_file)
        return
    
    icnt = 0
    good_cnt = 0
    fail_cnt = 0
    dup_cnt = 0
    id_url = {}
    for cur_u in url_list:
        icnt += 1
        try:
            m = id_pattern.match(cur_u)
            imgid = m.group('flickrid')
        except:
            print "\t err parsing URL" + cur_u # assume url already contains flickr.com/
            continue    
        
        flickrid = int(imgid)
        if flickrid in id_url:
            continue
            dup_cnt += 1
        else:            
            cur_key = api_keys [random.randint(0, len(api_keys)-1)]
            jinfo = cache_flickr_info(imgid, cur_key, rootdir=opts.json_dir)
            if 'stat' not in jinfo or not jinfo['stat']=='ok' :
                fail_cnt += 1               
                #if icnt%100 == 0 or DEBUG>2:
                #    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                #    print "%s %5d/%5d flickr img not found: %s" % (tt, icnt, len(url_list), flickrid)
                    
            else:
                good_cnt += 1
                id_url[flickrid] = cur_u
                   
        if icnt%200 == 0: 
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s %d/%d urls processed, %d good records, %d dups, %d failed" \
                % (tt, icnt, len(url_list), good_cnt, dup_cnt, fail_cnt)
            
    # write out the resulting tuples
    
    if not os.path.exists(out_file):
        fh = open(out_file, 'wt')
        for ii, uu in id_url.iteritems():
            fh.write("%s\t%11d\t%s\n" % (opts.wordnet_id, ii, uu) )
        fh.close()
        
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s %d urls processed, %d good records, %d dups, %d failed \n\n" \
            % (tt, len(url_list), good_cnt, dup_cnt, fail_cnt)
    else:
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s %d urls processed, SKIP existing output file %s \n\n" \
            % (tt, len(url_list), out_file)
    return

if __name__ == '__main__':  
    if "--no_tree" in sys.argv:
        argv = sys.argv
        argv.remove("--no_tree")
        ww = argv[1]
        url_list,n_allurl = get_imgnet_url(['-w', ww]) 
        if url_list:
            random.shuffle(url_list)
            tag_cnt = compile_flickr_info(url_list, ['-w', ww])
    else:
        wlist = get_substree(sys.argv[1])
        random.shuffle(wlist)
    
        wcnt = 0
        for ww in wlist:
            wcnt += 1
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s Processing %d of %d synsets" % (tt, wcnt, len(wlist))
             
            url_list,n_allurl = get_imgnet_url(['-w', ww]) #'n09470027' fast debug # 'n09421951' sandbar
            if url_list:
                random.shuffle(url_list)
                if len(sys.argv)<=2:
                    tag_cnt = compile_flickr_info(url_list, ['-w', ww])
                else:
                    tag_cnt = compile_flickr_info(url_list, ['-w', ww, '-j', sys.argv[1]])


        