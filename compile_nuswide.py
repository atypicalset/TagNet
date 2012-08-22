import sys, os
import pickle
from datetime import datetime
from random import random
from optparse import OptionParser

def parse_objbank_features(argv):
    if len(argv)<2:
        argv = ['-h']
    parser = OptionParser(description='load objbank output, according to train/test')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing data files')
    parser.add_option('-f', '--feat_dir', dest='feat_dir', default="", help='dir containing feature files')
    parser.add_option("", '--img_file_list', dest='img_file_list', default=["TestImagelist.txt", "TrainImagelist.txt"], 
                      help='list of files containing train/test img names')
    parser.add_option("", '--rm_list_perfix', dest='rm_list_perfix', default="C:\\ImageData\\Flickr\\",
                      help='prefix to image path, to remove')
    parser.add_option("", '--num_models', dest='num_models', type="int", default=177,
                      help='number of model ran on data')
    (opts, __args) = parser.parse_args(sys.argv)
    
    list_holder = map(lambda r: random(), range(opts.num_models))
    ## input lines in Test- and Train- img list look like this
    # C:\ImageData\Flickr\actor\0002_174174086.jpg
    # --> "actor/0002_174174086.jpg"  in in_name
    for listf in opts.img_file_list:
        in_lines = filter(len, open(os.path.join(opts.db_dir, listf), "rt").read().split("\n") )
        in_name = map(lambda s: s.replace(opts.rm_list_perfix, "").replace("\\", os.sep), in_lines)
        
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s read %d lines from %s" % (tt, len(in_name), listf)
        
        num_found = 0
        num_lost = 0
        out_txt_file =  os.path.join(opts.db_dir, os.path.splitext(listf)[0]+".feat.txt")
        #out_pickle_file = os.path.join(opts.db_dir, os.path.splitext(listf)[0]+".feat.pkl")
        fo = open(out_txt_file, "wt")
        
        file_dict = dict.fromkeys(in_name)      # key: file name, value: flickrid (int)
        feature_dict = {} #dict.fromkeys(in_name)  with missing keys  # key: file name, value: feature vector (list of float)
        for cn in in_name:
            feat_file = os.path.join(opts.feat_dir, cn+".feat")
            if os.path.isfile(feat_file):
                cur_id = int( cn.split(".")[0].split("_")[-1] )
                file_dict[cn] = cur_id
                featnum = filter(len, open(feat_file, "rt").read().split("\n") )
                featnum = map(float, featnum)
                nrows = len(featnum)/opts.num_models 
                feature_dict[cn] = list_holder[:] # create place-holder
                for i in range(opts.num_models):
                    feature_dict[cn][i] = max( featnum[i*nrows: (i+1)*nrows] )
                num_found += 1
                fo.write("%s\t%s\n" % (cn, " ".join(map(lambda x: "%0.5f"%x, feature_dict[cn])) ))
            else:
                file_dict[cn] = ""
                num_lost += 1
            if (num_found+num_lost)%1000 == 0:
                tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                print "%s %6d names processed, %d found, %d not found" % (tt, num_found+num_lost, num_found, num_lost)
        
        fo.close()
        
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s done processing %s, %6d lines, %d found, %d not found" % (tt, listf, num_found+num_lost, num_found, num_lost)
        
        #pickle.dump((feature_dict, file_dict), open(out_pickle_file, 'wb'))

if __name__ == '__main__':
    argv = sys.argv 
    if '--parse_features' in argv:
        argv.remove('--parse_features')
        parse_objbank_features(argv)
        """
            invoked this way:
            python compile_nuswide.py --parse_features -d /home/users/xlx/vault-xlx/NUSWide/nuswide-db 
            -f /home/users/xlx/vault-xlx/NUSWide/objbank
        """
    else:  
        print "usage: python %s --OP ARGLIST" % argv[0]
        print "\t OP: parse_features, ..." 