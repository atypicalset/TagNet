import sys, os
import urllib2
import re
import random
from datetime import datetime
import json
import codecs
from optparse import OptionParser

DEBUG = 1
FLICKR_XML_DIR = '/data1/vault/xlx/imgnet-flickr/json'

def cache_flickr_info(imgid, cur_key, rootdir=FLICKR_XML_DIR, hash_level=2, chars_per_hash=2):
	flickr_get_info = 'http://api.flickr.com/services/rest/?method=flickr.photos.getInfo&format=json&api_key=%s&photo_id=%s'
	curs = 0
	cure = chars_per_hash
	hdir = []
	for i in range(hash_level):
		curs += i*chars_per_hash
		cure = curs + chars_per_hash
		hdir.append(imgid[curs:cure])
	if rootdir:
		outdir = os.path.join(rootdir, '/'.join(hdir))
		if not os.path.exists(outdir):
			os.makedirs(outdir)
	
		meta_name = os.path.join(outdir, imgid+".json")
	else:
		meta_name = ""
	
	jinfo = {}
	if os.path.exists(meta_name):
		#print meta_name
		try:
			jstr = codecs.open(meta_name, encoding='utf-8', mode='rt').read()
			jinfo = json.loads( jstr )
		except:
			print "  error reading %s " % meta_name
			jinfo = {}
	else:
		try:
			req_url = flickr_get_info % (cur_key, imgid)
			jstr = urllib2.urlopen(req_url).read()
			#print req_url
			""" get rid of the flickr wrap 'jsonFlickrApi({"photo":{}})' """
			jstr = jstr.replace('jsonFlickrApi', '')
			jstr = jstr.strip('()')
			""" parse json info """
			jinfo = json.loads( jstr )
			if rootdir:
				open(meta_name, 'wt').write(json.dumps(jinfo, indent=4))
		except Exception:
			print "\tError getting metadata for %s: %s" % (imgid, sys.exc_info()[0])
			print "\n\t" + req_url
			jinfo = {}
	
	return jinfo

def get_imgnet_url(argv):
	if argv is None:
		argv = sys.argv
	if len(argv)==1:
		argv = argv.append("-h")
		
	parser = OptionParser() #OptionParser(description='return a list of filtered url given query WordNet ID')
	parser.add_option("-w", "--wnid", dest="wnid", default='', 
						help="wordnet id of the query, e.g. 'n03360622' ")
	parser.add_option("-g", "--imgnet_geturl", dest="imgnet_geturl", 
						default='http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=%s',
						help="url template for the get_URL method of imagenet")
	parser.add_option("-p", "--url_pattern", dest="url_pattern", 
						default='flickr.com/', help="string filter for URLs to keep")
	(opts, args) = parser.parse_args(argv)
	#print opts, args
	if len(args) > 1:
		parser.error("incorrect number of arguments")
	
	ustr = urllib2.urlopen(opts.imgnet_geturl % opts.wnid).read()
	url_list = map(lambda s: s.strip(), ustr.split())
	n0 = len(url_list)
	
	url_list = filter(lambda s: s.find(opts.url_pattern)>=0, url_list)
	n1 = len(url_list)
	tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
	print "%s %s: %d urls, %d with '%s'" % (tt, opts.wnid, n0, n1, opts.url_pattern)
	return url_list,n0

def get_flickr_info(url_list, argv=None):
	"""
	flickrurl_regexp = 'http://farm[0-9].static[\.*]flickr.com/[0-9]*/[0-9]*_{secret}.jpg';
	flickr_photoinfo = 'http://api.flickr.com/services/rest/?method=flickr.photos.getInfo&api_key=%s&photo_id=%s';
	"""
	parser = OptionParser(description='return co-occurring tag counts for a given list of flickr URLs')
	
	#parser.add_option("-g", "--flickr_get_info", dest="flickr_get_info", 
	#	default='http://api.flickr.com/services/rest/?method=flickr.photos.getInfo&format=json&api_key=%s&photo_id=%s',
	#	help="url template for the get_URL method of imagenet")
	parser.add_option('-j', '--json_dir', dest='json_dir', default=FLICKR_XML_DIR, 
					help='dir to cache json metadata of each photo')
	parser.add_option('-k', '--flickr_key_file', dest='flickr_key_file', 
		default='flickr.key.txt', help='file containing a list of API keys, one per line')
	""" 'http://static.flickr.com/2088/[id]_94dbc23839.jpg' """
	parser.add_option("-p", "--id_pattern", dest="id_pattern", 
		default="[^/]*//.*/[0-9]*/(?P<flickrid>[0-9]*)\_([0-9a-z]*).*", help="regexp to get flickr id")
	(opts, args) = parser.parse_args(argv)	#@UnusedVariable
	
	api_keys = open(opts.flickr_key_file, 'r').read().split()
	api_keys = map(lambda s: s.strip(), api_keys)
	id_pattern = re.compile(opts.id_pattern)
	
	usr_tag = {}
	usr_cnt = {}
	icnt = 0
	failcnt = 0
	for cur_u in url_list:
		try:
			m = id_pattern.match(cur_u)
			imgid = m.group('flickrid')
		except:
			print "\t err parsing URL" + cur_u
			continue
		icnt += 1
		""" get json rep of the image info from flickr"""
		cur_key = api_keys [random.randint(0, len(api_keys)-1)]

		jinfo = cache_flickr_info(imgid, cur_key, rootdir=opts.json_dir)
		if 'stat' not in jinfo or not jinfo['stat']=='ok' :
			if icnt%50==0 or DEBUG>2:
				tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
				print "%s %5d/%5d flickr.photos.getInfo failed: %s" % (tt, icnt, len(url_list), imgid)
				msg = 'UNKNOWN' if 'message' not in jinfo else jinfo['message']
				print '\t\tmessage: "' + msg + '"' 	#json.dumps(jinfo, sort_keys=True, indent=4)
			failcnt += 1
			continue
			#elif 'photo' not in jinfo:
			#print json.dumps(jinfo, indent=4)
			#print "error parsing %s " % imgid			
		else:
			pinfo = jinfo["photo"]
			usr = pinfo["owner"]["nsid"]
			tt = pinfo["tags"]["tag"]
			tag_raw = map(lambda s:s["_content"], tt)
			#print tag_raw
			if len(tt)==0:
				""" pic with no tags, skip """
				continue 
			""" collect tag + usr info for this pic"""
			if not usr in usr_tag:
				usr_tag[usr] = []
				usr_cnt[usr] = 0
			
			usr_cnt[usr] += 1
			for t in tt:
				tcleaned = clean_tag(t["_content"]) # does nothing for the moment
				if tcleaned not in usr_tag[usr]:
					usr_tag[usr].append( tcleaned )
			if icnt%50==0 or DEBUG>2:
				tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
				print "%s %5d/%5d usr %s img %s tags: %s" % (tt, icnt, len(url_list), usr, imgid,  u','.join(tag_raw) )
	
	tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
	print "%s %5d image queried, %5d failed, %d users" % (tt, icnt, failcnt, len(usr_tag))
	
	""" aggregate over all users """
	tag_cnt = {}
	ucnt = 0
	for u,tt in usr_tag.iteritems():
		for ct in tt:
			if ct in tag_cnt:
				tag_cnt[ct] += 1
			else:
				tag_cnt[ct] = 1
		ucnt += 1
		if ucnt%100==0 or DEBUG>2:
			tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
			print "%s %d/%d users %s, %d tags" % (tt, ucnt, len(usr_tag), u, len(tt))
		
	return tag_cnt

def write_tagcnt(tag_cnt, print_to_screen=True, outfile=None):
	from operator import itemgetter
	if print_to_screen:
		#tk = tag_cnt.keys()
		#tc = tag_cnt.values()
		for w in sorted(tag_cnt.iteritems(), key=itemgetter(1), reverse=True):
			if tag_cnt[w[0]]>=5:			
				print "%6d\t%s" % (tag_cnt[w[0]], w[0])  
	return


def get_substree(wnid, full=1):
	""" http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=[wnid]&full=1 """
	wn_struct_api = 'http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=%s&full=1'
	ustr = urllib2.urlopen(wn_struct_api % wnid).read()
	wn_list = map(lambda s: s.strip().strip('-'), ustr.split())
	
	print "%d nodes descending from %s" % (len(wn_list), wnid)
	return wn_list

def clean_tag(tag_in, varargin=None):
	parser = OptionParser(description='run various tag-cleaning')
	parser.add_option("-s", "--sqlite_file", dest="sqlite_file", 
		default='~/tmp/tags.cache', help="file storing tag cleaning history")
	parser.add_option('-k', '--bing_key_file', dest='bing_key_file', 
		default='bing.appid.txt', help='file containing a list of API keys, one per line')
	parser.add_option('-w', '--wordnick_key_file', dest='wordnick_key_file', 
		default='wordnik.key.txt', help='file containing a list of API keys, one per line')
	
	(opts, args) = parser.parse_args(varargin)	#@UnusedVariable
	api_keys = open(opts.bing_key_file, 'r').read().split()
	api_keys = map(lambda s: s.strip(), api_keys)
	_wordnik_key = open(opts.wordnick_key_file, 'r').read().strip()	
	
	tcleaned = tag_in
	"""dict lookup"""
	
	"""translate/spell correct and lookup again"""
	
	return tcleaned

if __name__ == '__main__':    
	if len(sys.argv)<2:
		print "usage: %s root_wn_id [cache_dir]" % sys.argv[0]
		
	else:
		wlist = get_substree(sys.argv[1])
		for ww in wlist:
			url_list,n0 = get_imgnet_url(['-w', ww]) #'n09470027' fast debug # 'n09421951' sandbar	
			if len(sys.argv)>2:
				tag_cnt = get_flickr_info(url_list, ['-j', sys.argv[1]])	
			else:
				tag_cnt = get_flickr_info(url_list)
			write_tagcnt(tag_cnt)
		
		
