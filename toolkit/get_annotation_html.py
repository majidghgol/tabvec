import json
import sys
import os
from os import walk
import re
import gzip
from random import shuffle

cdr_ids = dict()

inpath = sys.argv[2]
tables = dict()
with gzip.open(inpath, "r") as infile:
    for line in infile:
        jobj = json.loads(line)
        cdr_id = jobj['cdr_id']
        fp = jobj['fingerprint']
        tables[fp] = jobj
gt = []
outfile = open(sys.argv[3], 'w')
with open(sys.argv[1]) as ann_file:
    for line in ann_file:
        obj = json.loads(line)
        if 'THROW' in obj['labels']:
            continue
        cdr_id = obj['cdr_id']
        fp = obj['fingerprint']
        if fp not in tables:
            print 'how come?!'
            exit(0)
        obj['vec'] = tables[fp]['vec']
        obj['html'] = u'<html><body>'+tables[fp]['html']+u'</body></html>'

        outfile.write(json.dumps(obj) + '\n')