import json
import sys
import os
from os import walk
import re
import gzip
from random import shuffle

cdr_ids = dict()

gt = []
with open(sys.argv[1]) as ann_file:
    for line in ann_file:
        obj = json.loads(line)
        if 'THROW' in obj['labels']:
            continue
        cdr_id = obj['cdr_id']
        fp = obj['fingerprint']
        if cdr_id not in cdr_ids:
            cdr_ids[cdr_id] = set()
        cdr_ids[cdr_id].add(fp)
        gt.append(obj)
shuffle(gt)
inpath = sys.argv[2]
files = []
tables = []
with gzip.open(inpath, "r") as infile:
    for line in infile:
        jobj = json.loads(line)
        cdr_id = jobj['cdr_id']
        fp = jobj['fingerprint']
        if cdr_id in cdr_ids and fp in cdr_ids[cdr_id]:
            tables.append(jobj)
fps = set()
for t in tables:
    fps.add(t['fingerprint'])
diff = set(fps) - set([x['fingerprint'] for x in tables])
print len(diff)
# for x in diff:
#     print x
#     print ''
outfile = open(sys.argv[3], 'w')
for t in tables:
    t['html'] = u'<html><body>'+t['html']+u'</body></html>'
    outfile.write(json.dumps(t) + '\n')
outfile.close()

outfile = open(sys.argv[4], 'w')
counter = 0
print gt
for t in gt:
    if t['fingerprint'] in fps:
        if 'NON-DATA' in t['labels']:
            if counter > 120:
                continue
            counter += 1

        outfile.write(json.dumps(t) + '\n')
outfile.close()