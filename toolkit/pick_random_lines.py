import json
import random
import sys
import os
from os import walk
import re

if __name__ == '__main__':
    inpath = sys.argv[1]
    prev_ann_path = sys.argv[2]
    outfile = sys.argv[3]
    num_samples = int(sys.argv[4])
    files = []
    for (dirpath, dirnames, filenames) in walk(inpath):
        for ff in filenames:
            if re.match('^part-\d{5}$', ff):
                files.append(ff)
    tables = []
    prev_ann = set()
    for line in open(prev_ann_path):
        jobj = json.loads(line)
        prev_ann.add(jobj['fingerprint'])
    for f in files:
        with open(os.path.join(inpath, f), "r") as infile:
            for line in infile:
                jobj = json.loads(line)
                if jobj['fingerprint'] in prev_ann:
                    continue

                if len(jobj['tok_tarr']) > 2 and len(jobj['tok_tarr']) < 6 and len(jobj['tok_tarr'][0]) == 3:
                    if len(jobj['tok_tarr'][0][0]) > 1 and re.match('^\d\d$', jobj['tok_tarr'][0][0][1]):
                        continue
                    tables.append(line)
    if num_samples >= len(tables):
        print 'num_samples smaller than the input size...'
        exit(0)
    with open(outfile, 'w') as output:
        fingerprints = set()
        for x in random.sample(tables, num_samples):
            jobj = json.loads(x)
            if jobj['fingerprint'] in fingerprints:
                continue
            output.write(x)
            fingerprints.add(jobj['fingerprint'])

