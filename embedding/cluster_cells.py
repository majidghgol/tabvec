import gzip
import json

all_vecs = list()

with gzip.open('/Users/majid/DIG/data/ATF_new_clusters/kmeans_12.tar.gz') as infile:
    for line in infile:
        doc = json.loads(line)
        vec_tarr = doc['vec_tarr']
        vecs = []
        for row in vec_tarr:
            for x in row:
                vecs.append(x)
        all_vecs.extend(vecs)

