from clustering.EvaluateClustering import evaluate_clustering
import gzip
import json
import numpy as np
import os
import re
from os import walk

if __name__ == '__main__':
    for d in ['HT']:
        print('####### working on {} #######'.format(d))
        path = '/home/majid/my_drive/DIG/tabvec/output/{}'.format(d)
        outpath = '/home/majid/my_drive/DIG/tabvec/output/evaluation/{}.json'.format(d)
        GT_path = '/home/majid/my_drive/DIG/data/{}_annotated.jl'.format(d)

        eval_res = dict()
        files = []
        for (dirpath, dirnames, filenames) in walk(path):
            for ff in filenames:
                if re.match('^cl_.*\.gz$', ff):
                    files.append(ff)
        for i, f in enumerate(files):
            print('{}/{}: {} .....'.format(i+1, len(files), f))
            tables_clusters = []
            with gzip.open(os.path.join(path, f), "r") as infile:
                for line in infile:
                    tables_clusters.append(json.loads(line))
            table_vecs = np.matrix([x['vec'] for x in tables_clusters])
            eval_res[f] = evaluate_clustering(table_vecs, tables_clusters, GT_path)
        outfile = open(outpath, 'w')
        outfile.write(json.dumps(eval_res))
        outfile.close()

