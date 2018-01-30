import json
import sys
import os
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'toolkit'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from toolkit import TableToolkit, VizToolkit, TextToolkit, MLToolkit
from data_processing import get_table_from_jpath, create_tokenized_table

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

config = {
    'master': 'local[*]',
    'vector_dim': 100,
    'table_vector_dim': 3,
    'nbits': 2,
    'window': 2,
    'put_extractions': False,
    'regularize_tokens': True,
    'cut-off': 0.05,
    'table_path': '*.table.tables[*]',
    'max_product_th': 100
}

def get_clusters(vecs, method, n_clusters):
    if method == 'kmeans':
        cl = KMeans(n_clusters)
    elif method == 'dbscan':
        cl = DBSCAN(eps=0.009, min_samples=5)
    elif method == 'agg':
        cl = AgglomerativeClustering(n_clusters)
    return cl.fit_predict(vecs)


if __name__ == '__main__':
    conf = SparkConf().setAppName("tableEmbeddingApp") \
        .setMaster(config['master'])
    sc = SparkContext(conf=conf)
    _ = SparkSession(sc)
    input_path = sys.argv[1]
    table_counts = sc.textFile(input_path).count()
    print '{} tables found'.format(table_counts)
    tables = sc.textFile(input_path).map(lambda x: json.loads(x)).collect()
    table_vecs = np.zeros((table_counts, config['table_vector_dim']), dtype='float64')
    # ids = list()
    for i, t in enumerate(tables):
        # ids.append((t['cdr_id'], t['fingerprint']))
        for j, x in enumerate(t['vec']):
            table_vecs[i, j] = x
    method = sys.argv[3]
    num_clusters = int(sys.argv[4])
    clusters = get_clusters(table_vecs, method, num_clusters)
    outfile = open(sys.argv[2], 'w')
    for t, c in zip(tables, clusters):
        t['cluster'] = str(c)
        outfile.write(json.dumps(t) + '\n')
    outfile.close()



