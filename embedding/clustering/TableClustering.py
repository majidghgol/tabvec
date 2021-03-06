import json
import sys
import os
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gzip
from toolkit.toolkit import TableToolkit, VizToolkit, TextToolkit, MLToolkit
from data_processing import get_table_from_jpath, create_tokenized_table

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def get_clusters(vecs, method, n_clusters):
    if method == 'kmeans':
        cl = KMeans(n_clusters)
    elif method == 'dbscan':
        cl = DBSCAN(eps=0.5, min_samples=1000)
    elif method == 'agg':
        cl = AgglomerativeClustering(n_clusters)
    return cl.fit_predict(vecs), cl

def get_id_vector(doc):
    return dict(vector=np.array(doc['vector'], dtype='float64'),
                cdr_id=doc['cdr_id'],
                fingerprint=doc['fingerprint'])


def run_clustering(table_vecs, method, num_clusters, outpath):
    # table_id_vecs = sc.textFile(tables_path).map(lambda x: json.loads(x)).collect()
    # table_id_vecs = sc.textFile(tables_path).map(lambda x: get_id_vector(json.loads(x))).collect()
    clusters, cl = get_clusters(table_vecs, method, num_clusters)
    print 'writing model to output ...'
    pickle.dump(cl, open(outpath+'.pickle', 'wb'))
    return clusters

def run_clustering2(texts, vecs, method, num_clusters):
    clusters, cl = get_clusters(vecs, method, num_clusters)
    return clusters





