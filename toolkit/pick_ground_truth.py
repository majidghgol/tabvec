import json
import sys
import os
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

def foo(doc, gt_dic):
    if doc['cdr_id'] in gt_dic and doc['fingerprint'] in gt_dic[doc['cdr_id']]:
        return [doc]
    return []

if __name__ == '__main__':
    conf = SparkConf().setAppName("tableEmbeddingApp") \
        .setMaster(sys.argv[1])
    sc = SparkContext(conf=conf)
    _ = SparkSession(sc)
    input_path = sys.argv[2]
    groundTruth = open(sys.argv[3])
    groundTruth = [json.loads(x) for x in groundTruth]
    ids = dict()
    for x in groundTruth:
        cdr_id = x['cdr_id']
        fingerprint = x['fingerprint']
        if cdr_id not in ids:
            ids[cdr_id] = set()
        ids[cdr_id].add(fingerprint)

    tables = sc.textFile(input_path).\
        map(lambda x: json.loads(x)).\
        flatMap(lambda x: foo(x, ids)).\
        collect()
    outfile = open(sys.argv[4], 'w')
    for t in tables:
        outfile.write(json.dumps(t) + '\n')
    outfile.close()

