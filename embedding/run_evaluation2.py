from clustering.EvaluateClustering2 import evaluate_clustering
import gzip
import json
import numpy as np
import os
import re
from os import walk
import pickle
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from data_processing import load_GT, get_GT_tables, get_unique
import sys

if __name__ == '__main__':
    configs = {
        'spark': {
            'spark_master': 'local[20]',  # 'spark://kg2018a.isi.edu:7077',  # spark://majid-Precision-Tower-3420:7077
            # 'spark_master': 'spark://kg2018a.isi.edu:7077',
            'executor_mem': '15g',
            'driver_mem': '15g',
            'driver_max_res_size': '15g',
            'exec_instances': '1',
            'exec_cores': '5'
        }
    }

    conf = SparkConf().setAppName("evaluateClusteringApp") \
        .setMaster(configs['spark']['spark_master']) \
        .set("spark.executor.memory", configs['spark']['executor_mem']) \
        .set('spark.driver.memory', configs['spark']['driver_mem']) \
        .set('spark.driver.maxResultSize', configs['spark']['driver_max_res_size'])

    sc = SparkContext(conf=conf)
    _ = SparkSession(sc)
    sc.addPyFile('../tabvec_src.zip')


    tabvecs_path = sys.argv[1]
    cl_model_path = sys.argv[2]
    GT_path = sys.argv[3]

    cl_model = pickle.load(open(cl_model_path, 'rb'))

    eval_res = dict()

    gt = load_GT(GT_path)
    gt_ids = set([(x['cdr_id'], x['fingerprint']) for x in gt])
    intables = sc.textFile(tabvecs_path).flatMap(lambda x: get_GT_tables(json.loads(x), gt_ids)).collect()
    intables = get_unique(intables)

    intable_ids = set([(x['cdr_id'], x['fingerprint']) for x in intables])

    if len(gt) != len(intables):
        print('WARNING!! removing gt tables not in input file. gt={}, infile={}'.format(len(gt), len(intables)))
        gt = [x for x in gt if (x['cdr_id'], x['fingerprint']) in intable_ids]
        print len(gt_ids), len(intable_ids)

    gt = sorted(gt, key=lambda x: (x['cdr_id'], x['fingerprint']))
    intables = sorted(intables, key=lambda x: (x['cdr_id'], x['fingerprint']))

    assert(len(gt) == len(intables))

    print evaluate_clustering(intables, cl_model, gt)


