__author__ = 'majid'
import sys
import os
import json
from jsonpath_rw import jsonpath, parse
if __name__ == '__main__' and __package__ is None:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'util'))
    from toolkit import TableToolkit, VizToolkit, TextToolkit, MLToolkit

import gzip
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from __init__ import config, table_path
from __init__ import sentences, regularize_bool, domains


def get_table_dim(t):
    return t['features']['no_of_rows'], t['features']['max_cols_in_a_row']


def get_table_from_jpath(jobj, jpath, minrow, mincol):
    if jobj is None:
        return []
    if '_id' in jobj:
        cdr_id = jobj['_id']
    elif 'cdr_id' in jobj:
        cdr_id = jobj['_id']
    else:
        raise Exception('cdr_id not found in {}'.format(jobj))
    my_parser = parse(jpath)
    res = []
    for match in my_parser.find(jobj):
        val = match.value
        if val is not None:
            row,col = get_table_dim(val)
            if row >= minrow or col>=mincol:
                val['cdr_id'] = cdr_id
                res.append(val)
    return res


def create_tokenized_table(t, put_extractions, regularize):
    res = dict()
    res['cdr_id'] = t['cdr_id']
    res['fingerprint'] = t['fingerprint']
    tarr = tabletk.create_table_array(t, put_extractions, regularize)
    tabletk.clean_cells(tarr)
    res['tok_tarr'] = tabletk.create_tokenized_table_array(tarr)
    return res



if __name__ == '__main__':
    tabletk = TableToolkit()
    use_extractions = False
    # regularize_tokens = True if sys.argv[2] == 'true' else False
    conf = SparkConf().setAppName("tableEmbeddingApp") \
        .setMaster("spark://majid-Precision-Tower-3420:7077") \
        .set('spark.executor.memory', '2g') \
        .set('spark.driver.maxResultSize', '20g') \
        .set('spark.driver.memory', '6g')
    # .setMaster("spark://majid-Precision-Tower-3420:7077")
    sc = SparkContext(conf=conf)
    _ = SparkSession(sc)
    sc.addPyFile('/home/majid/my_drive/DIG/dig-table-extractor/experiments/data_processing/util/toolkit.py')
    sc.addPyFile('__init__.py')
    for domain in domains:
        for regularize_tokens in regularize_bool:
            print('domain: {}, regularization: {}, extractions: {}'.format(domain, regularize_tokens, use_extractions))

            reg = 'reg' if regularize_tokens else 'noreg'
            outpath = '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/tokenized_tables/{}/{}/'.format(reg, domain)
            etk_out = config[domain]['etk_out']
            sc.textFile(etk_out, minPartitions=600). \
                flatMap(lambda x: get_table_from_jpath(json.loads(x), table_path, 2, 2)). \
                map(lambda x: create_tokenized_table(x, use_extractions, regularize_tokens)). \
                map(lambda x: json.dumps(x)). \
                saveAsTextFile(outpath, compressionCodecClass='org.apache.hadoop.io.compress.GzipCodec')