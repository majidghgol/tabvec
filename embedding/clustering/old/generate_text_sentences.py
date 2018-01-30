__author__ = 'majid'
import sys
import os
import json
from jsonpath_rw import jsonpath, parse
if __name__ == '__main__' and __package__ is None:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'util'))
    from toolkit import TextToolkit

import gzip
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from __init__ import config, readibility_path
import re
from __init__ import sentences, regularize, domains


def get_table_dim(t):
    return t['features']['no_of_rows'], t['features']['max_cols_in_a_row']


def get_readability_sentences(jobj, jpath):
    mypath = parse(jpath)
    matches = mypath.find(jobj)
    if len(matches) == 0:
        return []
    text = matches[0].value
    sentences = [TextToolkit.clean_text(x) for x in text.split('.')]
    sentences = [re.split('\s+', x) for x in sentences]
    return sentences

if __name__ == '__main__':
    conf = SparkConf().setAppName("GetReadabilitySentences") \
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
    # for domain in ['HT_sample']:
        print('domain: {}'.format(domain))

        outpath = '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/readable_text/{}/'.format(domain)
        etk_out = config[domain]['etk_out']
        sc.textFile(etk_out, minPartitions=30). \
            flatMap(lambda x: get_readability_sentences(json.loads(x), readibility_path)). \
            saveAsTextFile(outpath, compressionCodecClass='org.apache.hadoop.io.compress.GzipCodec')