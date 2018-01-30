import pickle
from table_embedding import TableEmbedding
import gzip
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from __init__ import sentences, regularize, domains

if __name__ == '__main__':
    method = 'avg'  # concat or avg
    indexing_method = 'randomIndexing'
    # indexing_method = 'word2vec'
    tok_threshold = 10
    min_count = 100

    conf = SparkConf().setAppName("GetReadabilitySentences") \
        .setMaster("spark://majid-Precision-Tower-3420:7077") \
        .set('spark.executor.memory', '2g') \
        .set('spark.driver.maxResultSize', '20g') \
        .set('spark.driver.memory', '6g')
    # .setMaster("spark://majid-Precision-Tower-3420:7077")
    sc = SparkContext(conf=conf)
    _ = SparkSession(sc)
    sc.addPyFile('/home/majid/my_drive/DIG/dig-table-extractor/experiments/data_processing/util/toolkit.py')
    sc.addPyFile('random_indexing.py')
    sc.addPyFile('table_embedding.py')
    sc.addPyFile('__init__.py')
    for domain in domains:
        for reg in regularize:
            print 'domain: {}, regulize: {}'.format(domain, reg)
            tokenized_tables = '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/tokenized_tables/{}/{}/'.format(reg, domain)
            readable_texts = '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/readable_text/{}/'.format(domain)
            def f(x):
                print x
            tokenized_tables = sc.textFile(tokenized_tables)
            readable_texts = sc.textFile(readable_texts)
            # readable_texts.foreach(f)

            te = TableEmbedding(aggregate_method=method,
                                d=200, token_thresh=100)
            for k, x in te.train_models(sc, tokenized_tables, readable_texts, sentences, method=indexing_method, min_count=min_count):
                outpath = '/home/majid/my_drive/DIG/dig-table-extractor/experiments/data/word_embeddings/{}/{}.{}.pickle.gz'.format(domain, k, reg)
                # print x
                pickle.dump(x, gzip.open(outpath, 'wb'))
                del x