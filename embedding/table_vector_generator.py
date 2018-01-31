import sys
import os
import shutil
from itertools import product
from TableEmbedding import run_table_embedding
from WordEmbedding import run_word_embedding

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

if __name__ == '__main__':
    dig_home = "/Users/majid/DIG"
    # datasets = ['ATF', 'SEC', 'HT', 'WCC']
    datasets = ['ATF']
    cutoff = [0.005, 0.01, 0.05]
    nclusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    regularize = [True, False]
    vec_dim = [100, 200]
    # sentences = [['text', 'hrow', 'cell', 'adjcell', 'hcol'],
    #              ['text', 'cell'],
    #              ['text', 'hrow', 'hcol'],
    #              ['text', 'cell', 'hrow', 'hcol'],
    #              ['text'],
    #              ['cell'],
    #              ['adjcell'],
    #              ['hrow'],
    #              ['hcol']
    #              ]

    sentences = [['text', 'cell']]

    command = sys.argv[1]

    if command == 'create':
        conf = SparkConf().setAppName("tableEmbeddingApp") \
            .setMaster('local[*]')
        sc = SparkContext(conf=conf)
        _ = SparkSession(sc)
        sc.addPyFile('../toolkit/toolkit.py')
        sc.addPyFile('data_processing.py')
        sc.addPyFile('random_indexing.py')
        sc.addPyFile('random_indexing_wrapper.py')
        sc.addPyFile('TableEmbedding.py')
        sc.addPyFile('__init__.py')
        command2 = sys.argv[2]
        if command2 == 'words':
            # generate word embeddings with different settings
            for d in datasets:
                for c, r, dim, s in product(cutoff, regularize, vec_dim, sentences):
                    print 'creating word embeddings with: c={}, r={},d={},s=[{}]'.format(c,r,dim,','.join(s))
                    config = {
                        'vector_dim': dim,
                        'nbits': 2,
                        'window': 2,
                        'put_extractions': False,
                        'regularize_tokens': r,
                        'cut-off': 0.05,
                        'table_path': '*.table.tables[*]',
                        'max_product_th': 100,
                        'sentences': s,
                        'text_path': '*.table.html_text'
                    }
                    r_text = 'reg' if r else 'noreg'
                    inpath = '{}/data/{}.etk.out'.format(dig_home,d)
                    wepath = '{}/tabvec/output/{}/we_{}_{}_{}_{}'.format(dig_home,d,c,r_text,dim,'_'.join(s))
                    run_word_embedding(sc, config, inpath, wepath)
        elif command2 == 'tables':
            # generate table vectors with different settings
            for d in datasets:
                for c, r, dim, s in product(cutoff, regularize, vec_dim, sentences):
                    config = {
                        'vector_dim': dim,
                        'nbits': 2,
                        'window': 2,
                        'put_extractions': False,
                        'regularize_tokens': r,
                        'cut-off': 0.05,
                        'table_path': '*.table.tables[*]',
                        'max_product_th': 100,
                        'sentences': s,
                        'text_path': '*.table.html_text'
                    }
                    r_text = 'reg' if r else 'noreg'
                    inpath = '{}/data/{}.etk.out'.format(dig_home,d)
                    wepath = '{}/tabvec/output/{}/we_{}_{}_{}_{}'.format(dig_home,d,c,r_text,dim,'_'.join(s))
                    outpath = '{}/tabvec/output/{}/tablevecs_{}_{}_{}_{}'.format(dig_home,d, c, r_text, dim, '_'.join(s))

                    run_table_embedding(sc, config, inpath, wepath, outpath)
    if command == 'clear':
        for d in datasets:
            for c, r, dim, s in product(cutoff, regularize, vec_dim, sentences):
                r_text = 'reg' if r else 'noreg'
                try:
                    shutil.rmtree('{}/tabvec/output/{}/we_{}_{}_{}_{}'.format(dig_home,d, c, r_text, dim, '_'.join(s)))
                    shutil.rmtree('{}/tabvec/output/{}/tablevecs_{}_{}_{}_{}'.format(dig_home,d, c, r_text, dim, '_'.join(s)))
                except Exception as e:
                    print e
