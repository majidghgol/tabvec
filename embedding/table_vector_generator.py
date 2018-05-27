import sys
import os
import shutil
import json
from itertools import product
from TableEmbedding import run_table_embedding
from WordEmbedding import run_word_embedding, create_occurrences, read_occurrences
from WordCount import run_word_count
from data_processing import count_tables
from TokenizeTableText import run_tokenize_table, run_tokenize_text
import numpy as np
from os import walk
import re
import gzip

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

from clustering.TableClustering import run_clustering

def realse_list(a):
   del a[:]
   del a

if __name__ == '__main__':
    dig_home = "/home/majid/my_drive/DIG"
    cl_methods = ['kmeans']
    num_clusters = [4,6,8,10,12,14]
    # datasets = ['ATF', 'SEC', 'HT', 'WCC']
    datasets = ['HT']
    # cutoff = [1e-4, 1e-3, 0.005, 0.01]
    cutoff = [1e-4]
    nclusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    regularize = [True, False]
    vec_dim = [100, 200, 300, 400,500]
    sentences = [['text', 'hrow', 'cell', 'adjcell', 'hcol'],
                 ['text', 'cell'],
                 ['cell', 'hrow', 'hcol'],
                 ['text'],
                 ['cell'],
                 ['cell', 'hrow', 'hcol', 'adjcell'],
                ]

    # sentences = [['text', 'cell']]

    command = sys.argv[1]

    if command == 'create':
        conf = SparkConf().setAppName("tableEmbeddingApp") \
            .setMaster('spark://majid-Precision-Tower-3420:7077')\
            .set("spark.executor.memory", "4g") \
            .set('spark.driver.memory', '12g') \
            .set('spark.driver.maxResultSize', '2g')
            # .set("spark.executor.instances", "2")
            # .set("spark.executor.cores", "4")
        # conf = SparkConf().setAppName("tableEmbeddingApp") \
        #     .setMaster('local[*]')

        sc = SparkContext(conf=conf)
        _ = SparkSession(sc)
        sc.addPyFile('../toolkit/toolkit.py')
        sc.addPyFile('data_processing.py')
        sc.addPyFile('random_indexing.py')
        sc.addPyFile('random_indexing_wrapper.py')
        sc.addPyFile('TableEmbedding.py')
        sc.addPyFile('table_embedding.py')
        sc.addPyFile('WordCount.py')
        sc.addPyFile('TokenizeTableText.py')
        sc.addPyFile('WordEmbedding.py')
        sc.addPyFile('__init__.py')
        command2 = sys.argv[2]
        if command2 == 'tokenize':
            for d in datasets:
                for r in regularize:
                    config = {
                        'put_extractions': False,
                        'regularize_tokens': r,
                        'table_path': '*.table.tables[*]',
                        'max_product_th': 100,
                        'text_path': '*.table.html_text'
                    }
                    inpath = '{}/data/{}.etk.out'.format(dig_home, d)
                    r_text = 'reg' if r else 'noreg'
                    tok_table_path = '{}/tabvec/output/{}/tokenized/toktarr_{}'.format(dig_home, d, r_text)
                    if os.path.exists(tok_table_path):
                        print('"{}" existed .... continuing ...'.format(tok_table_path))
                    else:
                        run_tokenize_table(sc, config, inpath, tok_table_path)
                    tok_text_path = '{}/tabvec/output/{}/tokenized/text_{}'.format(dig_home, d, r_text)
                    if os.path.exists(tok_text_path):
                        print('"{}" existed .... continuing ...'.format(tok_text_path))
                    else:
                        run_tokenize_text(sc, config, inpath, tok_text_path)

        elif command2 == 'word_count':
            for d in datasets:
                for r in regularize:
                    config = {
                        'put_extractions': False,
                        'regularize_tokens': r,
                        'table_path': '*.table.tables[*]',
                        'max_product_th': 100,
                        'text_path': '*.table.html_text'
                    }
                    r_text = 'reg' if r else 'noreg'
                    for t in [True, False]:
                        t_text = 'text' if t else 'notext'
                        print('creating word counts for {} with: r={}, t={}'.format(d, r, t_text))
                        tok_table_path = '{}/tabvec/output/{}/tokenized/toktarr_{}'.format(dig_home, d, r_text)
                        tok_text_path = '{}/tabvec/output/{}/tokenized/text_{}'.format(dig_home, d, r_text)
                        wcpath = '{}/tabvec/output/{}/wc_{}_{}'.format(dig_home, d, r_text, t_text)
                        if os.path.exists(wcpath):
                            print('path existed .... continuing ...')
                            continue
                        run_word_count(sc, tok_table_path, tok_text_path, wcpath, t)
        elif command2 == 'word_occ':
            for d in datasets:
                for r in regularize:
                    config = {
                        'put_extractions': False,
                        'regularize_tokens': r,
                        'table_path': '*.table.tables[*]',
                        'max_product_th': 100,
                        'text_path': '*.table.html_text'
                    }
                    r_text = 'reg' if r else 'noreg'
                    for context in ['text', 'cell', 'adjcell', 'hrow', 'hcol']:
                        print('creating word occurrences for {} with: r={}, c={}'.format(d, r, context))
                        tok_table_path = '{}/tabvec/output/{}/tokenized/toktarr_{}'.format(dig_home, d, r_text)
                        tok_text_path = '{}/tabvec/output/{}/tokenized/text_{}'.format(dig_home, d, r_text)
                        occ_path = '{}/tabvec/output/{}/occurrences/occ_{}_{}'.format(dig_home, d, r_text, context)
                        if os.path.exists(occ_path):
                            print('path existed .... continuing ...')
                        else:
                            create_occurrences(sc, tok_table_path, tok_text_path, occ_path,
                                               context, window=2, max_product_th=100)
        elif command2 == 'word_embeddings':
            # generate word embeddings with different settings
            for d in datasets:
                doc_counts = None
                print('working on {} -------'.format(d))
                inpath = '{}/data/{}.etk.out'.format(dig_home, d)
                doc_counts = sc.textFile(inpath).count()
                print('{} documents in domain {}'.format(doc_counts, d))
                for r, s in product(regularize, sentences):
                    r_text = 'reg' if r else 'noreg'
                    if 'text' in sentences:
                        wcpath = '{}/tabvec/output/{}/wc_{}_text'.format(dig_home, d, r_text)
                    else:
                        wcpath = '{}/tabvec/output/{}/wc_{}_notext'.format(dig_home, d, r_text)

                    occ = None
                    word_counts = None
                    for c, dim in product(cutoff, vec_dim):
                        print('c={}, r={},d={},s=[{}]'.format(c, r, dim, ','.join(s)))
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
                        wepath = '{}/tabvec/output/{}/we_{}_{}_{}_{}'.format(dig_home, d, c, r_text, dim, '_'.join(s))
                        if os.path.exists(wepath):
                            print('path existed .... continuing ...')
                            continue

                        if word_counts is None:
                            print('word count and occ in process ...')
                            word_counts = sc.textFile(wcpath).map(lambda x: eval(x)). \
                                reduceByKey(lambda v1, v2: v1 + v2).collect()
                            tok_table_path = '{}/tabvec/output/{}/tokenized/toktarr_{}'.format(dig_home, d, r_text)
                            occ_path = '{}/tabvec/output/{}/occurrences/occ_{}'.format(dig_home, d, r_text)
                            occ = read_occurrences(sc, occ_path, s).persist()
                        print('creating word embeddings ...')
                        run_word_embedding(sc, config, occ, doc_counts, word_counts, wepath)
                    if occ is not None:
                        occ.unpersist()
        elif command2 == 'tables':
            # generate table vectors with different settings
            for d in datasets:
                for c, r, dim, s in product(cutoff, regularize, vec_dim, sentences):
                    print('creating table vectors with: c={}, r={},d={},s=[{}]'.format(c, r, dim, ','.join(s)))
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
                    tok_table_path = '{}/tabvec/output/{}/tokenized/toktarr_{}'.format(dig_home, d, r_text)
                    # inpath = '{}/data/{}.etk.out'.format(dig_home,d)
                    wepath = '{}/tabvec/output/{}/we_{}_{}_{}_{}'.format(dig_home,d,c,r_text,dim,'_'.join(s))
                    outpath = '{}/tabvec/output/{}/tablevecs_{}_{}_{}_{}'.format(dig_home,d, c, r_text, dim, '_'.join(s))
                    if os.path.exists(outpath):
                        print('path existed .... continuing ...')
                    else:
                        run_table_embedding(sc, config, tok_table_path, wepath, outpath)

    elif command == 'cluster':
        # generate table vectors with different settings
        for d in datasets:
            print('working on {} ....'.format(d))
            table_count = None
            for c, r, dim, s in product(cutoff, regularize, vec_dim, sentences):
                print('loading word embeddings: c={}, r={},d={},s=[{}]'.format(c, r, dim, ','.join(s)))
                r_text = 'reg' if r else 'noreg'

                wepath = '{}/tabvec/output/{}/we_{}_{}_{}_{}'.format(dig_home, d, c, r_text, dim, '_'.join(s))
                tables_path = '{}/tabvec/output/{}/tablevecs_{}_{}_{}_{}'.format(dig_home, d, c, r_text, dim,
                                                                             '_'.join(s))
                files = []
                for (dirpath, dirnames, filenames) in walk(tables_path):
                    for ff in filenames:
                        if re.match('^part-\d{5}.gz$', ff):
                            files.append(ff)
                tables = []
                for f in files:
                    with gzip.open(os.path.join(tables_path, f), "r") as infile:
                        for line in infile:
                            doc = json.loads(line)
                            tables.append(doc)
                table_vecs = [x['vec'] for x in tables]
                table_vecs_matrix = np.matrix(table_vecs)
                realse_list(table_vecs)
                print(table_vecs_matrix.shape)
                for method in cl_methods:
                    for n in num_clusters:
                        print('clustering with: m={},n={}'.format(method,n))
                        outpath = '{}/tabvec/output/{}/cl_{}_n{}_{}_{}_{}_{}.jl.tar.gz'.format(dig_home, d, method, n, c, r_text, dim,
                                                                                     '_'.join(s))
                        if os.path.exists(outpath):
                            print('path existed .... continuing ...')
                        else:
                            run_clustering(tables, table_vecs_matrix, method, n, outpath)
                del table_vecs_matrix
                realse_list(tables)
    if command == 'clear':
        for d in datasets:
            for c, r, dim, s in product(cutoff, regularize, vec_dim, sentences):
                r_text = 'reg' if r else 'noreg'
                try:
                    shutil.rmtree('{}/tabvec/output/{}/we_{}_{}_{}_{}'.format(dig_home,d, c, r_text, dim, '_'.join(s)))
                    shutil.rmtree('{}/tabvec/output/{}/tablevecs_{}_{}_{}_{}'.format(dig_home,d, c, r_text, dim, '_'.join(s)))
                except Exception as e:
                    print(e)
