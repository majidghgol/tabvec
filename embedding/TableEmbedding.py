import json
from jsonpath_rw import parse
import numpy as np

from data_processing import *
from random_indexing import RandomIndexing
from random_indexing_wrapper import wrap_random_indexing_spark
from table_embedding import TableEmbedding
from pyspark.mllib.feature import Word2VecModel

def vec_tarr_to_list(vectarr):
    return [[x.tolist() if x is not None else x for x in row] for row in vectarr]

def put_table_vector(doc, (vec, vec_tarr)):
    return json.dumps(dict (fingerprint=doc['fingerprint'],
                            cdr_id=doc['cdr_id'],
                            html=doc['html'],
                            tok_tarr=doc['tok_tarr'],
                            vec=vec.tolist(),
                            vec_tarr=vec_tarr_to_list(vec_tarr),
                            tarr=doc['tarr']))


def run_table_embedding(sc, config, tok_table_path, word_embeddings_path, table_vectors_path):
    word_embeddings = sc.textFile(word_embeddings_path).\
        map(lambda x: json.loads(x)).\
        filter(lambda x: sum([abs(xx) for xx in x['vector']]) > 100).\
        map(lambda x: (x['word'], np.array(x['vector'])))

    # print word_embeddings.count()
    # return
    word_embeddings = word_embeddings.collect()
    word_embeddings = dict(word_embeddings)
    # generate table vectors
    te = TableEmbedding()
    sc.textFile(tok_table_path). \
        map(lambda x: json.loads(x)).\
        map(lambda x: put_table_vector(x, te.calc_table_vector(x['tok_tarr'], [word_embeddings], [1]))). \
        saveAsTextFile(table_vectors_path, compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


        #saveAsTextFile(outpath, compressionCodecClass='org.apache.hadoop.io.compress.GzipCodec')

def run_table_embedding2(sc, tok_tables, we, table_vectors_path):
    te = TableEmbedding()
    tok_tables. \
        map(lambda x: json.loads(x)).\
        map(lambda x: put_table_vector(x, te.calc_table_vector(x['tok_tarr'], [we], [1]))). \
        saveAsTextFile(table_vectors_path, compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")