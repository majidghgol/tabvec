import json
from jsonpath_rw import parse
import numpy as np

from data_processing import *
from random_indexing import RandomIndexing
from random_indexing_wrapper import wrap_random_indexing_spark
from table_embedding import TableEmbedding



def put_table_vector(doc, vec):
    return json.dumps(dict(fingerprint=doc['fingerprint'],
                cdr_id=doc['cdr_id'],
                html=doc['html'],
                vec=vec.tolist()))


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
