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


def run_table_embedding(sc, config, input_path, word_embeddings_path, table_vectors_path):
    word_embeddings = sc.textFile(word_embeddings_path).\
        map(lambda x: json.loads(x)).map(lambda x: (x['word'], np.array(x['vector']))).collect()
    word_embeddings = dict(word_embeddings)
    # generate table vectors
    te = TableEmbedding()
    sc.textFile(input_path). \
        flatMap(lambda x: get_table_from_jpath(json.loads(x), config['table_path'], 2, 2)). \
        map(lambda x: create_tokenized_table(x, config['put_extractions'], config['regularize_tokens'])). \
        map(lambda x: put_table_vector(x, te.calc_table_vector(x['tok_tarr'], [word_embeddings], [1]))). \
        saveAsTextFile(table_vectors_path)


        #saveAsTextFile(outpath, compressionCodecClass='org.apache.hadoop.io.compress.GzipCodec')
